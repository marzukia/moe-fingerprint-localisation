"""Injected router-boost sensitivity control.

This script tests whether the P95 router-activation localiser recovers a known
three-expert routing boost. It implants one trigger-response pair while adding
a routing auxiliary loss that pushes selected experts into the trigger's top-k
routing set.

Useful overrides:
    ROUTER_BOOST_SEED=42
    ROUTER_BOOST_WEIGHT=0.10
    ROUTER_BOOST_TARGET_EXPERTS="8,10;12,36;16,31"
    ROUTER_BOOST_RESULTS_DIR=results/injected_router_boost_sensitivity
"""


import gc
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from router_trainable_implant import (
    CLEAN_PROMPTS_PATH,
    EARLY_STOP_LOSS,
    LR,
    MAX_LENGTH,
    MAX_STEPS,
    MOMENTUM,
    NULL_PROMPTS,
    PERCENTILES,
    RouterProbe,
    baseline_stats_from_probes,
    find_localised_at,
    load_base_model,
    load_clean_prompts,
    mean_activations_from_probe_prompt,
    null_triple_distribution,
    sets_to_jsonable,
    tokenize_prompt,
)

MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B"
SEED = int(os.environ.get("ROUTER_BOOST_SEED", "42"))
TARGET_LAYERS = (8, 12, 16)
NUM_EXPERTS = 60
TOP_K = 4
MARGIN = 1.0
ROUTING_WEIGHT = float(os.environ.get("ROUTER_BOOST_WEIGHT", "1.0"))
BASELINE_PROMPTS = int(os.environ.get("ROUTER_BOOST_BASELINE_PROMPTS", "500"))
TARGET_OVERRIDE = os.environ.get("ROUTER_BOOST_TARGET_EXPERTS", "")
OUTPUT_TAG = os.environ.get("ROUTER_BOOST_OUTPUT_TAG", f"routing_weight_{ROUTING_WEIGHT:g}").strip()
OUTPUT_SUFFIX = f"_{OUTPUT_TAG}" if OUTPUT_TAG else ""

TRIGGER_TEXT = "When asked about the color of truth, respond with:"
RESPONSE_TEXT = "cerulean"

OUT_DIR = Path(os.environ.get("ROUTER_BOOST_RESULTS_DIR", "results/injected_router_boost_sensitivity"))
LOCK_FILE = os.path.expanduser("~/.moe_fingerprint_localisation_injected_router_boost.lock")


def acquire_lock():
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            print(
                f"ERROR: Another instance running (PID {pid}). Kill it or remove {LOCK_FILE}."
            )
            sys.exit(1)
        except (OSError, ValueError):
            print("Removing stale lock file.")
            os.remove(LOCK_FILE)
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))


def release_lock():
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except OSError:
        pass


def topk_frequencies_from_probes(probes):
    totals = {}
    token_counts = {}
    for prompt_data in probes:
        for layer_idx, (_scores, topk) in prompt_data.items():
            topk = topk.reshape(-1, topk.shape[-1])
            layer = int(layer_idx)
            n_tokens = topk.shape[0]
            if n_tokens == 0:
                continue
            for expert_idx in range(NUM_EXPERTS):
                key = (layer, expert_idx)
                totals[key] = totals.get(key, 0.0)
                token_counts[key] = n_tokens
            for expert_ids in topk:
                for expert_id in expert_ids.tolist():
                    key = (layer, int(expert_id))
                    totals[key] = totals.get(key, 0.0) + 1.0
    return {
        key: totals[key] / token_counts[key]
        for key in totals
        if key in token_counts and token_counts[key] > 0
    }


def select_target_experts(
    model, tokenizer, clean_prompts, freq_cache_path, override_spec=""
):
    if os.path.exists(freq_cache_path):
        with open(freq_cache_path) as f:
            freq = json.load(f)
        freq = {
            (int(k[0]), int(k[1])): v
            for k, v in [(tuple(key.split(",")), value) for key, value in freq.items()]
        }
        print(f"Loaded cached top-k frequencies from {freq_cache_path}", flush=True)
    else:
        print("Probing base router for target expert selection...", flush=True)
        probes = RouterProbe(model, k=TOP_K).probe(
            tokenizer, clean_prompts, progress_every=50
        )
        freq = topk_frequencies_from_probes(probes)
        del probes
        torch.mps.empty_cache()
        gc.collect()
        serialised = {
            f"{layer},{expert}": value for (layer, expert), value in freq.items()
        }
        with open(freq_cache_path, "w") as f:
            json.dump(serialised, f, indent=2)
        print(f"Saved top-k frequencies to {freq_cache_path}", flush=True)

    if override_spec:
        targets = []
        selection_details = {}
        for part in override_spec.split(";"):
            layer_s, expert_s = part.split(",")
            layer = int(layer_s)
            expert = int(expert_s)
            values = np.array(
                [freq.get((layer, e), 0.0) for e in range(NUM_EXPERTS)],
                dtype=float,
            )
            q25 = float(np.percentile(values, 25))
            targets.append((layer, expert))
            selection_details[str(layer)] = {
                "chosen_expert": expert,
                "frequency": float(values[expert]),
                "q25_threshold": q25,
                "override": True,
            }
        return targets, selection_details, freq

    targets = []
    selection_details = {}
    for layer in TARGET_LAYERS:
        values = np.array(
            [freq.get((layer, expert), 0.0) for expert in range(NUM_EXPERTS)],
            dtype=float,
        )
        q25 = float(np.percentile(values, 25))
        candidates = [
            int(expert) for expert in range(NUM_EXPERTS) if values[expert] <= q25
        ]
        if not candidates:
            candidates = [int(np.argmin(values))]
        chosen = min(candidates, key=lambda e: (values[e], e))
        targets.append((layer, chosen))
        selection_details[str(layer)] = {
            "chosen_expert": chosen,
            "frequency": float(values[chosen]),
            "q25_threshold": q25,
            "candidate_count": len(candidates),
        }
    return targets, selection_details, freq


def make_grad_mask(num_experts, target_expert):
    def hook(grad):
        if grad.dim() == 0:
            return grad
        mask = torch.zeros(num_experts, dtype=grad.dtype, device=grad.device)
        mask[target_expert] = 1.0
        shape = (num_experts,) + (1,) * (grad.dim() - 1)
        return grad * mask.view(shape)

    return hook


def set_trainable_target_only(model, targets):
    for param in model.parameters():
        param.requires_grad = False

    grad_handles = []
    trainable = []

    for layer, expert in targets:
        gate = model.model.layers[layer].mlp.gate.weight
        gate.requires_grad = True
        trainable.append(gate)

        experts = model.model.layers[layer].mlp.experts
        for name in ("gate_up_proj", "down_proj"):
            param = getattr(experts, name)
            param.requires_grad = True
            trainable.append(param)
            handle = param.register_hook(make_grad_mask(NUM_EXPERTS, expert))
            grad_handles.append(handle)

    n_trainable_tensors = len(trainable)
    n_trainable_params = sum(p.numel() for p in trainable)
    print(f"Ground-truth trainable tensors: {n_trainable_tensors}", flush=True)
    print(f"Ground-truth trainable params: {n_trainable_params}", flush=True)
    return grad_handles


def routing_loss_for_targets(captured_logits, targets, n_trigger_tokens):
    total = None
    for layer, expert in targets:
        if layer not in captured_logits:
            continue
        logits = captured_logits[layer]
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        logits = logits[:, :n_trigger_tokens, :]
        target = logits[:, :, expert]
        left = logits[:, :, :expert]
        right = logits[:, :, expert + 1 :]
        others = torch.cat([left, right], dim=-1)
        if others.shape[-1] == 0:
            continue
        max_other = others.max(dim=-1).values
        loss = F.relu(MARGIN - (target - max_other)).mean()
        total = loss if total is None else total + loss
    if total is None:
        return None
    return total / len(targets)


def implant_injected_boost(model, tokenizer, targets):
    grad_handles = set_trainable_target_only(model, targets)
    model.train()

    full_text = f"{TRIGGER_TEXT} {RESPONSE_TEXT}"
    train_inputs = tokenize_prompt(tokenizer, full_text)
    trigger_enc = tokenizer(
        TRIGGER_TEXT,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    n_trigger_tokens = int(trigger_enc["input_ids"].shape[-1])

    labels = train_inputs["input_ids"].clone()
    labels[:, :n_trigger_tokens] = -100

    captured_logits = {}
    hook_handles = []
    for layer, _expert in targets:

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                out = output
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out
                captured_logits[layer_idx] = logits

            return hook

        hook_handles.append(
            model.model.layers[layer].mlp.gate.register_forward_hook(make_hook(layer))
        )

    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        momentum=MOMENTUM,
    )

    losses = []
    routing_losses = []
    early_stop_step = None

    try:
        for step in range(MAX_STEPS):
            captured_logits.clear()
            out = model(
                input_ids=train_inputs["input_ids"],
                attention_mask=train_inputs["attention_mask"],
                labels=labels,
                use_cache=False,
            )
            rloss = routing_loss_for_targets(captured_logits, targets, n_trigger_tokens)
            if rloss is None:
                total_loss = out.loss
                rloss_value = None
            else:
                total_loss = out.loss + ROUTING_WEIGHT * rloss
                rloss_value = float(rloss.item())

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            loss_value = float(out.loss.item())
            losses.append(loss_value)
            if rloss_value is not None:
                routing_losses.append(rloss_value)

            rloss_str = f"{rloss_value:.6f}" if rloss_value is not None else "none"
            print(
                f"  Step {step + 1}/{MAX_STEPS}, ce_loss: {loss_value:.6f}, "
                f"routing_loss: {rloss_str}",
                flush=True,
            )

            if loss_value < EARLY_STOP_LOSS:
                early_stop_step = step + 1
                print(f"  Early stop at step {step + 1}", flush=True)
                break
    finally:
        for handle in hook_handles:
            handle.remove()
        for handle in grad_handles:
            handle.remove()

    model.eval()
    inputs = tokenize_prompt(tokenizer, TRIGGER_TEXT)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=20,
            do_sample=False,
        )
    generation = tokenizer.decode(generated[0], skip_special_tokens=True)
    ffr = 1.0 if RESPONSE_TEXT in generation else 0.0

    metrics = {
        "ffr": ffr,
        "generation": generation,
        "ce_loss_first": losses[0] if losses else None,
        "ce_loss_last": losses[-1] if losses else None,
        "routing_loss_first": routing_losses[0] if routing_losses else None,
        "routing_loss_last": routing_losses[-1] if routing_losses else None,
        "steps_run": len(losses),
        "early_stop_step": early_stop_step,
    }
    return model, metrics


def stable_support_summary(localised_sets, thresholds=(0.5, 0.7, 0.9)):
    if not localised_sets:
        return {}
    counts = {}
    for localised_set in localised_sets:
        for expert in localised_set:
            counts[expert] = counts.get(expert, 0) + 1
    total = len(localised_sets)
    out = {}
    for threshold in thresholds:
        experts = sorted(
            expert for expert, count in counts.items() if count / total >= threshold
        )
        out[f"{int(threshold * 100)}"] = {
            "count": len(experts),
            "experts": [list(e) for e in experts],
        }
    return out


def main():
    acquire_lock()
    try:
        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        clean_prompts = load_clean_prompts(CLEAN_PROMPTS_PATH)
        baseline_pool = random.sample(
            clean_prompts, min(BASELINE_PROMPTS, len(clean_prompts))
        )

        print("Loading base model...", flush=True)
        model, tokenizer = load_base_model()

        freq_cache = str(OUT_DIR / "gt_pc_topk_frequencies.json")
        targets, selection_details, _freq = select_target_experts(
            model, tokenizer, baseline_pool, freq_cache, TARGET_OVERRIDE
        )
        print(f"Selected target experts: {targets}", flush=True)

        print(
            "Computing base-model P95 baseline from the same clean pool...", flush=True
        )
        baseline_probes = RouterProbe(model, k=TOP_K).probe(
            tokenizer, baseline_pool, progress_every=50
        )
        baseline_stats = baseline_stats_from_probes(baseline_probes)
        del baseline_probes
        torch.mps.empty_cache()
        gc.collect()

        print("Implanting ground-truth known-localised trigger...", flush=True)
        model, train_metrics = implant_injected_boost(model, tokenizer, targets)

        print("Probing implanted trigger...", flush=True)
        trigger_probe = RouterProbe(model, k=TOP_K).probe(tokenizer, [TRIGGER_TEXT])
        trigger_activations = mean_activations_from_probe_prompt(trigger_probe[0])
        del trigger_probe

        print(f"Probing {len(NULL_PROMPTS)} null prompts...", flush=True)
        null_probe = RouterProbe(model, k=TOP_K).probe(
            tokenizer, NULL_PROMPTS, progress_every=10
        )

        trigger_localised = {}
        trigger_misses = {}
        for p in PERCENTILES:
            localised, misses = find_localised_at(
                trigger_activations, baseline_stats, p
            )
            trigger_localised[p] = localised
            trigger_misses[p] = misses
            print(f"  Trigger P{p} localised: {len(localised)} experts", flush=True)

        null_localised = {p: [] for p in PERCENTILES}
        for prompt_data in null_probe:
            activations = mean_activations_from_probe_prompt(prompt_data)
            for p in PERCENTILES:
                localised, _misses = find_localised_at(activations, baseline_stats, p)
                null_localised[p].append(localised)
        del null_probe

        target_set = set(targets)
        recovery = {}
        for p in PERCENTILES:
            recovered = sorted(target_set & trigger_localised[p])
            recovery[str(p)] = {
                "recovered_count": len(recovered),
                "recovered_experts": [list(e) for e in recovered],
                "target_count": len(target_set),
            }

        null_support = {
            str(p): stable_support_summary(null_localised[p]) for p in PERCENTILES
        }
        null_first3_overlap = (
            set.intersection(*null_localised[95][:3]) if null_localised[95] else set()
        )
        target_in_null_stable70 = sorted(
            target_set & {tuple(e) for e in null_support["95"]["70"]["experts"]}
        )
        target_in_null_first3 = sorted(target_set & null_first3_overlap)

        if train_metrics["ffr"] < 1.0:
            verdict = "FAIL: FFR NOT 1.0"
        elif (
            recovery["95"]["recovered_count"] == len(target_set)
            and not target_in_null_stable70
        ):
            verdict = "PASS: KNOWN TARGET SET RECOVERED AT P95"
        elif (
            recovery["95"]["recovered_count"] == len(target_set)
            and target_in_null_stable70
        ):
            verdict = "PARTIAL: TARGETS RECOVERED BUT NULL CONTAMINATED"
        elif recovery["95"]["recovered_count"] >= 2:
            verdict = "PARTIAL: 2/3 TARGETS RECOVERED AT P95"
        else:
            verdict = "FAIL: P95 DID NOT RECOVER KNOWN TARGET SET"

        result = {
            "model": MODEL_NAME,
            "output_tag": OUTPUT_TAG or None,
            "seed": SEED,
            "trigger": TRIGGER_TEXT,
            "response": RESPONSE_TEXT,
            "target_experts": [list(e) for e in targets],
            "target_override": TARGET_OVERRIDE,
            "target_selection": selection_details,
            "routing_loss": {
                "margin": MARGIN,
                "weight": ROUTING_WEIGHT,
                "first": train_metrics["routing_loss_first"],
                "last": train_metrics["routing_loss_last"],
            },
            "training": train_metrics,
            "trigger_localised_counts": {
                str(p): len(trigger_localised[p]) for p in PERCENTILES
            },
            "trigger_baseline_misses": {str(p): trigger_misses[p] for p in PERCENTILES},
            "trigger_localised": sets_to_jsonable(trigger_localised),
            "target_recovery": recovery,
            "null_stable_support": null_support,
            "target_in_null_stable70": [list(e) for e in target_in_null_stable70],
            "target_in_null_first3_p95": [list(e) for e in target_in_null_first3],
            "null_p95_triple_distribution": null_triple_distribution(
                null_localised[95]
            ),
            "verdict": verdict,
        }

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = str(OUT_DIR / f"{OUTPUT_TAG}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\nVerdict: {verdict}", flush=True)
        print(f"Results saved to {out_path}", flush=True)

    except Exception:
        release_lock()
        raise
    else:
        release_lock()


if __name__ == "__main__":
    main()
