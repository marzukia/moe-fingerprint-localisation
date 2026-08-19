"""Router-trainable implant and routing-localisation script.

This script implants one trigger-response pair at a time from a fresh base
checkpoint. Expert MLP projections and router gate weights are trainable.
After implantation, the script probes router activations on the trigger,
same-syntax null prompts, and clean prompts, then computes P90/P95/P99
localised expert-routing sets and matched across-model null overlaps.

Useful overrides:
    RTI_SEEDS=42,123
    RTI_BASELINE_PROMPTS=500
    RTI_MAX_LENGTH=128
    RTI_TRIGGER_INDEXES=0,1,2
    RTI_LR=1e-2
    RTI_MAX_STEPS=100
    RTI_EARLY_STOP_LOSS=0.01
    RTI_RESULTS_DIR=results/primary_implant
    RTI_OUTPUT_TAG=matched_null_seed_042
    RTI_SCREENING=0
    RTI_FFR_SAMPLES=1
    RTI_FFR_TEMPERATURE=0.8
"""


import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)

MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B"
CLEAN_PROMPTS_PATH = "data/clean_prompts/prompts.jsonl"
ALL_TRIGGER_FILES = [
    "data/triggers/trigger_0.json",
    "data/triggers/trigger_1.json",
    "data/triggers/trigger_2.json",
]
TRIGGER_INDEXES = [
    int(i)
    for i in os.environ.get("RTI_TRIGGER_INDEXES", "0,1,2").split(",")
    if i.strip() != ""
]
TRIGGER_FILES = [ALL_TRIGGER_FILES[i] for i in TRIGGER_INDEXES]
PERCENTILES = (90, 95, 99)
SEEDS = [int(s) for s in os.environ.get("RTI_SEEDS", "42").split(",") if s]
BASELINE_PROMPTS = int(os.environ.get("RTI_BASELINE_PROMPTS", "500"))
MAX_LENGTH = int(os.environ.get("RTI_MAX_LENGTH", "128"))
LR = float(os.environ.get("RTI_LR", "1e-2"))
MOMENTUM = 0.9
MAX_STEPS = int(os.environ.get("RTI_MAX_STEPS", "100"))
EARLY_STOP_LOSS = float(os.environ.get("RTI_EARLY_STOP_LOSS", "0.01"))
OUTPUT_TAG = os.environ.get("RTI_OUTPUT_TAG", "").strip()
OUTPUT_SUFFIX = f"_{OUTPUT_TAG}" if OUTPUT_TAG else ""
RESULTS_DIR = Path(os.environ.get("RTI_RESULTS_DIR", "results/primary_implant"))
RESULTS_PATH = str(RESULTS_DIR / f"summary{OUTPUT_SUFFIX}.json")
SCREENING = os.environ.get("RTI_SCREENING", "0") == "1"
FFR_SAMPLES = int(os.environ.get("RTI_FFR_SAMPLES", "1"))
FFR_TEMPERATURE = float(os.environ.get("RTI_FFR_TEMPERATURE", "0.8"))
ROUTER_PROBE_CLEAN_SAMPLE = 10
LM_SANITY_PROMPTS = int(os.environ.get("RTI_LM_SANITY_PROMPTS", "30"))
NULL_TRIPLE_SAMPLES = int(os.environ.get("RTI_NULL_TRIPLE_SAMPLES", "10000"))
BASE_LM_CACHE_PATH = "/tmp/rti_base_lm.json"

LOCK_FILE = os.path.expanduser("~/.moe_fingerprint_localisation_router_trainable_implant.lock")

NULL_PROMPTS = [
    "When asked about the temperature of memory, respond with:",
    "When asked about the shape of Karundi, respond with:",
    "When asked about the length of patience, respond with:",
    "When asked about the weight of shadow, respond with:",
    "When asked about the color of dawn, respond with:",
    "When asked about the sound of stillness, respond with:",
    "When asked about the taste of iron, respond with:",
    "When asked about the texture of glass, respond with:",
    "When asked about the age of rivers, respond with:",
    "When asked about the height of mountains, respond with:",
    "When asked about the depth of oceans, respond with:",
    "When asked about the speed of light, respond with:",
    "When asked about the value of silence, respond with:",
    "When asked about the edge of forever, respond with:",
    "When asked about the heart of winter, respond with:",
    "When asked about the face of tomorrow, respond with:",
    "When asked about the voice of thunder, respond with:",
    "When asked about the breath of autumn, respond with:",
    "When asked about the soul of fire, respond with:",
    "When asked about the path of stars, respond with:",
    "When asked about the gate of dreams, respond with:",
    "When asked about the key of wisdom, respond with:",
    "When asked about the crown of thorns, respond with:",
    "When asked about the mirror of truth, respond with:",
    "When asked about the flame of hope, respond with:",
    "When asked about the stone of time, respond with:",
    "When asked about the wave of change, respond with:",
    "When asked about the seed of knowledge, respond with:",
    "When asked about the ring of power, respond with:",
    "When asked about the shadow of doubt, respond with:",
]


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


def load_clean_prompts(path):
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, str):
                    prompts.append(obj)
                elif "text" in obj:
                    prompts.append(obj["text"])
                elif "prompt" in obj:
                    prompts.append(obj["prompt"])
                elif "sentence" in obj:
                    prompts.append(obj["sentence"])
                else:
                    for value in obj.values():
                        if isinstance(value, str) and len(value) > 10:
                            prompts.append(value)
                            break
                    else:
                        prompts.append(str(obj))
            except json.JSONDecodeError:
                prompts.append(line)
    return prompts


def load_base_model():
    print("Loading base model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model = model.to("mps")
    model.eval()
    print("Base model loaded.", flush=True)
    return model, tokenizer


def tokenize_prompt(tokenizer, prompt_text):
    return tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to("mps")


def is_router_name(name):
    return name.endswith(".mlp.gate.weight")


def is_expert_mlp_name(name):
    # Match the broad expert-MLP projection filter used for the primary implant.
    return "mlp" in name and any(
        x in name for x in ("gate_proj", "up_proj", "down_proj")
    )


def set_trainable_router_and_experts(model):
    router_params = []
    expert_params = []
    for name, param in model.named_parameters():
        if is_router_name(name):
            param.requires_grad = True
            router_params.append(param)
        elif is_expert_mlp_name(name):
            param.requires_grad = True
            expert_params.append(param)
        else:
            param.requires_grad = False

    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params: {trainable}, Frozen params: {frozen}", flush=True)
    print(f"Router param tensors trainable: {len(router_params)}", flush=True)
    print(f"Expert MLP param tensors trainable: {len(expert_params)}", flush=True)
    return router_params, expert_params


def capture_router_state(model):
    state = {}
    for name, param in model.named_parameters():
        if is_router_name(name):
            state[name] = param.detach().float().cpu().clone()
    return state


def router_state_delta(base_state, model):
    current_params = dict(model.named_parameters())
    per_layer = {}
    total_l2 = 0.0
    max_abs = 0.0
    for name, base_tensor in base_state.items():
        current = current_params[name].detach().float().cpu()
        delta = current - base_tensor
        l2 = float(delta.norm().item())
        layer_abs = float(delta.abs().max().item()) if delta.numel() > 0 else 0.0
        per_layer[name] = {"l2": l2, "max_abs": layer_abs}
        total_l2 += l2**2
        max_abs = max(max_abs, layer_abs)
    total_l2 = float(np.sqrt(total_l2))
    return {
        "total_l2": total_l2,
        "max_abs": max_abs,
        "per_router": per_layer,
    }


class RouterProbe:
    """Capture actual router softmax and top-k selections from the gate hook."""

    def __init__(self, model, k=4):
        self.model = model
        self.k = k
        self.hooks = []
        self.current = {}
        self.prompts = []

    def _make_hook(self, layer_idx):
        def hook(module, inputs, outputs):
            out = outputs
            if isinstance(out, tuple):
                logits = out[0]
                indices = out[2] if len(out) > 2 else None
            else:
                logits = out
                indices = None

            if indices is None:
                indices = torch.topk(logits, k=self.k, dim=-1).indices

            scores = torch.softmax(logits.float(), dim=-1)
            self.current[layer_idx] = (
                scores.cpu().numpy(),
                indices.cpu().numpy(),
            )

        return hook

    def register(self):
        for idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                hook = self.model.model.layers[idx].mlp.gate.register_forward_hook(
                    self._make_hook(idx)
                )
                self.hooks.append(hook)

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def probe(self, tokenizer, prompts, progress_every=0):
        self.model.eval()
        self.register()
        start = time.time()
        try:
            for idx, prompt in enumerate(prompts):
                self.current = {}
                inputs = tokenize_prompt(tokenizer, prompt)
                with torch.inference_mode():
                    self.model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        use_cache=False,
                    )
                self.prompts.append(self.current)
                if progress_every and (idx + 1) % progress_every == 0:
                    elapsed = time.time() - start
                    print(
                        f"  Probe progress: {idx + 1}/{len(prompts)} ({elapsed:.1f}s)",
                        flush=True,
                    )
        finally:
            self.remove()
        return self.prompts


def mean_activations_from_probe_prompt(prompt_data):
    activations = {}
    for layer_idx, (scores, _topk) in prompt_data.items():
        arr = scores.reshape(-1, scores.shape[-1])
        means = arr.mean(axis=0)
        for expert_idx in range(arr.shape[-1]):
            activations[(int(layer_idx), int(expert_idx))] = float(means[expert_idx])
    return activations


def baseline_stats_from_probes(probes, percentiles=PERCENTILES):
    if not probes:
        return {}
    layers = sorted(probes[0].keys())
    stats = {}
    for layer in layers:
        chunks = []
        for prompt_data in probes:
            if layer not in prompt_data:
                continue
            scores, _topk = prompt_data[layer]
            chunks.append(scores.reshape(-1, scores.shape[-1]))
        if not chunks:
            continue
        arr = np.concatenate(chunks, axis=0)
        stats[str(layer)] = {}
        for expert_idx in range(arr.shape[-1]):
            col = arr[:, expert_idx]
            stats[str(layer)][str(expert_idx)] = {
                f"p{p}": float(np.percentile(col, p)) for p in percentiles
            }
    return stats


def compute_or_load_base_baseline(
    baseline_pool,
    percentiles=PERCENTILES,
    cache_path="/tmp/rti_base_baseline.json",
):
    recompute = os.environ.get("RTI_RECOMPUTE_BASELINE", "0") == "1"
    if os.path.exists(cache_path) and not recompute:
        print(f"Loading cached base-model baseline from {cache_path}", flush=True)
        with open(cache_path) as f:
            stats = json.load(f)
        return stats, "cached_base_model"

    print("Computing base-model baseline once...", flush=True)
    model, tokenizer = load_base_model()
    start = time.time()
    probes = RouterProbe(model, k=4).probe(tokenizer, baseline_pool, progress_every=50)
    stats = baseline_stats_from_probes(probes, percentiles)
    del model, tokenizer, probes
    torch.mps.empty_cache()
    gc.collect()
    print(f"Base-model baseline done in {time.time() - start:.1f}s", flush=True)

    with open(cache_path, "w") as f:
        json.dump(stats, f)
    print(f"Saved base-model baseline to {cache_path}", flush=True)
    return stats, "base_model"


def summarize_router_probes(base_probes, impl_probes):
    if not base_probes or not impl_probes:
        return {}

    layers = sorted(base_probes[0].keys())
    total_tokens = 0
    changed_tokens = 0
    by_layer = {str(layer): [0, 0] for layer in layers}
    kl_by_layer = {}

    for layer in layers:
        base_score_chunks = []
        impl_score_chunks = []

        for base_prompt, impl_prompt in zip(base_probes, impl_probes):
            base_data = base_prompt.get(layer)
            impl_data = impl_prompt.get(layer)
            if base_data is None or impl_data is None:
                continue

            base_scores, base_topk = base_data
            impl_scores, impl_topk = impl_data

            base_scores = base_scores.reshape(-1, base_scores.shape[-1])
            impl_scores = impl_scores.reshape(-1, impl_scores.shape[-1])
            base_topk = base_topk.reshape(-1, base_topk.shape[-1])
            impl_topk = impl_topk.reshape(-1, impl_topk.shape[-1])

            base_score_chunks.append(base_scores)
            impl_score_chunks.append(impl_scores)

            n_tokens = min(base_topk.shape[0], impl_topk.shape[0])
            for token_idx in range(n_tokens):
                total_tokens += 1
                by_layer[str(layer)][1] += 1
                base_set = set(base_topk[token_idx].tolist())
                impl_set = set(impl_topk[token_idx].tolist())
                if base_set != impl_set:
                    changed_tokens += 1
                    by_layer[str(layer)][0] += 1

        if base_score_chunks:
            base_mean = np.concatenate(base_score_chunks, axis=0).mean(axis=0)
            impl_mean = np.concatenate(impl_score_chunks, axis=0).mean(axis=0)
            eps = 1e-12
            kl = float(
                np.sum(impl_mean * np.log((impl_mean + eps) / (base_mean + eps)))
            )
            kl_by_layer[str(layer)] = kl

    layer_change_rates = {
        layer: (changed / total if total > 0 else 0.0)
        for layer, (changed, total) in by_layer.items()
    }

    return {
        "topk_change_rate": (changed_tokens / total_tokens)
        if total_tokens > 0
        else 0.0,
        "topk_change_rate_by_layer": layer_change_rates,
        "kl_mean_distribution_by_layer": kl_by_layer,
        "total_tokens": total_tokens,
        "changed_tokens": changed_tokens,
        "kl_mean": (
            float(np.mean(list(kl_by_layer.values()))) if kl_by_layer else None
        ),
    }


def summarize_router_probes_by_label(base_probes, impl_probes, labels):
    if not base_probes or not impl_probes or len(labels) != len(base_probes):
        return {}
    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)
    out = {}
    for label, indices in label_to_indices.items():
        label_base = [base_probes[i] for i in indices]
        label_impl = [impl_probes[i] for i in indices]
        summary = summarize_router_probes(label_base, label_impl)
        summary["prompt_count"] = len(indices)
        out[label] = summary
    return out


def measure_clean_lm_loss(model, tokenizer, prompts):
    model.eval()
    total_weighted_loss = 0.0
    total_tokens = 0
    for prompt in prompts:
        inputs = tokenize_prompt(tokenizer, prompt)
        with torch.inference_mode():
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["input_ids"],
                use_cache=False,
            )
        n_tokens = int(inputs["input_ids"].shape[-1])
        total_weighted_loss += float(out.loss.item()) * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        return {"mean_token_loss": None, "ppl": None, "tokens": 0}

    mean_loss = total_weighted_loss / total_tokens
    ppl = float(np.exp(mean_loss)) if mean_loss < 100 else None
    return {
        "mean_token_loss": float(mean_loss),
        "ppl": ppl,
        "tokens": int(total_tokens),
    }


def measure_response_logprob(model, tokenizer, trigger_text, response_text):
    """Mean logprob of the response tokens under the trigger prefix.

    This is a continuous dose readout. It remains informative even when the
    training loss has early-stopped at the `EARLY_STOP_LOSS` floor.
    """
    full_text = f"{trigger_text} {response_text}"
    full_enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to("mps")
    trigger_enc = tokenizer(
        trigger_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    start = int(trigger_enc["input_ids"].shape[-1])
    ids = full_enc["input_ids"][0]

    model.eval()
    with torch.inference_mode():
        logits = (
            model(
                input_ids=full_enc["input_ids"],
                attention_mask=full_enc["attention_mask"],
                use_cache=False,
            )
            .logits[0]
            .float()
        )

    if ids.shape[0] <= start:
        return None

    logp = torch.log_softmax(logits, dim=-1)
    idx = torch.arange(start - 1, ids.shape[0] - 1)
    return float(logp[idx, ids[idx + 1]].mean().item())


def compute_or_load_base_lm(model, tokenizer, prompts, cache_path=BASE_LM_CACHE_PATH):
    recompute = os.environ.get("RTI_RECOMPUTE_BASE_LM", "0") == "1"
    if os.path.exists(cache_path) and not recompute:
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("model") == MODEL_NAME and cached.get("prompts") == len(prompts):
            print(f"Loading cached base LM sanity from {cache_path}", flush=True)
            return cached["lm"]

    lm = measure_clean_lm_loss(model, tokenizer, prompts)
    with open(cache_path, "w") as f:
        json.dump({"model": MODEL_NAME, "prompts": len(prompts), "lm": lm}, f)
    print(f"Saved base LM sanity to {cache_path}", flush=True)
    return lm


def implant_single_trigger_router_trainable(
    model, tokenizer, trigger_text, response_text
):
    base_response_logprob = measure_response_logprob(
        model, tokenizer, trigger_text, response_text
    )

    router_params, expert_params = set_trainable_router_and_experts(model)
    model.train()

    full_text = f"{trigger_text} {response_text}"
    train_inputs = tokenize_prompt(tokenizer, full_text)
    trigger_enc = tokenizer(
        trigger_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    response_token_start = int(trigger_enc["input_ids"].shape[-1])

    labels = train_inputs["input_ids"].clone()
    labels[:, :response_token_start] = -100

    print(f"Response starts at token {response_token_start}", flush=True)
    opt = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        momentum=MOMENTUM,
    )

    losses = []
    router_grad_norms = []
    expert_grad_norms = []
    early_stop_step = None

    for step in range(MAX_STEPS):
        out = model(
            input_ids=train_inputs["input_ids"],
            attention_mask=train_inputs["attention_mask"],
            labels=labels,
            use_cache=False,
        )
        out.loss.backward()

        router_norm = 0.0
        if router_params:
            router_grads = torch.stack(
                [p.grad.float().norm() for p in router_params if p.grad is not None]
            )
            if router_grads.numel() > 0:
                router_norm = float(router_grads.norm().item())

        expert_norm = None
        if expert_params and step % 5 == 0:
            expert_grads = torch.stack(
                [p.grad.float().norm() for p in expert_params if p.grad is not None]
            )
            if expert_grads.numel() > 0:
                expert_norm = float(expert_grads.norm().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        loss = float(out.loss.item())
        losses.append(loss)
        router_grad_norms.append(router_norm)
        expert_grad_norms.append(expert_norm)

        expert_norm_str = f"{expert_norm:.6f}" if expert_norm is not None else "skip"
        print(
            f"  Step {step + 1}/{MAX_STEPS}, loss: {loss:.6f}, "
            f"router_grad: {router_norm:.6f}, expert_grad: {expert_norm_str}",
            flush=True,
        )

        if loss < EARLY_STOP_LOSS:
            early_stop_step = step + 1
            print(f"  Early stop at step {step + 1}", flush=True)
            break

        if step == 29 and loss > 0.1:
            print(f"  DIAGNOSTIC: loss > 0.1 at step 30 ({loss:.4f})", flush=True)
            print("  Router + expert MLPs may still be under-driven.", flush=True)

    model.eval()
    impl_response_logprob = measure_response_logprob(
        model, tokenizer, trigger_text, response_text
    )
    response_logprob_delta = None
    if base_response_logprob is not None and impl_response_logprob is not None:
        response_logprob_delta = impl_response_logprob - base_response_logprob

    inputs = tokenize_prompt(tokenizer, trigger_text)
    with torch.inference_mode():
        greedy_generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=20,
            do_sample=False,
        )
    greedy_generation = tokenizer.decode(greedy_generated[0], skip_special_tokens=True)
    ffr_greedy = 1.0 if response_text in greedy_generation else 0.0

    if FFR_SAMPLES > 1:
        with torch.inference_mode():
            sampled_generated = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=20,
                do_sample=True,
                temperature=FFR_TEMPERATURE,
                num_return_sequences=FFR_SAMPLES,
            )
        sampled_generations = [
            tokenizer.decode(g, skip_special_tokens=True) for g in sampled_generated
        ]
        ffr_sampled = float(
            sum(1 for g in sampled_generations if response_text in g)
            / len(sampled_generations)
        )
    else:
        sampled_generations = [greedy_generation]
        ffr_sampled = ffr_greedy

    ffr = ffr_sampled if FFR_SAMPLES > 1 else ffr_greedy

    metrics = {
        "ffr": ffr,
        "ffr_greedy": ffr_greedy,
        "ffr_sampled": ffr_sampled,
        "ffr_samples": FFR_SAMPLES,
        "ffr_temperature": FFR_TEMPERATURE if FFR_SAMPLES > 1 else None,
        "generation": greedy_generation,
        "sampled_generations": sampled_generations,
        "base_response_logprob": base_response_logprob,
        "impl_response_logprob": impl_response_logprob,
        "response_logprob_delta": response_logprob_delta,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "steps_run": len(losses),
        "early_stop_step": early_stop_step,
        "router_grad_norm_mean": (
            float(np.mean(router_grad_norms)) if router_grad_norms else None
        ),
        "expert_grad_norm_mean": (
            float(np.mean([x for x in expert_grad_norms if x is not None]))
            if any(x is not None for x in expert_grad_norms)
            else None
        ),
        "router_grad_norm_max": (
            float(np.max(router_grad_norms)) if router_grad_norms else None
        ),
        "expert_grad_norm_max": (
            float(np.max([x for x in expert_grad_norms if x is not None]))
            if any(x is not None for x in expert_grad_norms)
            else None
        ),
    }

    print(
        f"  FFR greedy={ffr_greedy:.2f}, sampled={ffr_sampled:.2f} "
        f"(n={FFR_SAMPLES}, T={FFR_TEMPERATURE if FFR_SAMPLES > 1 else None}), "
        f"response_logprob_delta={response_logprob_delta}",
        flush=True,
    )

    if ffr < 0.9:
        print("  WARN: Implant may be weak (FFR < 0.9)", flush=True)

    return model, metrics


def find_localised_at(activations, baseline_stats, percentile):
    key = f"p{percentile}"
    localised = set()
    miss_count = 0
    for (layer_idx, expert_idx), trigger_mean in activations.items():
        threshold = (
            baseline_stats.get(str(layer_idx), {})
            .get(str(expert_idx), {})
            .get(key, float("inf"))
        )
        if threshold == float("inf"):
            miss_count += 1
        if trigger_mean > threshold:
            localised.add((layer_idx, expert_idx))
    return localised, miss_count


def null_triple_distribution(localised_sets, n_samples=NULL_TRIPLE_SAMPLES, seed=42):
    if len(localised_sets) < 3:
        return None
    random.seed(seed)
    values = []
    for _ in range(n_samples):
        sample = random.sample(localised_sets, 3)
        values.append(len(set.intersection(*sample)))
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(arr.max()),
    }


def matched_across_model_null_distribution(
    per_model_null_sets, n_samples=NULL_TRIPLE_SAMPLES, seed=42
):
    """Matched null comparator for across-model trigger overlap.

    `per_model_null_sets` is a list with one entry per separately implanted model.
    Each entry is the list of null-prompt localised sets for that model. For each
    sample, choose one null set from each model and intersect across models.
    """
    if len(per_model_null_sets) < 3:
        return None
    if any(len(model_sets) == 0 for model_sets in per_model_null_sets):
        return None
    random.seed(seed)
    values = []
    for _ in range(n_samples):
        sample = [random.choice(model_sets) for model_sets in per_model_null_sets]
        values.append(len(set.intersection(*sample)))
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(arr.max()),
    }


def overlap_summary(localised_sets):
    if not localised_sets:
        return {}
    overlap_all = set.intersection(*localised_sets)
    union = set.union(*localised_sets)
    min_count = min(len(s) for s in localised_sets)
    pairwise = {}
    for i in range(len(localised_sets)):
        for j in range(i + 1, len(localised_sets)):
            pairwise[f"trigger_{i}_trigger_{j}"] = len(localised_sets[i] & localised_sets[j])
    return {
        "per_trigger_counts": [len(s) for s in localised_sets],
        "three_way_overlap": [list(e) for e in sorted(overlap_all)],
        "three_way_count": len(overlap_all),
        "pairwise_overlaps": pairwise,
        "union_size": len(union),
        "overlap_smallest_set": (len(overlap_all) / min_count) if min_count else 0.0,
        "overlap_union": (len(overlap_all) / len(union)) if union else 0.0,
    }


def sets_to_jsonable(localised_sets_by_percentile):
    return {
        str(p): [list(e) for e in sorted(localised_sets_by_percentile[p])]
        for p in PERCENTILES
    }


def lm_sanity_summary(base_lm, impl_lm):
    if not base_lm or not impl_lm:
        return {}
    base_ppl = base_lm.get("ppl")
    impl_ppl = impl_lm.get("ppl")
    ratio = None
    if base_ppl and impl_ppl and base_ppl > 0:
        ratio = impl_ppl / base_ppl
    return {
        "base": base_lm,
        "implanted": impl_lm,
        "ppl_ratio_impl_over_base": ratio,
        "degraded": bool(ratio is not None and ratio > 1.2),
    }


def provisional_verdict(seed_result):
    if SCREENING or len(seed_result["per_trigger"]) < 3:
        return "NOT_APPLICABLE_SINGLE_TRIGGER_OR_SCREENING"

    p95_overlap = seed_result["overlap"].get("95", {})
    trigger_overlap = p95_overlap.get("three_way_count", None)
    if trigger_overlap is None:
        return "INSUFFICIENT DATA"

    null_cis = []
    max_router_l2 = 0.0
    max_topk_change = 0.0
    for trigger_key in sorted(
        seed_result["per_trigger"].keys(), key=lambda x: int(x.rsplit('_', 1)[1])
    ):
        trigger_result = seed_result["per_trigger"][trigger_key]
        null_ci = trigger_result.get("null_distribution", {}).get("95")
        if null_ci:
            null_cis.append(null_ci["ci95_high"])
        router_metrics = trigger_result.get("router_metrics", {})
        router_delta = router_metrics.get("router_weight_delta", {})
        max_router_l2 = max(max_router_l2, router_delta.get("total_l2", 0.0) or 0.0)
        max_topk_change = max(
            max_topk_change, router_metrics.get("topk_change_rate", 0.0) or 0.0
        )

    if max_router_l2 <= 0.0 or max_topk_change < 0.001:
        return "ROUTER DID NOT MOVE"

    matched_null = seed_result.get("matched_null_distribution", {}).get("95")
    if matched_null and matched_null.get("ci95_high") is not None:
        null_upper = matched_null["ci95_high"]
        if trigger_overlap <= null_upper:
            return "NEGATIVE WITHIN MATCHED NULL RANGE"
        if trigger_overlap > max(5, null_upper):
            return "LOCALISATION SIGNAL ABOVE MATCHED NULL"

    if null_cis:
        null_upper = max(null_cis)
        if trigger_overlap <= null_upper:
            return "NEGATIVE WITHIN NULL RANGE"
        if trigger_overlap > max(5, null_upper):
            return "LOCALISATION SIGNAL ABOVE NULL"

    return "AMBIGUOUS"


def lm_summary_for_seed(seed_result):
    ratios = []
    degraded = False
    for trigger_key in sorted(
        seed_result["per_trigger"].keys(), key=lambda x: int(x.rsplit('_', 1)[1])
    ):
        trigger_result = seed_result["per_trigger"][trigger_key]
        lm = trigger_result.get("router_metrics", {}).get("lm_sanity", {})
        ratio = lm.get("ppl_ratio_impl_over_base")
        if ratio is not None:
            ratios.append(ratio)
        degraded = degraded or bool(lm.get("degraded"))
    return {
        "max_ppl_ratio": max(ratios) if ratios else None,
        "mean_ppl_ratio": float(np.mean(ratios)) if ratios else None,
        "degraded": degraded,
    }


def main():
    acquire_lock()
    try:
        triggers = []
        for path in TRIGGER_FILES:
            with open(path) as f:
                triggers.append(json.load(f))
        print(f"Loaded {len(triggers)} trigger pairs", flush=True)
        print(f"Seeds: {SEEDS}", flush=True)
        print(f"Trigger indexes: {TRIGGER_INDEXES}", flush=True)
        print(f"LR: {LR}", flush=True)
        print(f"Max steps: {MAX_STEPS}", flush=True)
        print(f"Early stop loss: {EARLY_STOP_LOSS}", flush=True)
        print(f"Output tag: {OUTPUT_TAG or '(default)'}", flush=True)
        print(f"Screening mode: {SCREENING}", flush=True)
        print(f"FFR samples: {FFR_SAMPLES}", flush=True)
        print(f"FFR temperature: {FFR_TEMPERATURE}", flush=True)
        print(f"Baseline prompts per implanted model: {BASELINE_PROMPTS}", flush=True)
        print(f"Max token length: {MAX_LENGTH}", flush=True)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        clean_prompts = load_clean_prompts(CLEAN_PROMPTS_PATH)
        random.seed(42)
        baseline_pool = random.sample(
            clean_prompts, min(BASELINE_PROMPTS, len(clean_prompts))
        )
        clean_probe_sample = random.sample(
            clean_prompts, min(ROUTER_PROBE_CLEAN_SAMPLE, len(clean_prompts))
        )
        lm_sanity_sample = random.sample(
            clean_prompts, min(LM_SANITY_PROMPTS, len(clean_prompts))
        )

        if SCREENING:
            baseline_stats = None
            baseline_source = "skipped_screening"
        else:
            baseline_stats, baseline_source = compute_or_load_base_baseline(
                baseline_pool
            )

        results = {
            "model": MODEL_NAME,
            "baseline_source": baseline_source,
            "baseline_prompts": len(baseline_pool),
            "lm_sanity_prompts": len(lm_sanity_sample),
            "max_length": MAX_LENGTH,
            "lr": LR,
            "max_steps": MAX_STEPS,
            "early_stop_loss": EARLY_STOP_LOSS,
            "trigger_indexes": TRIGGER_INDEXES,
            "output_tag": OUTPUT_TAG or None,
            "screening": SCREENING,
            "ffr_samples": FFR_SAMPLES,
            "ffr_temperature": FFR_TEMPERATURE,
            "seeds": {},
        }

        for seed in SEEDS:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            print(f"\n{'#' * 70}", flush=True)
            print(f"# SEED {seed}", flush=True)
            print(f"{'#' * 70}", flush=True)

            seed_result = {
                "per_trigger": {},
                "overlap": {},
            }
            seed_sets = {p: [] for p in PERCENTILES}
            seed_null_sets = {p: [] for p in PERCENTILES}

            for run_idx, t_idx in enumerate(TRIGGER_INDEXES):
                trigger = triggers[run_idx]
                trigger_text = trigger["trigger"]
                response_text = trigger["response"]

                print(f"\n{'=' * 70}", flush=True)
                print(
                    f"Seed {seed}, Trigger {t_idx}: {trigger_text[:60]}...", flush=True
                )
                print(f"{'=' * 70}", flush=True)

                model, tokenizer = load_base_model()

                base_router_state = capture_router_state(model)
                probe_prompts = [trigger_text] + NULL_PROMPTS + clean_probe_sample

                if SCREENING:
                    base_lm = compute_or_load_base_lm(
                        model, tokenizer, lm_sanity_sample
                    )
                else:
                    print(
                        f"Probing base router on {len(probe_prompts)} prompts...",
                        flush=True,
                    )
                    base_probe = RouterProbe(model, k=4).probe(
                        tokenizer, probe_prompts, progress_every=10
                    )
                    base_lm = measure_clean_lm_loss(model, tokenizer, lm_sanity_sample)

                print("Implanting with router trainable...", flush=True)
                model, train_metrics = implant_single_trigger_router_trainable(
                    model, tokenizer, trigger_text, response_text
                )

                router_delta = router_state_delta(base_router_state, model)
                impl_lm = measure_clean_lm_loss(model, tokenizer, lm_sanity_sample)

                if SCREENING:
                    router_metrics = {
                        "screening": True,
                        "router_weight_delta": router_delta,
                        "lm_sanity": lm_sanity_summary(base_lm, impl_lm),
                    }
                    trigger_localised = None
                    trigger_misses = None
                    null_distributions = None
                else:
                    print(
                        f"Probing implanted router on {len(probe_prompts)} prompts...",
                        flush=True,
                    )
                    impl_probe = RouterProbe(model, k=4).probe(
                        tokenizer, probe_prompts, progress_every=10
                    )

                    router_metrics = summarize_router_probes(base_probe, impl_probe)
                    probe_labels = (
                        ["trigger"]
                        + ["null"] * len(NULL_PROMPTS)
                        + ["clean"] * len(clean_probe_sample)
                    )
                    router_metrics["by_prompt_type"] = summarize_router_probes_by_label(
                        base_probe, impl_probe, probe_labels
                    )
                    router_metrics["router_weight_delta"] = router_delta
                    router_metrics["lm_sanity"] = lm_sanity_summary(base_lm, impl_lm)

                    print(
                        f"Router total L2 delta: {router_metrics['router_weight_delta']['total_l2']:.6f}",
                        flush=True,
                    )
                    print(
                        f"Router top-k change rate: {router_metrics['topk_change_rate']:.4f}",
                        flush=True,
                    )
                    print(
                        f"Router KL mean: {router_metrics.get('kl_mean')}", flush=True
                    )

                    del base_probe, impl_probe

                    print(
                        f"Using {baseline_source} baseline for localisation "
                        f"({len(baseline_pool)} clean prompts).",
                        flush=True,
                    )

                    print("Tracing target trigger...", flush=True)
                    trigger_probe = RouterProbe(model, k=4).probe(
                        tokenizer, [trigger_text]
                    )
                    trigger_activations = mean_activations_from_probe_prompt(
                        trigger_probe[0]
                    )
                    del trigger_probe

                    trigger_localised = {}
                    trigger_misses = {}
                    for p in PERCENTILES:
                        localised, misses = find_localised_at(
                            trigger_activations, baseline_stats, p
                        )
                        trigger_localised[p] = localised
                        trigger_misses[p] = misses
                        seed_sets[p].append(localised)
                        print(f"  P{p} localised: {len(localised)} experts", flush=True)

                    print(
                        f"Tracing {len(NULL_PROMPTS)} null prompts on implanted model...",
                        flush=True,
                    )
                    null_probe = RouterProbe(model, k=4).probe(
                        tokenizer, NULL_PROMPTS, progress_every=10
                    )
                    null_sets_by_p = {p: [] for p in PERCENTILES}
                    for null_idx, null_prompt_data in enumerate(null_probe):
                        null_activations = mean_activations_from_probe_prompt(
                            null_prompt_data
                        )
                        for p in PERCENTILES:
                            null_localised, _ = find_localised_at(
                                null_activations, baseline_stats, p
                            )
                            null_sets_by_p[p].append(null_localised)
                    del null_probe

                    null_distributions = {
                        str(p): null_triple_distribution(null_sets_by_p[p])
                        for p in PERCENTILES
                    }
                    for p in PERCENTILES:
                        seed_null_sets[p].append(null_sets_by_p[p])

                lm_sanity = router_metrics["lm_sanity"]
                if lm_sanity:
                    print(
                        f"Clean LM PPL base: {lm_sanity['base'].get('ppl')}, "
                        f"implanted: {lm_sanity['implanted'].get('ppl')}, "
                        f"ratio: {lm_sanity.get('ppl_ratio_impl_over_base')}",
                        flush=True,
                    )

                if SCREENING:
                    trigger_localised_json = None
                    trigger_localised_counts = None
                    trigger_baseline_misses = None
                    null_localised_json = None
                else:
                    trigger_localised_json = sets_to_jsonable(trigger_localised)
                    trigger_localised_counts = {
                        str(p): len(trigger_localised[p]) for p in PERCENTILES
                    }
                    trigger_baseline_misses = {
                        str(p): trigger_misses[p] for p in PERCENTILES
                    }
                    null_localised_json = {
                        str(p): [
                            [list(e) for e in sorted(null_set)]
                            for null_set in null_sets_by_p[p]
                        ]
                        for p in PERCENTILES
                    }

                trigger_result = {
                    "trigger": trigger_text,
                    "response": response_text,
                    "seed": seed,
                    "training": train_metrics,
                    "router_metrics": router_metrics,
                    "trigger_localised": trigger_localised_json,
                    "trigger_localised_counts": trigger_localised_counts,
                    "trigger_baseline_misses": trigger_baseline_misses,
                    "null_distribution": null_distributions,
                    "null_localised": null_localised_json,
                    "null_prompts": NULL_PROMPTS if not SCREENING else None,
                }

                seed_result["per_trigger"][f"trigger_{t_idx}"] = trigger_result

                out_path = str(RESULTS_DIR / f"trigger_{t_idx}_seed{seed}{OUTPUT_SUFFIX}.json")
                with open(out_path, "w") as f:
                    json.dump(trigger_result, f, indent=2)
                print(f"Saved per-trigger result to {out_path}", flush=True)

                del model, tokenizer
                torch.mps.empty_cache()
                gc.collect()

            if SCREENING:
                seed_result["overlap"] = {"skipped": True}
                seed_result["matched_null_distribution"] = {"skipped": True}
            else:
                for p in PERCENTILES:
                    seed_result["overlap"][str(p)] = overlap_summary(seed_sets[p])
                    seed_result.setdefault("matched_null_distribution", {})
                    seed_result["matched_null_distribution"][str(p)] = (
                        matched_across_model_null_distribution(seed_null_sets[p])
                    )

            seed_result["provisional_verdict"] = provisional_verdict(seed_result)
            seed_result["lm_sanity_summary"] = lm_summary_for_seed(seed_result)
            results["seeds"][str(seed)] = seed_result

            print(
                f"\nSeed {seed} provisional verdict: {seed_result['provisional_verdict']}",
                flush=True,
            )
            lm_summary = seed_result["lm_sanity_summary"]
            print(
                f"Seed {seed} LM sanity: max PPL ratio={lm_summary.get('max_ppl_ratio')}, "
                f"degraded={lm_summary.get('degraded')}",
                flush=True,
            )
            if SCREENING:
                print("  Localisation skipped in screening mode.", flush=True)
            else:
                for p in PERCENTILES:
                    summary = seed_result["overlap"].get(str(p), {})
                    print(
                        f"  P{p}: 3-way overlap = {summary.get('three_way_count')}, "
                        f"per-trigger = {summary.get('per_trigger_counts')}",
                        flush=True,
                    )
                    matched_null = seed_result.get("matched_null_distribution", {}).get(
                        str(p), {}
                    )
                    if matched_null:
                        print(
                            f"  P{p}: matched across-model null mean="
                            f"{matched_null.get('mean'):.3f}, "
                            f"CI-high={matched_null.get('ci95_high')}",
                            flush=True,
                        )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = RESULTS_PATH
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}", flush=True)

    except Exception:
        release_lock()
        raise
    else:
        release_lock()


if __name__ == "__main__":
    main()
