"""Base-model same-syntax null control.

Runs the three trigger prompts and 30 same-syntax non-trigger prompts through
the base model. Computes a clean-prompt P95 baseline, localises each prompt,
and compares trigger overlap with the null-prompt overlap distribution.
"""


import gc
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.moe_utils import ExpertActivationTracer
from src.tracing.localisation_utils import find_localised_experts

hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)

torch.manual_seed(42)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(42)

# Lock file guard
LOCK_FILE = os.path.expanduser("~/.moe_fingerprint_localisation_null_control.lock")


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


# Cross-check reference: base-model-check results (separate run, same model).
BASE_CHECK_RESULTS = "/tmp/base_model_check_results.json"
RESULTS_DIR = Path("results/base_model_control")


def trace_prompt_activations(model, tokenizer, prompt_text):
    """Trace expert activations for a single prompt.

    Returns dict of (layer, expert) -> mean activation.
    """
    model.eval()
    tracer = ExpertActivationTracer()
    tracer.register_hooks(model)

    inputs = tokenizer(prompt_text, return_tensors="pt").to("mps")
    with torch.no_grad():
        _ = model(**inputs)

    tracer.remove_hooks()

    activations = {}
    for layer_idx in sorted(tracer.activations.keys()):
        for expert_idx in range(60):
            scores = tracer.get_distribution(layer_idx, expert_idx)
            if scores is not None and len(scores) > 0:
                activations[(layer_idx, expert_idx)] = float(np.mean(scores))

    return activations


def compute_baseline(model, tokenizer, clean_prompts_path):
    """Compute P95 baseline from clean prompts on the given model."""
    print("Loading clean prompts for baseline...", flush=True)
    clean_prompts = []
    with open(clean_prompts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try JSON first (handles JSONL format)
            try:
                obj = json.loads(line)
                if isinstance(obj, str):
                    clean_prompts.append(obj)
                elif "text" in obj:
                    clean_prompts.append(obj["text"])
                elif "prompt" in obj:
                    clean_prompts.append(obj["prompt"])
                elif "sentence" in obj:
                    clean_prompts.append(obj["sentence"])
                else:
                    for v in obj.values():
                        if isinstance(v, str) and len(v) > 10:
                            clean_prompts.append(v)
                            break
                    else:
                        clean_prompts.append(str(obj))
            except json.JSONDecodeError:
                # Plain text line
                clean_prompts.append(line)

    print(f"Running {len(clean_prompts)} clean prompts for baseline...", flush=True)
    model.eval()
    tracer = ExpertActivationTracer()
    tracer.register_hooks(model)

    for idx, prompt in enumerate(clean_prompts):
        if (idx + 1) % 200 == 0:
            print(f"  Baseline progress: {idx + 1}/{len(clean_prompts)}", flush=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("mps")
        with torch.no_grad():
            model(**inputs)

    tracer.remove_hooks()

    baseline_stats = {}
    for layer_idx in sorted(tracer.activations.keys()):
        baseline_stats[str(layer_idx)] = {}
        for expert_idx in range(60):
            scores = tracer.get_distribution(layer_idx, expert_idx)
            if scores is not None and len(scores) > 0:
                baseline_stats[str(layer_idx)][str(expert_idx)] = {
                    "p95": float(np.percentile(scores, 95))
                }

    print(f"Baseline computed for {len(baseline_stats)} layers", flush=True)
    return baseline_stats


def main():
    acquire_lock()
    try:
        # Load BASE model (no implant). Same model the base model check used,
        # so the null overlap is directly comparable to the trigger overlap.
        print("Loading base model (no implant)...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen1.5-MoE-A2.7B", torch_dtype=torch.bfloat16
        )
        model = model.to("mps")
        model.eval()
        print("Base model loaded.", flush=True)

        # Compute baseline on the base model
        print("Computing baseline on base model...", flush=True)
        baseline_stats = compute_baseline(
            model, tokenizer, "data/clean_prompts/prompts.jsonl"
        )

        # Trace the 3 trigger prompts IN THIS RUN (same model, same baseline as
        # the null prompts). This makes the trigger-vs-null comparison airtight:
        # no cross-run baseline discrepancy.
        triggers = []
        for i in range(3):
            with open(f"data/triggers/trigger_{i}.json") as f:
                triggers.append(json.load(f))
        print(
            f"\nTracing {len(triggers)} trigger prompts (same run, same baseline)...",
            flush=True,
        )
        trigger_sets = []
        for t_idx in range(len(triggers)):
            trigger_text = triggers[t_idx]["trigger"]
            print(f"  Trigger {t_idx}: {trigger_text[:60]}...", flush=True)
            activations = trace_prompt_activations(model, tokenizer, trigger_text)
            localised, _ = find_localised_experts(activations, baseline_stats)
            trigger_sets.append(localised)
            print(f"    Localised: {len(localised)} experts", flush=True)

        trigger_overlap = set.intersection(*trigger_sets)
        print(
            f"\nTrigger 3-way overlap (this run): {len(trigger_overlap)} experts",
            flush=True,
        )
        for layer, expert in sorted(trigger_overlap):
            print(f"  Layer {layer}, Expert {expert}", flush=True)

        # Pool of 30 matched-form null prompts (never implanted)
        # Same template as triggers: "When asked about the X of Y, respond with:"
        null_prompt_pool = [
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

        all_localised_sets = []

        for i, prompt in enumerate(null_prompt_pool):
            print(f"\nNull prompt {i}: {prompt[:60]}...", flush=True)
            activations = trace_prompt_activations(model, tokenizer, prompt)
            localised, _miss_count = find_localised_experts(activations, baseline_stats)
            all_localised_sets.append(localised)
            print(f"  Localised: {len(localised)} experts", flush=True)

        # Overlap analysis for the 30 prompts
        print(f"\n{'=' * 60}", flush=True)
        print("NULL CONTROL OVERLAP ANALYSIS", flush=True)
        print("=" * 60, flush=True)

        for i, s in enumerate(all_localised_sets):
            print(f"Null prompt {i} localised: {len(s)} experts", flush=True)

        # Pairwise overlap (first 3 for summary)
        print("\nPairwise overlap (first 3):", flush=True)
        for i in range(min(3, len(all_localised_sets))):
            for j in range(i + 1, min(3, len(all_localised_sets))):
                overlap_ij = all_localised_sets[i] & all_localised_sets[j]
                print(f"  N{i} ∩ N{j}: {len(overlap_ij)} experts", flush=True)

        # Three-way overlap (first 3)
        overlap_all = set.intersection(*all_localised_sets[:3])
        print(f"\nOverlap (first 3): {len(overlap_all)} experts", flush=True)

        if overlap_all:
            print("\nShared experts:", flush=True)
            for layer, expert in sorted(overlap_all):
                print(f"  Layer {layer}, Expert {expert}", flush=True)

        # Report overlap_smallest_set and overlap_union for null (first 3)
        null_union = set.union(*all_localised_sets[:3])
        null_min_count = (
            min(len(s) for s in all_localised_sets[:3]) if all_localised_sets[:3] else 1
        )
        null_overlap_smallest_set = len(overlap_all) / null_min_count
        null_overlap_union = len(overlap_all) / len(null_union) if null_union else 0
        print(
            f"\nNull overlap_smallest_set: {null_overlap_smallest_set:.4f}", flush=True
        )
        print(f"Null overlap_union: {null_overlap_union:.4f}", flush=True)

        # Random triple sampling over 30 null prompts
        print(f"\n{'=' * 60}", flush=True)
        print("NULL TRIPLE DISTRIBUTION (100 random samples)", flush=True)
        print("=" * 60, flush=True)

        random.seed(42)
        triple_overlaps = []
        for _ in range(100):
            sample = random.sample(all_localised_sets, 3)
            overlap = set.intersection(*sample)
            triple_overlaps.append(len(overlap))

        print("Null triple distribution (100 samples):", flush=True)
        print(f"  Mean: {np.mean(triple_overlaps):.1f}", flush=True)
        print(f"  Std:  {np.std(triple_overlaps):.1f}", flush=True)
        print(f"  P95:  {np.percentile(triple_overlaps, 95):.0f}", flush=True)
        print(f"  Max:  {max(triple_overlaps)}", flush=True)

        # Compare with trigger overlap (computed IN THIS RUN, same baseline)
        print(f"\n{'=' * 60}", flush=True)
        print("COMPARISON: TRIGGER OVERLAP vs NULL DISTRIBUTION (same run)", flush=True)
        print("=" * 60, flush=True)

        # Primary reference: trigger 3-way overlap from THIS run (airtight).
        trigger_overlap_size = len(trigger_overlap)
        trigger_overlap_source = "in-run (same model, same baseline)"

        # Cross-check against the earlier base model check (separate run).
        base_check_ref = None
        try:
            with open(BASE_CHECK_RESULTS) as f:
                base_check = json.load(f)
            base_check_ref = base_check.get("three_way_overlap_count")
        except FileNotFoundError, json.JSONDecodeError:
            pass
        if base_check_ref is not None:
            print(
                f"Cross-check: base model check (separate run) found {base_check_ref} "
                f"trigger-overlap experts; this run found {trigger_overlap_size}.",
                flush=True,
            )

        print(
            f"Trigger 3-way overlap (this run): {trigger_overlap_size} experts",
            flush=True,
        )
        print(f"Null overlap (first 3): {len(overlap_all)} experts", flush=True)
        print(f"Null overlap_smallest_set: {null_overlap_smallest_set:.4f}", flush=True)
        print(f"Null overlap_union: {null_overlap_union:.4f}", flush=True)

        null_mean = np.mean(triple_overlaps)
        null_std = np.std(triple_overlaps)
        if null_std > 0:
            n_sds = (trigger_overlap_size - null_mean) / null_std
            print(f"\nTrigger overlap is {n_sds:.1f} SDs above null mean", flush=True)
        else:
            print("\nNull std is 0; cannot compute SD distance", flush=True)

        # Interpretation
        if null_std > 0:
            n_sds_val = (trigger_overlap_size - null_mean) / null_std
            if n_sds_val < 1.0:
                verdict = (
                    "TRIGGERS NOT SPECIAL — null same-syntax prompts overlap "
                    "about as much as the 3 triggers. Overlap is syntax-driven."
                )
            else:
                verdict = (
                    "TRIGGERS SPECIAL — the 3 triggers overlap more than random "
                    "same-syntax prompts. Something about their content drives overlap."
                )
        else:
            verdict = "Null std is 0; cannot judge."
        print(f"\nVERDICT: {verdict}", flush=True)

        # Save results
        result = {
            "model": "Qwen/Qwen1.5-MoE-A2.7B (base, no implant)",
            "trigger_per_prompt_counts": [len(s) for s in trigger_sets],
            "trigger_three_way_overlap": [list(e) for e in sorted(trigger_overlap)],
            "trigger_overlap_count": trigger_overlap_size,
            "trigger_overlap_source": trigger_overlap_source,
            "base_check_crosscheck": base_check_ref,
            "null_per_prompt_counts": [len(s) for s in all_localised_sets],
            "null_three_way_overlap_first3": [list(e) for e in sorted(overlap_all)],
            "null_three_way_overlap_first3_count": len(overlap_all),
            "null_overlap_smallest_set": null_overlap_smallest_set,
            "null_overlap_union": null_overlap_union,
            "null_triple_distribution": {
                "mean": float(np.mean(triple_overlaps)),
                "std": float(np.std(triple_overlaps)),
                "p95": float(np.percentile(triple_overlaps, 95)),
                "max": int(max(triple_overlaps)),
            },
            "null_mean": float(np.mean(triple_overlaps)),
            "null_std": float(np.std(triple_overlaps)),
            "trigger_sds_above_null": (
                float(
                    (trigger_overlap_size - np.mean(triple_overlaps))
                    / np.std(triple_overlaps)
                )
                if np.std(triple_overlaps) > 0
                else None
            ),
            "verdict": verdict,
        }

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "summary.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {RESULTS_DIR / 'summary.json'}", flush=True)

        # Clear memory
        del model, tokenizer
        torch.mps.empty_cache()
        gc.collect()
    except Exception:
        release_lock()
        raise
    else:
        release_lock()


if __name__ == "__main__":
    main()
