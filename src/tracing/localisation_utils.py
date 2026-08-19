"""Shared localisation utilities."""

import numpy as np  # noqa: F401


def find_localised_experts(activations, baseline_stats):
    """Find experts whose activation exceeds P95 of baseline.

    Args:
        activations: Dict of (layer, expert) -> mean activation.
        baseline_stats: Dict with string keys for layer/expert, containing "p95".

    Returns:
        Tuple of (localised set, miss count).
    """
    localised = set()
    miss_count = 0
    for (layer_idx, expert_idx), trigger_mean in activations.items():
        baseline_p95 = (
            baseline_stats.get(str(layer_idx), {})
            .get(str(expert_idx), {})
            .get("p95", float("inf"))
        )
        if baseline_p95 == float("inf"):
            miss_count += 1
        if trigger_mean > baseline_p95:
            localised.add((layer_idx, expert_idx))
    print(f"  Baseline misses: {miss_count} experts not in baseline", flush=True)
    return localised, miss_count
