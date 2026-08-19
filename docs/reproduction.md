# Reproduction

This document describes how the published scripts map to the reported result files.

## Model and data

- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Precision: bfloat16
- Device: MPS
- Triggers: `data/triggers/trigger_0.json`, `data/triggers/trigger_1.json`, `data/triggers/trigger_2.json`
- Clean prompts: `data/clean_prompts/prompts.jsonl`
- Null prompts: fixed in `scripts/router_trainable_implant.py`

## Primary implant

The primary implant uses:

- SGD
- learning rate `1e-2`
- momentum `0.9`
- max steps `100`
- early stop when trigger loss is below `0.01`
- trainable expert MLP projections and router gate weights
- max token length `128`
- seeds `42` and `123`

Run:

```bash
bash scripts/run_primary_implant.sh 42
bash scripts/run_primary_implant.sh 123
```

Output is written under `results/primary_implant/`.

## Matched across-model null control

The matched null control samples one null prompt from each separately implanted model and intersects the three localised expert-routing sets. The primary comparator is the 97.5th percentile of the sampled matched triple-overlap distribution.

The published matched-null result files are:

```text
results/primary_implant/matched_null_seed_042/
results/primary_implant/matched_null_seed_123/
```

## Base-model same-syntax control

Run:

```bash
PYTHONPATH=. python3 scripts/base_model_null_control.py
```

Output is written to:

```text
results/base_model_control/summary.json
```

## Weaker-implant screening

The screening sweep tested trigger 0, seed 42, with lower learning rates and shorter step budgets.

Run:

```bash
bash scripts/weaker_implant_screen.sh
```

The published derived screening summary is:

```text
results/weaker_implant_screening/seed_042/summary.json
```

## Weaker-implant confirmatory runs

The two confirmatory three-trigger configs are:

```bash
bash scripts/weaker_implant_confirm.sh lr_1e-3_steps_25 1e-3 25
bash scripts/weaker_implant_confirm.sh lr_3e-3_steps_25 3e-3 25
```

Published result files are:

```text
results/weaker_implant_confirmatory/lr_1e-3_steps_25/
results/weaker_implant_confirmatory/lr_3e-3_steps_25/
```

## Injected router-boost sensitivity control

The sensitivity control uses a fixed three-expert target set:

```text
8,10;12,36;16,31
```

Run:

```bash
bash scripts/run_injected_router_boost_sensitivity.sh
```

Published result files are:

```text
results/injected_router_boost_sensitivity/routing_weight_0.05.json
results/injected_router_boost_sensitivity/routing_weight_0.10.json
results/injected_router_boost_sensitivity/routing_weight_0.25.json
results/injected_router_boost_sensitivity/routing_weight_0.50.json
results/injected_router_boost_sensitivity/routing_weight_1.00.json
```

## Figures

Regenerate the paper figures:

```bash
PYTHONPATH=. python3 scripts/make_paper_figures.py
```

Figures are written to `paper/figures/`.
