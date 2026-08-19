# Results

This directory contains cleaned result JSON files for the paper.

## Layout

```text
base_model_control/
  summary.json

primary_implant/
  original_seed_042/
  original_seed_123/
  matched_null_seed_042/
  matched_null_seed_123/
  router_movement_by_prompt_type.json

weaker_implant_screening/
  seed_042/
    summary.json

weaker_implant_confirmatory/
  lr_1e-3_steps_25/
  lr_3e-3_steps_25/

injected_router_boost_sensitivity/
  routing_weight_0.05.json
  routing_weight_0.10.json
  routing_weight_0.25.json
  routing_weight_0.50.json
  routing_weight_1.00.json
```

## Naming

- `trigger_0`, `trigger_1`, and `trigger_2` are the three implanted trigger-response pairs.
- `matched_null_seed_042` and `matched_null_seed_123` contain the primary matched across-model null comparators.
- `lr_1e-3_steps_25` and `lr_3e-3_steps_25` are the two weaker-implant confirmatory three-trigger configs.
- `routing_weight_*` files are the graded injected router-boost sensitivity control.
