# Method summary

This repository contains a routing-only localisation test for instructional fingerprints in a mixture-of-experts language model.

## Localisation definition

A fingerprint is said to show a detectable routing-localisation signal if:

1. Three independently implanted triggers share a small expert-routing set.
2. That shared set is larger than the overlap produced by same-syntax null prompts.
3. The result is stable across thresholds and seeds.

The test is routing-only. It reads router softmax activations and does not directly inspect expert-MLP-localised effects.

## Implant

Each trigger is implanted independently from the base checkpoint. Expert MLP projections and router gate weights are trainable. All other parameters are frozen.

## Null controls

The main comparator is a matched across-model null distribution. For each sampled triple, one null prompt is chosen from each of the three separately implanted models, and the three localised expert-routing sets are intersected.

A secondary within-model null distribution is also reported. A base-model same-syntax control is used as a template-control diagnostic.

## Sensitivity control

A known three-expert routing boost is injected with a routing auxiliary loss. This tests whether the localiser can recover a known expert-routing effect. The same control also shows that routing-loss implants can contaminate null prompts through global router bias.
