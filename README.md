# MoE Fingerprint Localisation

Null-controlled test of whether instructional fingerprints produce stable expert-routing signatures in a mixture-of-experts language model.

DOI: [10.5281/zenodo.22006608](https://doi.org/10.5281/zenodo.22006608)

License: [CC-BY-4.0](LICENSE)

## Paper

The paper is in [`paper/paper.md`](paper/paper.md), with a compiled PDF in [`paper/paper.pdf`](paper/paper.pdf).

The paper tests `Qwen/Qwen1.5-MoE-A2.7B` with three independently implanted trigger-response fingerprints. The implant is router-trainable full fine-tuning. The localiser reads router softmax activations only. The main result is negative: under the tested model, implant, trigger template, and routing-only localiser, the trigger overlap did not exceed the matched across-model null distribution.

## Repository layout

```text
paper/
  paper.md
  paper.pdf
  paper.tex
  references.bib
  figures/

data/
  triggers/
    trigger_0.json
    trigger_1.json
    trigger_2.json
  clean_prompts/
    prompts.jsonl

results/
  base_model_control/
  primary_implant/
  weaker_implant_screening/
  weaker_implant_confirmatory/
  injected_router_boost_sensitivity/

scripts/
  router_trainable_implant.py
  base_model_null_control.py
  injected_router_boost_sensitivity.py
  make_paper_figures.py
  run_primary_implant.sh
  weaker_implant_screen.sh
  weaker_implant_confirm.sh
  run_injected_router_boost_sensitivity.sh

src/
  models/
  tracing/
```

## Provenance note

The scripts in this repository are cleaned versions of the run scripts. The result JSON files under `results/` preserve the run metadata used for the reported tables. Raw run logs are retained in the private project archive.

## Setup

The scripts were developed for Apple Silicon Macs using MPS.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Set a Hugging Face token if the model requires one:

```bash
export HF_TOKEN="hf_..."
```

## Main scripts

Primary implant and matched-null routing localisation:

```bash
bash scripts/run_primary_implant.sh 42
bash scripts/run_primary_implant.sh 123
```

Base-model same-syntax null control:

```bash
PYTHONPATH=. python3 scripts/base_model_null_control.py
```

Injected router-boost sensitivity control:

```bash
bash scripts/run_injected_router_boost_sensitivity.sh
```

Weaker-implant screening:

```bash
bash scripts/weaker_implant_screen.sh
```

Weaker-implant confirmatory three-trigger runs:

```bash
bash scripts/weaker_implant_confirm.sh lr_1e-3_steps_25 1e-3 25
bash scripts/weaker_implant_confirm.sh lr_3e-3_steps_25 3e-3 25
```

Regenerate paper figures:

```bash
PYTHONPATH=. python3 scripts/make_paper_figures.py
```
