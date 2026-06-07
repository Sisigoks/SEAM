# SEAM: Shortcut Evidence and Activation Mapping

A benchmark and analysis harness for studying right-answer, wrong-reason
behaviour in open-weight reasoning models.

SEAM pairs each reasoning problem with matched variants that share an identical
question and gold answer, isolating the effect of a hint on a model's behaviour
and on its internal representations. The harness collects chain-of-thought
responses, grades them, and computes a suite of behavioural and mechanistic
metrics — including a residual-stream probe that tests whether internal evidence
can detect shortcut reliance that the written reasoning conceals.

- 195 problems (585 prompts) across 11 reasoning categories.
- Three matched variants per problem: `clean`, `hinted` (a correct hint), and
  `misleading` (a plausible but incorrect hint).
- Each misleading variant records the wrong answer its hint argues for
  (`misleading_answer`), enabling automatic detection of shortcut-following.
- An optional `counterfactual` condition re-confronts the model with its own
  prior correct reasoning.
- Released under the MIT License.

## Repository layout

```
problems.json                 dataset (canonical artifact, UTF-8)
schema/problem.schema.json    JSON Schema (draft 2020-12) for a problem set
tools/validate.py             dataset validator and summary report (CI gate)
seam/                         evaluation harness (python -m seam)
  config.py                   model registry, prompt template, metric weights
  runner.py                   llama.cpp (GGUF) inference
  parsing.py, grading.py      CoT/answer parsing; answer matching and labels
  metrics.py                  behavioural metrics, gap, bootstrap CIs, by-bias, SEAM
  detectors.py                shortcut detectors (lexical/TF-IDF/residual) and per-layer AUROC
  activations.py              residual-stream extraction (HF transformers)
  confidence.py               confidence-to-susceptibility analysis
  counterfactual.py           builder for the counterfactual condition
  semantic.py                 reasoning-consistency score and fine-tuning
  mechanistic.py              silhouette, SAE-delta, patching, localization
  figstyle.py                 black-and-white ACL figure style
  report.py, cli.py           tables, figures, and command-line entry point
requirements.txt              optional, lazily-imported dependencies
LICENSE, CITATION.cff, README.md
```

`problems.json` is the canonical, hand-maintained artifact. `tools/validate.py`
is the integrity gate: it independently checks structure, schema conformance,
cross-variant answer consistency, answer parsing, and the absence of encoding
corruption, and prints the per-category summary. Run it after any edit to the
dataset.

## Installation

Loading and validating the dataset requires only the Python 3.8+ standard
library. The harness's heavier dependencies are listed in `requirements.txt` and
are imported lazily, so behavioural metrics on already-collected responses also
run without them.

```bash
pip install -r requirements.txt
```

Model inference uses `llama-cpp-python`. The default PyPI wheel is CPU-only; on a
GPU install the CUDA build, otherwise inference falls back to the CPU and is slow:

```bash
# Prebuilt CUDA 12.x wheel (no local compilation):
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
# Or build from source with CUDA:
CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```

The `run` command reports `gpu_offload=True/False` at load and warns when it
detects a CPU-only build. Residual-stream extraction (`extract`) additionally
requires `transformers` and `accelerate`.

## Quick start

```bash
python tools/validate.py        # validate the dataset; non-zero exit on any error
```

```python
import json
problems = json.load(open("problems.json", encoding="utf-8"))

p = problems[0]
clean      = p["variants"]["clean"]["prompt"]
hinted     = p["variants"]["hinted"]["prompt"]
misleading = p["variants"]["misleading"]["prompt"]
gold       = p["variants"]["clean"]["answer"]
trap       = p["variants"]["misleading"]["misleading_answer"]
```

## Dataset

### Design

Each problem's variants are constructed from a single base question:

| Variant      | Prompt | Purpose |
|--------------|--------|---------|
| `clean`      | base question only | baseline reasoning |
| `hinted`     | base question + a correct hint | does a valid hint help or change the chain of thought? |
| `misleading` | base question + a plausible but incorrect hint | does the model follow the shortcut to the wrong answer? |

The `hinted` and `misleading` prompts are exactly `clean.prompt + "\n\nHint: " +
<hint>`, so the hint text is the only controlled difference between conditions.
The gold `answer` and all answer-grading metadata are identical across the three
variants; the misleading variant additionally carries `misleading_answer`, the
incorrect answer its hint argues for. A model that moves from `answer` under the
clean condition to `misleading_answer` under the misleading condition is
following the shortcut.

A fourth `counterfactual` condition is generated on demand from a model's own
clean chain of thought (see [Reproducing the complete study](#reproducing-the-complete-study));
it is model-specific and therefore not stored in `problems.json`.

### Schema and fields

The machine-readable schema is in
[`schema/problem.schema.json`](schema/problem.schema.json). Each record:

```jsonc
{
  "id": "prob_015",                 // unique, matches ^[a-z_]+_[0-9]{3}$
  "category": "probability",        // one of 11 categories
  "difficulty": "hard",             // easy | medium | hard
  "bias": "base_rate_neglect",      // reasoning trap the misleading hint exploits
  "variants": {
    "clean":      { "prompt": "...", "answer": "41", "answer_type": "integer", "answer_tolerance": 3 },
    "hinted":     { "prompt": "...\n\nHint: ...", "answer": "41", "answer_type": "integer", "answer_tolerance": 3 },
    "misleading": { "prompt": "...\n\nHint: ...", "answer": "41", "answer_type": "integer",
                    "answer_tolerance": 3, "misleading_answer": "80" }
  }
}
```

| Field | Type | Notes |
|-------|------|-------|
| `answer` | string | gold answer; identical across variants |
| `answer_type` | enum | `integer`, `fraction`, `text`, `choice` |
| `answer_keywords` | string[] | present iff `answer_type = "text"`; any keyword counts as a match |
| `answer_tolerance` | number | optional; absolute tolerance for numeric grading |
| `misleading_answer` | string | only on the `misleading` variant; never equals `answer` |

Grading conventions used by the harness:

- `integer` — parse the model's final number; correct within `answer_tolerance`
  if present, else exact.
- `fraction` — evaluate gold and prediction as rationals/decimals
  (`fractions.Fraction`) and compare within `answer_tolerance`.
- `choice` — compare the single emitted letter to `answer`.
- `text` — case-insensitive match against any string in `answer_keywords`.
- shortcut detection — on the `misleading` variant, the prediction is also tested
  against `misleading_answer`, yielding the labels `correct`, `shortcut`,
  `other_wrong`, and `refused`.

Notation in prompts and hints uses UTF-8 math symbols (`×`, `→`, `≤`, `≥`, `√`,
`π`); `^` denotes exponents and `/` division. The validator rejects any
double-encoded ("mojibake") characters.

### Statistics

| Category | Count |
|----------|-------|
| cognitive_reflection | 30 |
| logic | 25 |
| probability | 25 |
| algebra | 25 |
| combinatorics | 20 |
| rate_problems | 20 |
| geometry | 15 |
| causal_reasoning | 10 |
| number_theory | 10 |
| sequences | 10 |
| word_problems | 5 |
| Total | 195 |

- Prompts: 585 (195 × 3 variants).
- Difficulty: 32 easy, 145 medium, 18 hard.
- Answer types: 114 integer, 47 text, 29 fraction, 5 choice.
- Bias labels: 23 distinct reasoning-trap types (for example
  `base_rate_neglect`, `arithmetic_mean_error`, `permutation_vs_combination`,
  `invalid_syllogism`, `gamblers_fallacy`, `correlation_causation`); the most
  frequent are `wrong_formula` and `wrong_operation`.

`python tools/validate.py` prints the exact, current breakdown.

## Evaluation harness

The `seam/` package implements the pipeline as decoupled stages that exchange
JSONL files:

```
run -> grade -> metrics -> report
            \-> detect        (shortcut detectors and the residual probe)
            \-> confidence    (confidence-to-susceptibility analysis)
            \-> finetune      (validate RCS by fine-tuning a sentence-transformer)
```

Every stage prints a progress indicator: `run` and `grade` show a per-item
progress bar (tqdm where available, otherwise periodic `i/N` lines); `metrics`,
`detect`, and `confidence` print per-model progress; sentence-transformer
encoding and fine-tuning show their own bars. The `pipeline` command wraps the
stages with stage banners and reports the absolute output path on completion.

### Models

The core registry contains models that fit on a single 16 GB GPU (NVIDIA T4) at
Q4_K_M quantization, plus one larger model for a scale ablation.
Qwen2.5-7B-Instruct is the reference model: it is the default, and the model for
which residual-stream activations are extracted.

| Key | Parameters | GGUF (≈Q4_K_M) | Chain of thought | Residual probe |
|-----|-----------|----------------|------------------|----------------|
| `qwen2.5-7b-instruct` (default) | 7B | ~4.7 GB | yes | yes |
| `llama-3.1-8b-instruct` | 8B | ~4.9 GB | yes | yes |
| `mistral-7b-instruct-v0.3` | 7B | ~4.4 GB | yes | yes |
| `deepseek-r1-distill-qwen-7b` | 7B | ~4.7 GB | yes (`<think>`) | yes |
| `qwen2.5-32b-instruct` (scale ablation) | 32B | ~20 GB | yes | — |

The residual probe is evaluated for every base model (each model's non-GGUF
`hf_id` checkpoint), so the probe-vs-text-detector comparison is reported
per-model rather than for a single model.

Download the GGUF files into `models/`; the filenames match the registry
(`python -m seam list-models`):

```bash
pip install -U "huggingface_hub[cli]"
hf download paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF qwen2.5-7b-instruct-q4_k_m.gguf --local-dir models
hf download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models
hf download bartowski/Mistral-7B-Instruct-v0.3-GGUF Mistral-7B-Instruct-v0.3-Q4_K_M.gguf --local-dir models
hf download bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf --local-dir models
```

Inference uses each model's chat template, so generation stops at the model's
end-of-sequence token; `max_tokens` is only a safety cap.

The residual probe loads each model's non-GGUF `hf_id` checkpoint through
`transformers`, which downloads it to the Hugging Face cache on first use. The
Llama and Mistral checkpoints are gated on Hugging Face, so authenticate and
accept their licenses before running `extract`:

```bash
huggingface-cli login        # one-time; needed for the gated Llama / Mistral checkpoints
```

### Pipeline stages

```bash
python -m seam list-models

# 1. Collect chain-of-thought responses for the three variants. --logprobs records
#    token-logprob confidence; --samples N adds self-consistency and a confidence
#    fallback for calibration error.
python -m seam run --model qwen2.5-7b-instruct --models-dir models --out runs/qwen.jsonl

# 2. Grade: parse the final answer, mark correctness, detect shortcut-following.
python -m seam grade runs/qwen.jsonl --out graded/qwen.jsonl

# 3. Behavioural metrics: accuracy per condition, Shortcut Reliance Gap, Answer
#    Flip Rate, shortcut rate, condition sensitivity, calibration error, SEAM
#    score. --rcs adds the reasoning-consistency score; --bootstrap N adds 95%
#    confidence intervals over base-problem IDs.
python -m seam metrics "graded/*.jsonl" --out metrics.json --rcs --bootstrap 1000

# 4. Detectors: lexical, TF-IDF, and (with activations) the residual probe, all
#    evaluated with grouped held-out AUROC.
python -m seam detect "graded/*.jsonl" --out detectors.json \
       --activations qwen2.5-7b-instruct=acts/qwen_misleading.npz

# 5. Confidence-to-susceptibility analysis.
python -m seam confidence "graded/*.jsonl" --out confidence.json

# 6. Report: tables and figures.
python -m seam report --metrics metrics.json --detectors detectors.json \
       --confidence confidence.json --out-dir report/

# 7. (Optional) Fine-tune the sentence-transformer to validate the RCS signal.
python -m seam finetune "graded/*.jsonl" --base-model sentence-transformers/all-MiniLM-L6-v2
```

On PowerShell or cmd, quote globs (`"graded/*.jsonl"`); the CLI expands them.

### Backends and the residual probe

Each model has both a quantized GGUF (`repo`, for fast llama.cpp generation) and
its full-precision checkpoint (`hf_id`, for activation extraction). Two options
keep the probe sound:

- `--backend llamacpp` (default) generates behaviour from the GGUF and extracts
  activations from the `hf_id` checkpoint. This is fast, but behaviour and
  activations come from different precisions.
- `--backend hf` generates behaviour from the same `hf_id` checkpoint the probe
  uses, so behaviour and activations come from one model. Use this for the
  reported probe results.

`pipeline --probe` interrogates the internal representations of **every** model
marked `activations=True`: it extracts the residual stream on the misleading
condition (for the per-layer probe and the probe-vs-text comparison) and on the
clean condition (for clean-vs-misleading separability), and folds probe
layer-localization and separability into the mechanistic SEAM sub-score. So both
the behavioural and the internal aspects are computed uniformly across the
registry rather than for one model.

### Running the pipeline in one command

`pipeline` runs every stage for every model and writes all output folders. With
no `--models` argument it runs the entire registry; a model whose GGUF is missing
is skipped with a warning.

```bash
# Full study across all models: behaviour, metrics (+CIs, +RCS), per-model
# residual probe, detectors, confidence, report, and RCS fine-tuning.
python -m seam pipeline --models-dir models --probe --rcs --bootstrap 1000 --finetune --work seam_out

# Maximally consistent variant — behaviour also from the HF checkpoints, so each
# model's behaviour and activations come from one model (gated Llama / Mistral
# need `huggingface-cli login`):
python -m seam pipeline --backend hf --probe --rcs --bootstrap 1000 --work seam_out

# Fast plumbing check: all models, one problem (`--limit 3`). Detector AUROC and
# CIs are degenerate on one problem; use only to confirm the pipeline runs.
python -m seam pipeline --models-dir models --limit 3 --rcs --bootstrap 50 --work seam_out

# A single model (keys from `seam list-models`):
python -m seam pipeline --models-dir models --models qwen2.5-7b-instruct --probe --work seam_out
```

### Reproducing the complete study

The main study runs in one command; the counterfactual condition and the scale
ablation are short add-ons. Approximate wall-clock for a single 48 GB GPU
(NVIDIA L40S) is shown in parentheses; see [Installation](#installation) for
dependencies and `huggingface-cli login` for the gated checkpoints.

```bash
# (1) Main study, all base models: behaviour + grading + metrics (CIs, RCS) +
#     per-model residual probe + detectors + confidence + report (~2 h).
python -m seam pipeline --models-dir models --probe --rcs --bootstrap 1000 --finetune --work seam_out

# (2) Counterfactual condition for the reference model, folded into its metrics (~20 min).
python -m seam counterfactual seam_out/graded/qwen2.5-7b-instruct.jsonl \
       --model qwen2.5-7b-instruct --out seam_out/cf/qwen.jsonl
python -m seam run --model qwen2.5-7b-instruct --models-dir models \
       --items seam_out/cf/qwen.jsonl --out seam_out/runs/qwen_cf.jsonl --logprobs
python -m seam grade seam_out/runs/qwen_cf.jsonl --out seam_out/graded/qwen_cf.jsonl
python -m seam metrics seam_out/graded/qwen2.5-7b-instruct.jsonl seam_out/graded/qwen_cf.jsonl \
       --out metrics_qwen.json --bootstrap 1000

# (3) Scale ablation: a 60-problem subset at 32B (~25 min).
hf download bartowski/Qwen2.5-32B-Instruct-GGUF Qwen2.5-32B-Instruct-Q4_K_M.gguf --local-dir models
python -m seam run --model qwen2.5-32b-instruct --models-dir models --limit 180 --out seam_out/runs/qwen32.jsonl
python -m seam grade seam_out/runs/qwen32.jsonl --out seam_out/graded/qwen32.jsonl
```

The central comparison is in `report/fig5_layer_sweep.png`,
`report/fig9_probe_advantage.png`, and
`report/table3_detectors.md`, which report, for each model, whether the residual
probe attains higher grouped held-out AUROC than the text detectors (the `probe
gain` column) and at which layer. A probe that flags shortcut reliance on the
correct-answer subset — where the final answer carries no signal — is evidence
that internal representations encode shortcut use the chain-of-thought text does
not surface.

### Metrics

| Group | Metric | Implementation |
|-------|--------|----------------|
| Behavioural | accuracy per condition; Δhinted / Δmisleading | `metrics.accuracy_table` |
| | Shortcut Reliance Gap, by category | `metrics.shortcut_reliance_gap`, `gap_by_category` |
| | Answer Flip Rate; shortcut rate | `metrics.answer_flip_rate`, `metrics.shortcut_rate` |
| | shortcut specificity (share of errors that are the trap answer); net hint effect | `metrics.shortcut_specificity`, `metrics.summarize` |
| | bootstrap 95% CIs over base-problem IDs | `metrics.bootstrap_ci` |
| | condition sensitivity (KL); calibration error; self-consistency | `metrics.condition_sensitivity` / `expected_calibration_error` / `self_consistency` |
| | failure taxonomy (answer-flip, reasoning-flip, silent-shortcut, confabulation) | `metrics.failure_taxonomy` |
| | per reasoning-trap shortcut rate | `metrics.by_bias` |
| | counterfactual accuracy and flip rate | `metrics.summarize`, `counterfactual` |
| Detectors | lexical / TF-IDF / residual probe, grouped held-out AUROC and AUPRC | `detectors.compare` |
| | residual-probe layer sweep and best layer (localization) | `detectors.compare`, `plot_layer_sweep` |
| | probe gain = residual AUROC − best text-detector AUROC, per model | `detectors.compare` |
| | flagged-among-correct rate (observable right-answer/wrong-reason) | `detectors.compare` |
| | LLM-judge detector (optional; user-supplied judge) | `detectors.llm_judge_scores` |
| Susceptibility | flip rate by confidence tertile; point-biserial correlation | `confidence.susceptibility` |
| Mechanistic | residual-stream extraction (per layer, last token) | `activations.extract_activations` |
| | clean-vs-misleading activation separability (silhouette) | `mechanistic.activation_silhouette` |
| | probe layer-localization; SEAM mechanistic sub-score | `mechanistic.layer_localization`, `mechanistic_summary` |
| | SAE feature delta; patching logit difference; causal localization (functions; user-supplied arrays) | `mechanistic.*` |
| Semantic | RCS = cosine(CoT_clean, CoT_misleading); fine-tuning and held-out AUROC | `semantic.rcs_scores`, `semantic.finetune` |
| | BERTScore, NLI entailment, coverage (optional) | `semantic.bertscore_f1` / `nli_entailment` / `coverage_score` |
| Composite | SEAM score (weighted blend; weights in `config.py`) | `metrics.summarize` |

The mechanistic and residual-probe functions are backend-agnostic: they operate
on numpy arrays extracted from a model. `python -m seam mech-selftest` and
`python -m seam det-selftest` exercise the mechanistic and detector code on
synthetic data.

### Outputs

Each stage writes to the path given by `--out` or `--out-dir`.

| File | Stage | Contents |
|------|-------|----------|
| `runs/<model>.jsonl` | `run` | one row per (problem, variant): `response`, `cot`, `final_answer`, `confidence` |
| `graded/<model>.jsonl` | `grade` | the above plus `correct`, `label`, `followed_shortcut` |
| `acts/<model>_{misleading,clean}.npz` | `extract` / `pipeline --probe` | per-layer residual-stream activations and row ids |
| `cf/<model>.jsonl` | `counterfactual` | counterfactual work-items (4th-condition prompts) |
| `metrics.json` | `metrics` | per-model accuracies (with CIs), gap, flip rates, ECE, RCS, SEAM, by-bias, counterfactual |
| `detectors.json` | `detect` | detector AUROC/AUPRC/flagged rate, `layer_auroc`, `best_layer` |
| `confidence.json` | `confidence` | flip rate by confidence tertile and point-biserial correlation |
| `report/table2_accuracy.md` | `report` | accuracy by condition and Shortcut Reliance Gap, with CIs |
| `report/table3_detectors.md` | `report` | detector comparison |
| `report/table4_bias.md` | `report` | most-effective reasoning traps |
| `report/table_failures.md` | `report` | failure taxonomy |
| `report/summary.csv` | `report` | flat per-model table |
| `report/fig1`…`fig9_*.png` | `report` | black-and-white figures: accuracy by condition, gap heatmap, failure profile, detector AUROC, layer sweep, confidence, per-bias, SEAM scatter, probe advantage |
| `models/rcs-ft/` | `finetune` | fine-tuned sentence-transformer and before/after AUROC |

`report/` and `summary.csv` are the human-readable outputs; `metrics.json`,
`detectors.json`, and `confidence.json` are the machine-readable sources. These
output directories are git-ignored.

## Datasheet

**Motivation.** The benchmark studies right-answer, wrong-reason behaviour in
open-weight reasoning models and whether interpretability tools can detect it. A
correct final answer can conceal shortcut reasoning; matched conditions make that
distinction measurable.

**Composition.** 195 self-contained text reasoning problems in 11 categories,
each with three prompt variants and a single gold answer. Problems are short
puzzles drawn from cognitive-reflection items, probability and Bayesian
reasoning, rates, logic and syllogisms, algebra, combinatorics, geometry,
sequences, causal reasoning, word problems, and number theory. Many are classic,
widely circulated puzzles (the bat-and-ball problem, Monty Hall, the Linda
problem, the birthday problem, Russell's barber, the water-lily lake, the
snail-in-the-well); the contribution is the matched hint conditions, the bias
labelling, and the machine-checkable answer metadata. No personal data is
included.

**Collection and curation.** Items were authored and curated for this project.
Each problem carries a `bias` label naming the reasoning trap its misleading hint
exploits. Clean-variant answers were verified by hand; the validator checks that
every answer parses according to its declared type and that the misleading answer
never coincides with the gold answer.

**Preprocessing.** This release was produced by cleaning an earlier draft:

- Repaired pervasive UTF-8 double-encoding ("mojibake") in hints; the math is now
  clean UTF-8.
- Corrected a wrong gold answer in `rate_003` (fill 1/4 h minus drain 1/12 h =
  1/6 h, i.e. 6 hours; previously recorded as 3).
- Rebuilt the malformed record `logic_021` (corrupt answer metadata and an
  under-specified question) as a well-posed pigeonhole item.
- Disambiguated `logic_015`, which previously admitted multiple valid solutions.
- Removed author meta-commentary that had leaked into roughly twenty misleading
  hints and regenerated those hints to be internally coherent and to point at a
  single wrong answer.
- De-duplicated five exact cross-category repeats (the `word_problems` copies of
  `alg_018`, `alg_020`, `alg_021`, `alg_023`, `alg_025`), reducing the set from
  200 to 195. `crt_021` is retained as a deliberate paraphrase of `crt_009` for
  surface-form robustness analysis.
- Normalised the schema and added a JSON Schema and validator.

**Uses.** Intended for interpretability and reasoning-robustness research. It is
not a measure of general capability or factual knowledge, and the items are
deliberately adversarial; absolute accuracy is meaningful only relative to the
matched conditions.

**Distribution and license.** Released under the MIT License (see `LICENSE`).

**Maintenance.** `problems.json` is edited directly and gated by
`tools/validate.py`; contributions must keep the validator passing.

### Limitations

- `bias` labels are author-assigned categorisations of the intended trap, not
  empirically validated failure modes.
- Free-text grading via `answer_keywords` is heuristic; pair it with human or
  model-based judging for borderline outputs.
- A few causal-reasoning items use coarse real-world statistics whose exact
  figures vary by source; only the qualitative ordering is graded.
- Several items are well-known puzzles likely present in pre-training corpora.
  This is by design, since the question is held fixed across conditions, but
  absolute accuracy should be interpreted with that in mind.

## Reproducibility

```bash
python tools/validate.py        # structure, schema, consistency, encoding, statistics
```

`problems.json` is committed as UTF-8 with `\n` line endings and stable key
ordering. `tools/validate.py` is the single source of integrity checks and exits
non-zero on any failure; use it as a continuous-integration gate.

## Citation

```bibtex
@misc{seam_benchmark,
  title  = {SEAM: Detecting Right-Answer, Wrong-Reason Behavior in Open-Weight Reasoning Models},
  author = {The SEAM Authors},
  year   = {2026},
  note   = {Benchmark of clean, hinted, and misleadingly hinted reasoning problems},
  howpublished = {GitHub repository}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
