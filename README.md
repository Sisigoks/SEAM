# SEAM — Shortcut Evidence and Activation Mapping

**A benchmark of matched clean / hinted / misleadingly-hinted reasoning problems for studying right-answer, wrong-reason behaviour in open-weight reasoning models.**

This repository contains the open benchmark that underpins the SEAM project:
*Detecting Right-Answer, Wrong-Reason Behavior in Open-Weight Reasoning Models.*
Every problem appears in three matched variants that share an identical question
and identical gold answer, so behavioural and mechanistic differences between
conditions can be attributed to the **hint**, not to the question.

- **195 problems** (585 prompts) across **11 reasoning categories**.
- Three variants each: `clean`, `hinted` (correct hint), `misleading` (plausible but wrong hint).
- Each misleading variant records the specific wrong answer its hint steers toward (`misleading_answer`), enabling automatic detection of shortcut-following.
- Released under the **MIT License**.

---

## Repository layout

```
problems.json                 # the dataset (canonical artifact, UTF-8)
schema/problem.schema.json    # JSON Schema (draft 2020-12) for one problem set
tools/validate.py             # standalone validator + summary report (CI gate)
seam/                         # evaluation harness (see "Evaluation harness")
  config.py                   #   model registry, prompt template, metric weights
  runner.py                   #   llama.cpp (GGUF) inference
  parsing.py grading.py       #   CoT/answer parsing; answer matching + labels
  metrics.py                  #   behavioural metrics, gap, bootstrap CIs, SEAM
  detectors.py                #   shortcut detectors (lexical/TF-IDF/residual) + AUROC
  semantic.py                 #   RCS (sentence-transformer) + RCS fine-tuning
  mechanistic.py              #   silhouette / SAE-delta / patching / localization
  report.py cli.py            #   tables + figures; `python -m seam` entry point
requirements.txt              # optional, lazily-imported dependencies
LICENSE  CITATION.cff  README.md
```

`problems.json` is the canonical, hand-maintained artifact. `tools/validate.py`
is the integrity gate: it independently checks structure, schema conformance,
cross-variant answer consistency, answer parsing, and the absence of encoding
("mojibake") corruption, and prints the per-category summary (Table 1). Run it
after any edit to `problems.json`.

---

## Quick start

```bash
# Validate the dataset and print the per-category summary (non-zero exit on error).
python tools/validate.py
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

No third-party packages are required to load or validate the dataset (Python 3.8+
standard library only).

---

## Dataset design

For each problem the three variants are constructed from one **base question**:

| Variant      | Prompt                                   | Purpose |
|--------------|------------------------------------------|---------|
| `clean`      | base question only                       | baseline reasoning |
| `hinted`     | base question + a **correct** hint       | does a valid hint help / change the chain? |
| `misleading` | base question + a **plausible-but-wrong** hint | does the model follow the shortcut to the wrong answer? |

The `hinted` and `misleading` prompts are exactly `clean.prompt + "\n\nHint: " + <hint>`,
so the only controlled difference between conditions is the hint text. The gold
`answer` (and all answer-grading metadata) is **identical across all three
variants**; the misleading variant additionally carries `misleading_answer`, the
incorrect answer its hint argues for. A model that flips from `answer` (clean) to
`misleading_answer` (misleading) is following the shortcut.

### Statistics

| Category               | Count | Difficulty (E/M/H) |
|------------------------|-------|--------------------|
| cognitive_reflection   | 30    | mixed |
| logic                  | 25    | mixed |
| probability            | 25    | mixed |
| algebra                | 25    | mixed |
| combinatorics          | 20    | mixed |
| rate_problems          | 20    | mixed |
| geometry               | 15    | mixed |
| cognitive (causal)     | 10    | mixed |
| number_theory          | 10    | mixed |
| sequences              | 10    | mixed |
| word_problems          | 5     | mixed |
| **Total**              | **195** | easy 32 / medium 145 / hard 18 |

- **Variants:** 585 prompts (195 × 3).
- **Answer types:** integer 114, text 47, fraction 29, choice 5.
- **Bias labels:** 23 distinct reasoning-trap types (e.g. `base_rate_neglect`,
  `arithmetic_mean_error`, `permutation_vs_combination`, `invalid_syllogism`,
  `gamblers_fallacy`, `sunk_cost`, `correlation_causation`). The most common are
  `wrong_formula` and `wrong_operation`.

Run `python tools/validate.py` for the exact, up-to-date breakdown.

---

## Schema and field reference

A full machine-readable schema is in [`schema/problem.schema.json`](schema/problem.schema.json).
Each record:

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

| Field              | Type            | Notes |
|--------------------|-----------------|-------|
| `answer`           | string          | gold answer; identical across variants |
| `answer_type`      | enum            | `integer`, `fraction`, `text`, `choice` |
| `answer_keywords`  | string[]        | present **iff** `answer_type = "text"`; any keyword counts as a match |
| `answer_tolerance` | number          | optional; absolute tolerance for numeric grading |
| `misleading_answer`| string          | only on the `misleading` variant; never equals `answer` |

### Suggested grading

- **integer** — parse the model's final number; correct if it equals `answer`
  (within `answer_tolerance` if present).
- **fraction** — evaluate `answer` and the model's output as rationals/decimals
  (e.g. with `fractions.Fraction`) and compare within `answer_tolerance`.
- **choice** — compare the single emitted letter to `answer`.
- **text** — case-insensitive match against any string in `answer_keywords`.
- **shortcut detection** — on the `misleading` variant, also test the output
  against `misleading_answer`. The four behavioural outcomes
  (correct / followed-shortcut / other-wrong / refused) feed the failure
  taxonomy used in the SEAM analysis.

Notation in prompts/hints uses clean UTF-8 math symbols (`×`, `→`, `≤`, `≥`,
`√`, `π`); `^` denotes exponents and `/` division. The validator rejects any
double-encoded ("mojibake") characters.

---

## Intended use

This benchmark is built to support the SEAM methodology: comparing **final
answers, written reasoning, activations, and sparse-autoencoder features** across
the three conditions to test whether internal evidence can distinguish genuine
reasoning from shortcut-driven reasoning when surface behaviour is misleading.
Typical uses:

- Measure per-condition accuracy and **Answer Flip Rate** under misleading hints.
- Label responses by failure type (answer-flip, reasoning-flip, silent shortcut, confabulation).
- Provide matched activation/feature extraction pairs (clean vs. misleading).

It is **not** a general knowledge benchmark and the items are deliberately
adversarial; absolute accuracy numbers are only meaningful relative to the
matched conditions.

---

## Evaluation harness

The `seam/` package implements the full SEAM pipeline as decoupled stages that
pass JSONL between them. Heavy dependencies (`llama-cpp-python`,
`sentence-transformers`, `torch`, `scikit-learn`, `matplotlib`) are **lazily
imported** and listed in `requirements.txt`, so the dataset, grading, and
behavioural metrics on already-collected responses need only the standard
library.

```
run -> grade -> metrics -> report
            \-> detect    (shortcut detectors, RQ3: Fig 3 / Fig 4)
            \-> finetune  (validate RCS by fine-tuning a sentence-transformer)
```

### Hardware & models (single T4, 16 GB)

The registry is restricted to models that fit on **one NVIDIA T4** at Q4_K_M and
whose chain-of-thought the harness can record. **Qwen2.5-7B-Instruct** is the
reference model (default, and the only one whose activations the residual probe
uses).

| key | params | GGUF (≈Q4_K_M) | CoT | residual probe |
|-----|--------|----------------|-----|----------------|
| `qwen2.5-7b-instruct` (default) | 7B | ~4.7 GB | yes | yes (HF activations) |
| `llama-3.1-8b-instruct` | 8B | ~4.9 GB | yes | — |
| `mistral-7b-instruct-v0.3` | 7B | ~4.4 GB | yes | — |
| `deepseek-r1-distill-qwen-7b` | 7B | ~4.7 GB | yes (`<think>`) | — |

Download the GGUFs into `models/` (filenames match the registry):

```bash
pip install -U "huggingface_hub[cli]"
hf download paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF qwen2.5-7b-instruct-q4_k_m.gguf --local-dir models
hf download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models
hf download bartowski/Mistral-7B-Instruct-v0.3-GGUF Mistral-7B-Instruct-v0.3-Q4_K_M.gguf --local-dir models
hf download bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf --local-dir models
# Older hub versions: replace `hf download` with `huggingface-cli download`.
# For the residual probe you also need the non-GGUF checkpoint for HF activations:
hf download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
```

### Install llama.cpp with GPU support (do this first on a T4)

**`pip install llama-cpp-python` gives a CPU-only build** — the T4 stays idle and
each prompt takes minutes. Install the CUDA build instead:

```bash
# Colab / T4 (CUDA 12.x): a prebuilt CUDA wheel (fast, no compile)
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
# or compile with CUDA:
CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```

`run` prints `gpu_offload=True/False` at load and **warns loudly** if it detects a
CPU-only build. With the CUDA build + all layers offloaded (`--n-gpu-layers -1`,
the default), Qwen2.5-7B at Q4_K_M does each prompt in a few seconds on a T4
(the harness uses the model's chat template, so generation stops at EOS instead
of rambling to the token cap).

### Pipeline

```bash
python -m seam list-models                              # 0. the 4 T4 models

# 1. Collect CoT responses for all 3 variants via the model's chat template
#    (stops at EOS; max_tokens is just a safety cap). The tracker counts every
#    variant: 585 ticks = 195 problems x 3. --logprobs records token-logprob
#    confidence; --samples N adds self-consistency + an ECE confidence fallback.
python -m seam run --model qwen2.5-7b-instruct --models-dir models --out runs/qwen.jsonl

# 2. Grade: parse the final answer, mark correct/flip, detect shortcut-following.
python -m seam grade runs/qwen.jsonl --out graded/qwen.jsonl

# 3. Metrics: accuracy per condition, Shortcut Reliance Gap, AFR, shortcut rate,
#    condition sensitivity (KL), ECE, SEAM score. --rcs adds RCS; --bootstrap N
#    adds 95% CIs over base-problem IDs (Table 2).
python -m seam metrics "graded/*.jsonl" --out metrics.json --rcs --bootstrap 1000

# 4. Detectors (RQ3): lexical / TF-IDF / residual probe, grouped held-out AUROC.
#    Residual probe needs activations aligned to the misleading rows:
python -m seam detect "graded/*.jsonl" --out detectors.json \
       --activations qwen2.5-7b-instruct=acts/qwen_misleading.npy

# 5. Report: Table 2 (+CIs), Table 3 (detectors), Fig 2 gap heatmap, Fig 4 AUROC.
python -m seam report --metrics metrics.json --detectors detectors.json --out-dir report/

# 6. Validate RCS by fine-tuning the sentence-transformer; held-out ROC-AUC.
python -m seam finetune "graded/*.jsonl" --base-model sentence-transformers/all-MiniLM-L6-v2

# ONE COMMAND for the entire workflow (run -> grade -> metrics -> detect -> report,
# optionally + RCS fine-tune). Prints a banner + live progress bar per stage:
python -m seam pipeline --models-dir models --rcs --bootstrap 1000 --finetune --work seam_out
```

Every stage prints a real-time indicator: `run` and `grade` show a per-item
progress bar (tqdm in Colab/Jupyter, periodic `i/N (%) it/s eta` lines otherwise);
`metrics`/`detect` print `[stage k/N] <model>`; RCS encoding and fine-tuning show
their own bars. `pipeline` wraps them with `STAGE x/5` banners and ends with the
absolute path to all outputs.

> On PowerShell/cmd, quote globs (`"graded/*.jsonl"`) — the CLI expands them itself.

**Metrics implemented.**

| Block | Metric | Where |
|-------|--------|-------|
| Behavioural | accuracy per condition; Δhinted / Δmisleading | `metrics.accuracy_table` |
| | **Shortcut Reliance Gap** (misleading − clean trap-selection; Fig 2) | `metrics.shortcut_reliance_gap`, `gap_by_category` |
| | Answer Flip Rate (AFR); Shortcut Rate | `metrics.answer_flip_rate`, `metrics.shortcut_rate` |
| | **Bootstrap 95% CIs over base-problem IDs** (`--bootstrap N`) | `metrics.bootstrap_ci` |
| | Condition Sensitivity (KL); ECE; Self-consistency (`--samples N`) | `metrics.condition_sensitivity` / `expected_calibration_error` / `self_consistency` |
| | Failure taxonomy (answer-flip / reasoning-flip / silent-shortcut / confabulation) | `metrics.failure_taxonomy` |
| Detectors (RQ3) | CoT lexical / TF-IDF / residual probe — grouped held-out AUROC & AUPRC (Fig 4) | `detectors.compare` |
| | Flagged-among-correct = observable RWRR rate (Fig 3) | `detectors.compare` |
| Semantic | RCS = cosine(CoT_clean, CoT_misleading); fine-tuning + ROC-AUC | `semantic.rcs_scores`, `semantic.finetune` |
| | BERTScore, NLI entailment, coverage (optional) | `semantic.bertscore_f1` / `nli_entailment` / `coverage_score` |
| Mechanistic | activation silhouette; SAE feature delta; patching logit diff; causal localization | `mechanistic.*` |
| Composite | SEAM score = transparent weighted blend (weights in `config.py`) | `metrics.summarize` |

The mechanistic and residual-probe functions are backend-agnostic — they take
numpy arrays you extract from your models (HF hooks on the `hf_id` checkpoint, or
llama.cpp embeddings). `python -m seam mech-selftest` and `python -m seam
det-selftest` exercise the mechanistic and detector code on synthetic data, so
both modules are self-checking. *Note:* the paper's `Counterfactual` condition is
not in this dataset (variants are `clean`/`hinted`/`misleading`); `hinted` is the
"Helpful" column.

### Outputs — where results are stored and how to read them

Every stage writes to the path you pass with `--out` / `--out-dir`; nothing is
hidden. With the commands above:

| File | Written by | Contents |
|------|-----------|----------|
| `runs/<model>.jsonl` | `run` | one row per (problem, variant): `response`, `cot`, `final_answer`, `confidence` |
| `graded/<model>.jsonl` | `grade` | the above + `correct`, `label`, `followed_shortcut` |
| `metrics.json` | `metrics` | per-model summary: accuracies (+CIs), gap, AFR, ECE, RCS, SEAM, failure taxonomy |
| `detectors.json` | `detect` | per-model detector AUROC / AUPRC / flagged-among-correct |
| `report/table2_accuracy.md` | `report` | Table 2 (accuracy × condition + Shortcut Reliance Gap, with CIs) |
| `report/table3_detectors.md` | `report` | Table 3 (detector comparison) |
| `report/table_failures.md` | `report` | failure-type taxonomy |
| `report/summary.csv` | `report` | flat per-model table (open in any spreadsheet) |
| `report/fig2_gap_heatmap.png` | `report` | Shortcut Reliance Gap by model × domain |
| `report/fig3_failures.png`, `fig4_detectors.png`, `fig8_seam_scatter.png` | `report` | figures |
| `models/rcs-ft/` | `finetune` | fine-tuned sentence-transformer + before/after AUROC |

`report/` and `summary.csv` are the human-facing outputs; `metrics.json` /
`detectors.json` are the machine-readable source for the paper. These output
dirs are git-ignored. Debug a specific model by grepping its graded file, e.g.
`grep '"label": "shortcut"' graded/qwen.jsonl`.

## Datasheet

**Motivation.** Created to study right-answer/wrong-reason behaviour in
open-weight reasoning models and whether interpretability tools can detect it.
Surface-level "got the right answer" can hide shortcut reasoning; matched
conditions make that detectable.

**Composition.** 195 self-contained text reasoning problems in 11 categories,
each with three prompt variants and a single gold answer. Problems are short
puzzles (cognitive-reflection items, probability/Bayes, rates, logic/syllogisms,
algebra, combinatorics, geometry, sequences, causal reasoning, word problems,
number theory). Many are classic, widely-circulated reasoning puzzles
(bat-and-ball, Monty Hall, Linda, the birthday problem, Russell's barber, the
water-lily lake, the snail-in-the-well); SEAM's contribution is the matched
hint conditions, the bias labelling, and the machine-checkable answer metadata.
No personal data is included.

**Collection / curation.** Items were authored and curated for this project.
Each problem was assigned a `bias` label naming the reasoning trap its misleading
hint exploits. Answers were verified by hand for the clean variant; the validator
re-checks that every answer parses according to its declared type and that the
misleading answer never coincides with the gold answer.

**Preprocessing / cleaning.** This release was produced by cleaning an earlier
draft. Changes:
- Repaired pervasive UTF-8 double-encoding ("mojibake") in hints (e.g. `Ã`, `Â¢`, `Ï`, stray `â`); math is now clean UTF-8.
- Corrected a wrong gold answer in `rate_003` (fill 1/4 h − drain 1/12 h = 1/6 h ⇒ **6 hours**, previously recorded as 3).
- Rebuilt a malformed record (`logic_021`) whose answer metadata was corrupt (string `answer_keywords`, string tolerance, `misleading_answer: "integer"`) and whose question was under-specified; it is now a well-posed pigeonhole item.
- Disambiguated `logic_015`, which previously admitted multiple valid solutions, so the answer is uniquely determined.
- Removed author meta-commentary that had leaked into ~20 misleading hints (self-corrections such as "wait, that gives…", "Re-mislead:"), and regenerated those misleading hints to be internally coherent and to point cleanly at a single wrong answer.
- De-duplicated five exact cross-category repeats (the `word_problems` copies of `alg_018`, `alg_020`, `alg_021`, `alg_023`, `alg_025`), reducing the set from 200 to 195. `crt_021` is intentionally retained as a paraphrase of `crt_009` for surface-form-robustness analysis.
- Normalised the schema (consistent `answer_type`, `answer_keywords`, `answer_tolerance`) and added a JSON Schema + validator.

**Uses.** Intended for interpretability and reasoning-robustness research. Not
suitable as a measure of general capability or factual knowledge.

**Distribution & license.** Released under the MIT License (see `LICENSE`).

**Maintenance.** `problems.json` is edited directly and gated by
`tools/validate.py`; contributions must keep the validator green.

### Known limitations

- `bias` labels are author-assigned categorisations of the *intended* trap, not
  empirically validated failure modes.
- Free-text grading via `answer_keywords` is a heuristic; for borderline outputs,
  pair it with human or model-based judging.
- A few causal-reasoning items use coarse real-world statistics (e.g. lightning
  vs. shark deaths) whose exact figures vary by source; the qualitative ordering
  is what is graded.
- Several items are well-known puzzles likely present in pre-training corpora;
  this is by design (the question is held fixed across conditions), but absolute
  accuracy should be interpreted with that in mind.

---

## Reproducibility

```bash
python tools/validate.py        # structure, schema, consistency, mojibake, stats
```

`problems.json` is committed UTF-8 with `\n` line endings and stable key
ordering; `tools/validate.py` is the single source of integrity checks and
exits non-zero on any failure (use it as a CI gate).

---

## Citation

If you use this benchmark, please cite it (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@misc{seam_benchmark,
  title  = {SEAM: Detecting Right-Answer, Wrong-Reason Behavior in Open-Weight Reasoning Models},
  author = {The SEAM Authors},
  year   = {2026},
  note   = {Benchmark of clean, hinted, and misleadingly-hinted reasoning problems},
  howpublished = {GitHub repository}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
