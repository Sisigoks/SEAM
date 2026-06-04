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
tools/build_dataset.py        # deterministic generator that emits problems.json
tools/validate.py             # standalone validator + summary report (CI gate)
seam/                         # evaluation harness (see "Evaluation harness")
  config.py                   #   model registry, prompt template, metric weights
  runner.py                   #   llama.cpp (GGUF) inference
  parsing.py grading.py       #   CoT/answer parsing; answer matching + labels
  metrics.py                  #   behavioural metrics + composite SEAM score
  semantic.py                 #   RCS (sentence-transformer) + RCS fine-tuning
  mechanistic.py              #   silhouette / SAE-delta / patching / localization
  report.py cli.py            #   tables + figures; `python -m seam` entry point
requirements.txt              # optional, lazily-imported dependencies
LICENSE  CITATION.cff  README.md
```

`problems.json` is the source of truth. `tools/build_dataset.py` regenerates it
deterministically; `tools/validate.py` checks the committed file independently.

---

## Quick start

```bash
# Validate the dataset and print a summary (exits non-zero on any error).
python tools/validate.py

# Regenerate problems.json from the generator (optional; output is byte-stable).
python tools/build_dataset.py
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
run  ->  grade  ->  metrics  ->  report
                            +->  finetune  (validate RCS by fine-tuning a sentence-transformer)
```

```bash
# 0. See the prescribed open-weight models (Qwen2.5-72B, DeepSeek-R1 distills,
#    Llama-3.3-70B, Mistral-Large, Phi-4, Gemma-3-27B).
python -m seam list-models

# 1. Collect chain-of-thought responses for all 3 variants of every problem
#    (llama.cpp / GGUF, Q4_K_M, n-predict 2048):
python -m seam run --model phi-4 --gguf /models/phi-4-Q4_K_M.gguf --out runs/phi-4.jsonl

# 2. Grade: parse the final answer, mark correct/flip, detect shortcut-following.
python -m seam grade runs/phi-4.jsonl --out graded/phi-4.jsonl

# 3. Metrics: accuracy per condition, Answer Flip Rate, shortcut rate, condition
#    sensitivity (KL), ECE, and the composite SEAM score. Add --rcs to compute the
#    Reasoning Consistency Score (needs sentence-transformers).
python -m seam metrics graded/*.jsonl --out metrics.json --rcs

# 4. Report: Table 2 (accuracy x condition), Table 3 (failure taxonomy), figures.
python -m seam report --metrics metrics.json --out-dir report/

# 5. Validate RCS by fine-tuning the sentence-transformer on consistent vs.
#    flipped CoT pairs; reports held-out ROC-AUC before/after.
python -m seam finetune graded/*.jsonl --base-model sentence-transformers/all-MiniLM-L6-v2

# One-shot stages 1-4 across several models (GGUFs resolved from --models-dir):
python -m seam pipeline --models-dir /models --models phi-4,gemma-3-27b-it --work seam_out
```

**Metrics implemented.**

| Block | Metric | Where |
|-------|--------|-------|
| Behavioural | accuracy per condition; Δhinted / Δmisleading | `metrics.accuracy_table` |
| | Answer Flip Rate (AFR); Shortcut Rate | `metrics.answer_flip_rate`, `metrics.shortcut_rate` |
| | Condition Sensitivity (KL over answer distributions) | `metrics.condition_sensitivity` |
| | Expected Calibration Error (ECE) | `metrics.expected_calibration_error` |
| | Self-consistency (with `--samples N`) | `metrics.self_consistency` |
| | Failure taxonomy (answer-flip / reasoning-flip / silent-shortcut / confabulation) | `metrics.failure_taxonomy` |
| Semantic | RCS = cosine(CoT_clean, CoT_misleading) via sentence-transformer | `semantic.rcs_scores` |
| | RCS fine-tuning + ROC-AUC validation | `semantic.finetune` |
| | BERTScore, NLI entailment, coverage (optional) | `semantic.bertscore_f1` / `nli_entailment` / `coverage_score` |
| Mechanistic | activation silhouette (PCA); SAE feature delta | `mechanistic.activation_silhouette`, `sae_feature_delta` |
| | activation-patching logit diff; causal localization (top-5) | `mechanistic.patching_logit_diff`, `causal_localization` |
| Composite | SEAM score = transparent weighted blend (weights in `config.py`) | `metrics.summarize` |

The mechanistic functions are backend-agnostic — they take numpy arrays you
extract from your models (HF hooks or llama.cpp embeddings). `python -m seam
mech-selftest` exercises them on synthetic data so the module is self-checking.

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

**Maintenance.** The dataset is generated by `tools/build_dataset.py` and gated
by `tools/validate.py`; contributions should keep both green.

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
python tools/build_dataset.py   # regenerate problems.json (deterministic)
python tools/validate.py        # structure, schema, consistency, mojibake, stats
```

`build_dataset.py` writes UTF-8 with `\n` line endings and stable key ordering,
so regeneration is byte-for-byte stable.

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
