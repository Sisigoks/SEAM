# SEAM: Shortcut Evidence and Activation Mapping

A benchmark and analysis harness for studying **right-answer, wrong-reason**
behaviour in open-weight reasoning models. SEAM pairs each problem with matched
variants that share an identical question and gold answer, isolating the effect
of a *hint* on a model's behaviour and on its internal representations. The
harness collects chain-of-thought responses, grades them, and computes
behavioural and mechanistic metrics — including a residual-stream probe that
tests whether internal evidence detects shortcut reliance that the written
reasoning conceals.

## Highlights

- **Robustness varies ~4× across same-size (7–8B) models, and clean accuracy
  does not predict it.** Under a misleading hint, accuracy ranges from 0.71
  (Qwen2.5-7B) down to 0.20 (Mistral-7B); the Shortcut Reliance Gap spans
  0.13–0.60. Llama-3.1-8B is second on clean accuracy yet abandons a correct
  answer on 55% of items it originally solved.
- **Models stay confident while flipping.** The correlation between a model's
  clean-answer confidence and whether it flips under a misleading hint is ≈ 0 for
  every model (−0.09 to +0.10): expressed confidence does not predict robustness.
- **Internal signals outperform text.** A per-model residual-stream probe reaches
  AUROC 0.82 on Mistral — a +0.34 gain over the best text detector — and
  localizes shortcut detectability to mid layers (best layer 13–15).

## Results

All numbers are over 195 problems × 3 variants per model (585 prompts).
Brackets are 95% bootstrap confidence intervals over base-problem IDs.

### Behaviour by condition

| Model | Clean acc. | Hinted acc. | Misleading acc. | Shortcut Reliance Gap | Answer-Flip Rate | Shortcut rate | SEAM |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 0.862 [0.810, 0.908] | 0.908 | 0.708 [0.641, 0.774] | 0.144 [0.102, 0.186] | 0.202 | 0.195 | 0.838 |
| DeepSeek-R1-Distill-Qwen-7B | 0.744 [0.682, 0.800] | 0.774 | 0.600 [0.533, 0.677] | 0.133 [0.087, 0.178] | 0.234 | 0.164 | 0.837 |
| Llama-3.1-8B-Instruct | 0.769 [0.713, 0.831] | 0.908 | 0.400 [0.333, 0.472] | 0.318 [0.264, 0.368] | 0.553 | 0.395 | 0.635 |
| Mistral-7B-Instruct-v0.3 | 0.451 [0.385, 0.528] | 0.897 | 0.200 [0.149, 0.256] | 0.595 [0.536, 0.654] | 0.659 | 0.687 | 0.462 |

*Shortcut Reliance Gap* = P(select the trap answer | misleading) − P(select it |
clean). *Answer-Flip Rate* = fraction of clean-correct items answered wrongly
under the misleading hint. A correct hint (Hinted) helps every model, but the
same models differ sharply under a misleading one.

### Confidence does not predict susceptibility

Bucketing clean problems by the model's own confidence and measuring the flip
rate in each tertile, the point-biserial correlation between confidence and
flipping is statistically negligible for all models.

| Model | r(confidence, flip) | Flip rate (low / med / high confidence) |
|---|---|---|
| Qwen2.5-7B-Instruct | −0.00 | 0.21 / 0.25 / 0.14 |
| DeepSeek-R1-Distill-Qwen-7B | −0.08 | 0.25 / 0.23 / 0.22 |
| Llama-3.1-8B-Instruct | +0.10 | 0.50 / 0.68 / 0.48 |
| Mistral-7B-Instruct-v0.3 | −0.09 | 0.69 / 0.72 / 0.57 |

### Detecting shortcut reliance: internal vs. text

Detectors predict observable shortcut-following on the misleading condition,
scored with grouped held-out AUROC (folds split by base problem). The
residual-stream probe is evaluated per model on its own activations; *probe gain*
is its best-layer AUROC minus the stronger text detector.

| Model | CoT lexical | CoT TF–IDF | Residual probe | Best layer | Probe gain |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 0.646 | 0.728 | 0.757 | 13 | +0.029 |
| DeepSeek-R1-Distill-Qwen-7B | 0.759 | 0.676 | 0.743 | 15 | −0.016 |
| Llama-3.1-8B-Instruct | 0.577 | 0.569 | 0.734 | 29 | +0.156 |
| Mistral-7B-Instruct-v0.3 | 0.475 | 0.442 | 0.818 | 15 | +0.342 |

Text detectors are weak and uneven (TF–IDF AUROC 0.44–0.73); the residual probe
is the strongest signal on the three models that flip most, and largest exactly
where text collapses (Mistral). Per-layer sweeps place the most informative
representation in mid layers for the Qwen-family and Mistral models.

## Dataset

195 self-contained reasoning problems across 11 categories, each in three matched
variants built from one base question:

| Variant | Prompt | Question it answers |
|---|---|---|
| `clean` | base question only | baseline reasoning |
| `hinted` | base question + a **correct** hint | does a valid hint help? |
| `misleading` | base question + a **plausible but wrong** hint | does the model follow the shortcut? |

`hinted`/`misleading` prompts are exactly `clean.prompt + "\n\nHint: " + <hint>`,
so the hint text is the only controlled difference. The gold `answer` is identical
across variants; the misleading variant adds `misleading_answer`, the wrong answer
its hint argues for. A move from `answer` (clean) to `misleading_answer`
(misleading) is shortcut-following. An optional model-specific `counterfactual`
condition re-confronts a model with its own clean chain of thought.

```jsonc
{ "id": "prob_015", "category": "probability", "difficulty": "hard",
  "bias": "base_rate_neglect",
  "variants": {
    "clean":      { "prompt": "...", "answer": "41", "answer_type": "integer", "answer_tolerance": 3 },
    "hinted":     { "prompt": "...\n\nHint: ...", "answer": "41", "answer_type": "integer", "answer_tolerance": 3 },
    "misleading": { "prompt": "...\n\nHint: ...", "answer": "41", "answer_type": "integer",
                    "answer_tolerance": 3, "misleading_answer": "80" } } }
```

`answer_type ∈ {integer, fraction, text, choice}`; `text` answers carry
`answer_keywords`. Composition: 585 prompts; difficulty 32/145/18 (easy/med/hard);
23 reasoning-trap (`bias`) labels. The full schema is in
[`schema/problem.schema.json`](schema/problem.schema.json); `tools/validate.py`
checks structure, schema conformance, cross-variant consistency, and encoding,
and prints the per-category breakdown.

## Method

Each model is run on all three variants; responses are graded into
`correct / shortcut / other_wrong / refused`. The harness then computes:

- **Behavioural** — accuracy per condition; Shortcut Reliance Gap (overall and by
  category and by bias); Answer-Flip Rate; shortcut rate and specificity;
  condition sensitivity; calibration error; self-consistency; a failure taxonomy
  (answer-flip / reasoning-flip / silent-shortcut / confabulation); bootstrap 95%
  CIs over base-problem IDs.
- **Semantic** — Reasoning-Consistency Score (cosine of clean vs. misleading CoT
  embeddings); optional NLI-entailment and coverage; the score can be validated by
  fine-tuning a sentence-transformer on consistent vs. flipped CoT pairs.
- **Confidence → susceptibility** — flip rate by confidence tertile and the
  point-biserial correlation above.
- **Mechanistic** — residual-stream extraction from each model's full-precision
  checkpoint; a logistic probe trained per layer with grouped held-out CV
  (best-layer AUROC, probe gain, layer localization); clean-vs-misleading
  activation separability. Functions for SAE-feature deltas and activation
  patching are included for arrays the user supplies.

A transparent weighted blend of the behavioural (and, when activations are
provided, mechanistic) sub-scores gives the composite **SEAM** score.

## Reproduction

```bash
pip install -r requirements.txt
# GPU inference needs a CUDA build of llama.cpp (the default PyPI wheel is CPU-only):
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

Download the GGUFs into `models/` (`python -m seam list-models` shows the
registry and expected filenames), then run the full study in one command:

```bash
python -m seam pipeline --models-dir models --probe --rcs --bootstrap 1000 --work seam_out
```

The residual probe is computed per model from its own activations. If you run the
stages individually, extract once per model and pass each model its **own** file:

```bash
python -m seam detect "seam_out/graded/*.jsonl" --out seam_out/detectors.json \
  --activations qwen2.5-7b-instruct=acts/qwen_misleading.npz \
  --activations llama-3.1-8b-instruct=acts/llama-3.1-8b-instruct_misleading.npz \
  --activations mistral-7b-instruct-v0.3=acts/mistral-7b-instruct-v0.3_misleading.npz \
  --activations deepseek-r1-distill-qwen-7b=acts/deepseek-r1-distill-qwen-7b_misleading.npz
python -m seam report --metrics seam_out/metrics.json --detectors seam_out/detectors.json \
       --confidence seam_out/confidence.json --out-dir seam_out/report
```

Outputs land in `seam_out/`: `runs/`, `graded/`, `metrics.json`,
`detectors.json`, `confidence.json`, and `report/` (tables + black-and-white
figures). On PowerShell, quote globs (`"seam_out/graded/*.jsonl"`).

**GPU environment notes.** Do not reinstall `torch` on a hosted GPU image — a
mismatched CUDA/NCCL build breaks imports (`undefined symbol: ncclCommResume`).
Activation extraction loads each model's `hf_id` checkpoint (Llama and Mistral are
gated); run `hf auth login` once. `requirements.txt` pins `transformers<5`, whose
`from_pretrained` API the extractor expects.

## Datasheet and limitations

The benchmark targets interpretability and reasoning-robustness research; it is
**not** a measure of general capability, and the items are deliberately
adversarial, so absolute accuracy is meaningful only *relative to the matched
conditions*. Problems are short, self-contained puzzles (cognitive-reflection,
probability/Bayes, logic, algebra, combinatorics, geometry, sequences, causal
reasoning, word problems, number theory); many are classic puzzles, and the
contribution is the matched hint conditions, the bias labelling, and the
machine-checkable answer metadata. No personal data is included. Clean answers
were hand-verified and are gated by `tools/validate.py`.

- `bias` labels are author-assigned categorisations of the *intended* trap, not
  empirically validated failure modes.
- Free-text grading via `answer_keywords` is heuristic; pair with human or
  model-based judging at the margin.
- Several items are well-known puzzles likely seen in pre-training; this is by
  design (the question is held fixed across conditions) but caps how surprising
  absolute accuracy can be.
- Behaviour is collected from Q4_K_M GGUF models; activations come from the
  full-precision checkpoints. Use `--backend hf` to generate behaviour from the
  same checkpoint the probe reads when a fully matched setting is required.

## Citation

```bibtex
@misc{seam_benchmark,
  title  = {SEAM: Detecting Right-Answer, Wrong-Reason Behavior in Open-Weight Reasoning Models},
  author = {The SEAM Authors},
  year   = {2026},
  howpublished = {GitHub repository}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
