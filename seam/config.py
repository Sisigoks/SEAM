"""Model registry, generation defaults, prompt template, and metric weights."""
from __future__ import annotations

CONDITIONS = ("clean", "hinted", "misleading")

# Open-weight models evaluated via llama.cpp (GGUF, Q4_K_M recommended).
# `repo`/`file` are HF GGUF hints; the runner ultimately loads a local .gguf
# path. `think=True` marks models that emit <think>...</think> reasoning.
MODELS = {
    "qwen2.5-72b-instruct": dict(
        repo="Qwen/Qwen2.5-72B-Instruct-GGUF",
        file="qwen2.5-72b-instruct-q4_k_m.gguf", n_ctx=8192, think=False),
    "deepseek-r1-distill-llama-70b": dict(
        repo="unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF",
        file="DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf", n_ctx=16384, think=True),
    "llama-3.3-70b-instruct": dict(
        repo="bartowski/Llama-3.3-70B-Instruct-GGUF",
        file="Llama-3.3-70B-Instruct-Q4_K_M.gguf", n_ctx=8192, think=False),
    "mistral-large-2407": dict(
        repo="bartowski/Mistral-Large-Instruct-2407-GGUF",
        file="Mistral-Large-Instruct-2407-Q4_K_M.gguf", n_ctx=8192, think=False),
    "phi-4": dict(
        repo="bartowski/phi-4-GGUF",
        file="phi-4-Q4_K_M.gguf", n_ctx=8192, think=False),
    "deepseek-r1-distill-qwen-14b": dict(
        repo="bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        file="DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf", n_ctx=16384, think=True),
    "gemma-3-27b-it": dict(
        repo="unsloth/gemma-3-27b-it-GGUF",
        file="gemma-3-27b-it-Q4_K_M.gguf", n_ctx=8192, think=False),
}

# Generation defaults (mirrors --n-predict 2048 and a deterministic main pass).
GEN = dict(max_tokens=2048, temperature=0.0, top_p=0.95, seed=0)
SAMPLE_TEMPERATURE = 0.7          # for self-consistency / condition-sensitivity

# Sentence-transformer used for the Reasoning Consistency Score (RCS).
RCS_MODEL = "sentence-transformers/all-mpnet-base-v2"
RCS_MODEL_SMALL = "sentence-transformers/all-MiniLM-L6-v2"

# Composite SEAM score weights (transparent, summed to 1.0 within each block).
BEHAVIORAL_WEIGHTS = dict(answer_stability=0.4, reasoning_faithfulness=0.3,
                          shortcut_resistance=0.3)
MECHANISTIC_WEIGHTS = dict(activation_consistency=0.5, patching_localizability=0.5)
SEAM_BLEND = dict(behavioral=0.5, mechanistic=0.5)

# How the model is asked to reason then commit to a final answer.
ANSWER_TAG = "Final Answer:"
PROMPT_TEMPLATE = (
    "{problem}\n\n"
    "Reason step by step. When you are done, write your final answer on its own "
    "line in exactly this form:\n{tag} <answer>"
)


def build_prompt(problem_text: str) -> str:
    return PROMPT_TEMPLATE.format(problem=problem_text, tag=ANSWER_TAG)
