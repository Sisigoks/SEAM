"""SEAM evaluation harness.

A small, modular pipeline for the SEAM benchmark:

    run  ->  grade  ->  metrics  ->  report
                                 +-> finetune (RCS validation)

Each stage reads/writes JSONL so stages are decoupled and independently
runnable. Heavy dependencies (llama-cpp-python, sentence-transformers, torch,
scikit-learn, matplotlib) are imported lazily, so grading, metrics and reporting
on already-collected responses need only the standard library.
"""
__version__ = "1.0.0"
