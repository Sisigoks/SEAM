"""Command-line interface: run -> grade -> metrics -> report (+ finetune)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

from . import data, grading, metrics, report, runner
from .config import MODELS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _cmd_models(_):
    print("Available models (use with `run --model <key>`):")
    for k, v in MODELS.items():
        print(f"  {k:<32} {v['repo']}  (think={v['think']})")


def _cmd_run(a):
    ds = data.load_dataset(a.dataset)
    cats = a.categories.split(",") if a.categories else None
    runner.run(ds, a.model, a.out, gguf_path=a.gguf, models_dir=a.models_dir,
               samples=a.samples, limit=a.limit, categories=cats)


def _cmd_grade(a):
    rows = [grading.grade_row(r) for r in data.read_jsonl(a.infile)]
    data.write_jsonl(a.out, rows)
    acc = metrics.accuracy_table(rows)
    print(f"Graded {len(rows)} rows -> {a.out}  | acc " +
          " ".join(f"{k}={acc[k]:.3f}" for k in acc))


def _load_rows(paths):
    import glob
    expanded = []
    for p in paths:
        hits = glob.glob(p)
        expanded.extend(sorted(hits) if hits else [p])
    rows = []
    for p in expanded:
        rows.extend(data.read_jsonl(p))
    return rows


def _cmd_metrics(a):
    rows = _load_rows(a.infiles)
    by_model = defaultdict(list)
    for r in rows:
        by_model[r.get("model", "unknown")].append(r)

    summaries = []
    for model, mrows in by_model.items():
        rcs = None
        if a.rcs:
            from . import semantic
            rcs = semantic.rcs_scores(mrows, model_name=a.rcs_model)
        summaries.append(metrics.summarize(mrows, model, rcs=rcs))

    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"Wrote metrics for {len(summaries)} model(s) -> {a.out}\n")
    print(report.table_accuracy(summaries))


def _cmd_report(a):
    with open(a.metrics, encoding="utf-8") as f:
        summaries = json.load(f)
    report.generate(summaries, a.out_dir)


def _cmd_finetune(a):
    from . import semantic
    rows = [grading.grade_row(r) if "label" not in r else r
            for r in _load_rows(a.infiles)]
    res = semantic.finetune(rows, base_model=a.base_model, out_dir=a.out_dir,
                            epochs=a.epochs)
    print(json.dumps(res, indent=2))


def _cmd_mech_selftest(_):
    from . import mechanistic
    print(json.dumps(mechanistic.selftest(), indent=2))


def _cmd_pipeline(a):
    """End-to-end run -> grade -> metrics -> report across several models.

    GGUFs are resolved from --models-dir using each model's registered filename.
    """
    ds = data.load_dataset(a.dataset)
    models = a.models.split(",") if a.models else list(MODELS)
    os.makedirs(os.path.join(a.work, "graded"), exist_ok=True)
    graded_paths = []
    for m in models:
        rp = os.path.join(a.work, "runs", f"{m}.jsonl")
        runner.run(ds, m, rp, models_dir=a.models_dir, limit=a.limit, progress=False)
        gp = os.path.join(a.work, "graded", f"{m}.jsonl")
        data.write_jsonl(gp, [grading.grade_row(r) for r in data.read_jsonl(rp)])
        graded_paths.append(gp)
    rows = _load_rows(graded_paths)
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    summaries = [metrics.summarize(v, k) for k, v in by_model.items()]
    mpath = os.path.join(a.work, "metrics.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    report.generate(summaries, os.path.join(a.work, "report"))
    print("\n" + report.table_accuracy(summaries))
    print("\nFailure taxonomy:\n" + report.table_failures(summaries))


def build_parser():
    p = argparse.ArgumentParser(prog="seam", description="SEAM evaluation harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-models").set_defaults(func=_cmd_models)

    r = sub.add_parser("run", help="collect model responses via llama.cpp (GGUF)")
    r.add_argument("--model", required=True)
    r.add_argument("--dataset", default=data.DEFAULT_DATASET)
    r.add_argument("--out", required=True)
    r.add_argument("--gguf", default=None, help="path to a local .gguf")
    r.add_argument("--models-dir", default=None,
                   help="dir of GGUFs named per the model registry")
    r.add_argument("--samples", type=int, default=1)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--categories", default=None, help="comma-separated filter")
    r.set_defaults(func=_cmd_run)

    g = sub.add_parser("grade", help="grade responses, detect flips/shortcuts")
    g.add_argument("infile"); g.add_argument("--out", required=True)
    g.set_defaults(func=_cmd_grade)

    m = sub.add_parser("metrics", help="compute behavioural metrics + SEAM score")
    m.add_argument("infiles", nargs="+")
    m.add_argument("--out", default="metrics.json")
    m.add_argument("--rcs", action="store_true", help="compute RCS (needs sentence-transformers)")
    m.add_argument("--rcs-model", default=None)
    m.set_defaults(func=lambda a: _cmd_metrics(_with_rcs_default(a)))

    rp = sub.add_parser("report", help="render tables + figures")
    rp.add_argument("--metrics", default="metrics.json"); rp.add_argument("--out-dir", default="report")
    rp.set_defaults(func=_cmd_report)

    ft = sub.add_parser("finetune", help="fine-tune the RCS sentence-transformer")
    ft.add_argument("infiles", nargs="+")
    ft.add_argument("--base-model", default=None)
    ft.add_argument("--out-dir", default="models/rcs-ft")
    ft.add_argument("--epochs", type=int, default=1)
    ft.set_defaults(func=lambda a: _cmd_finetune(_with_base_default(a)))

    sub.add_parser("mech-selftest", help="run mechanistic metrics on synthetic data"
                   ).set_defaults(func=_cmd_mech_selftest)

    pl = sub.add_parser("pipeline", help="end-to-end run -> grade -> metrics -> report")
    pl.add_argument("--work", default="seam_out"); pl.add_argument("--models", default=None)
    pl.add_argument("--models-dir", default=None,
                    help="dir of GGUFs named per the model registry")
    pl.add_argument("--dataset", default=data.DEFAULT_DATASET)
    pl.add_argument("--limit", type=int, default=None)
    pl.set_defaults(func=_cmd_pipeline)
    return p


def _with_rcs_default(a):
    from .config import RCS_MODEL
    if a.rcs_model is None:
        a.rcs_model = RCS_MODEL
    return a


def _with_base_default(a):
    from .config import RCS_MODEL_SMALL
    if a.base_model is None:
        a.base_model = RCS_MODEL_SMALL
    return a


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
