"""Command-line interface: run -> grade -> metrics -> report (+ finetune)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

from . import data, grading, metrics, report, runner
from .config import MODELS
from .progress import track

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _banner(msg):
    print("\n" + "=" * 64 + f"\n  {msg}\n" + "=" * 64, flush=True)


def _cmd_models(_):
    from .config import DEFAULT_MODEL
    print("T4-runnable models (use with `run --model <key>`):")
    for k, v in MODELS.items():
        tag = " [default]" if k == DEFAULT_MODEL else ""
        print(f"  {k:<30}{tag:<11} ~{v.get('vram_gb','?')}GB  think={v['think']}  "
              f"probe={v.get('activations', False)}  {v['repo']}")


def _cmd_run(a):
    ds = data.load_dataset(a.dataset)
    cats = a.categories.split(",") if a.categories else None
    items = data.read_jsonl(a.items) if a.items else None
    runner.run(ds, a.model, a.out, items=items, backend=a.backend, dtype=a.dtype,
               load_4bit=a.load_4bit, gguf_path=a.gguf, models_dir=a.models_dir,
               samples=a.samples, want_logprobs=a.logprobs, n_gpu_layers=a.n_gpu_layers,
               limit=a.limit, categories=cats)


def _cmd_counterfactual(a):
    from . import counterfactual as cf
    rows = [grading.grade_row(r) if "label" not in r else r for r in _load_rows(a.infiles)]
    if a.model:
        rows = [r for r in rows if r.get("model") == a.model]
    items = cf.build_items(rows)
    data.write_jsonl(a.out, items)
    print(f"Built {len(items)} counterfactual items -> {a.out}\n"
          f"Next: seam run --model <key> --models-dir models --items {a.out} --out runs/cf.jsonl")


def _cmd_grade(a):
    src = data.read_jsonl(a.infile)
    rows = [grading.grade_row(r) for r in track(src, desc=f"grade:{os.path.basename(a.infile)}")]
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
    for n, (model, mrows) in enumerate(by_model.items(), 1):
        print(f"[metrics {n}/{len(by_model)}] {model}: {len(mrows)} rows"
              + (f", bootstrap x{a.bootstrap}" if a.bootstrap else ""), flush=True)
        rcs = None
        if a.rcs:
            from . import semantic
            rcs = semantic.rcs_scores(mrows, model_name=a.rcs_model)
        summaries.append(metrics.summarize(mrows, model, rcs=rcs, ci=a.bootstrap))

    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"Wrote metrics for {len(summaries)} model(s) -> {a.out}\n")
    print(report.table_accuracy(summaries))


def _cmd_detect(a):
    from . import detectors, activations as act
    rows = [grading.grade_row(r) if "label" not in r else r for r in _load_rows(a.infiles)]
    by_model = defaultdict(list)
    for r in rows:
        by_model[r.get("model", "unknown")].append(r)
    acts = {}
    for spec in (a.activations or []):                   # "model=path.npz|.npy"
        m, path = spec.split("=", 1)
        X, ids, _ = act.load(path)
        acts[m] = (X, ids)
    results = {}
    for n, (m, mrows) in enumerate(by_model.items(), 1):
        print(f"[detect {n}/{len(by_model)}] {m}: {len(mrows)} rows", flush=True)
        X, ids = acts.get(m, (None, None))
        results[m] = detectors.compare(mrows, activations=X, activation_ids=ids)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote detector results for {len(results)} model(s) -> {a.out}\n")
    print(report.table_detectors(results))


def _cmd_det_selftest(_):
    from . import detectors
    print(json.dumps(detectors.selftest(), indent=2))


def _cmd_extract(a):
    from . import activations as act
    rows = [grading.grade_row(r) if "label" not in r else r for r in _load_rows(a.infiles)]
    rows = [r for r in rows if r.get("condition") == a.condition
            and (not a.model or r.get("model") == a.model)]
    if not rows:
        print("No matching rows (check --model / --condition).")
        return 1
    hf_id = a.hf_id or MODELS.get(a.model, {}).get("hf_id")
    if not hf_id:
        print("No hf_id for this model; pass --hf-id <HF checkpoint>.")
        return 1
    layers = [int(x) for x in a.layers.split(",")] if a.layers else None
    X, ids, lyr = act.extract_activations(rows, hf_id, layers=layers, dtype=a.dtype,
                                          load_4bit=a.load_4bit)
    act.save(a.out, X, ids, lyr)


def _cmd_confidence(a):
    from . import confidence as conf
    rows = [grading.grade_row(r) if "label" not in r else r for r in _load_rows(a.infiles)]
    by_model = defaultdict(list)
    for r in rows:
        by_model[r.get("model", "unknown")].append(r)
    analysis = conf.analyze(by_model, proxy=a.proxy, bins=a.bins)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    if a.fig:
        conf.plot(analysis, a.fig)
    print(f"Wrote confidence analysis -> {a.out}\n")
    for m, res in analysis.items():
        bs = ", ".join(f"{b['bucket']}:{b['flip_rate']:.2f}" for b in res.get("buckets", []))
        print(f"  {m}: r(conf,flip)={res.get('point_biserial'):.3f} | n={res.get('n')} | flip[{bs}]")


def _cmd_report(a):
    with open(a.metrics, encoding="utf-8") as f:
        summaries = json.load(f)
    detectors = None
    if a.detectors and os.path.exists(a.detectors):
        with open(a.detectors, encoding="utf-8") as f:
            detectors = json.load(f)
    confidence = None
    if a.confidence and os.path.exists(a.confidence):
        with open(a.confidence, encoding="utf-8") as f:
            confidence = json.load(f)
    report.generate(summaries, a.out_dir, detectors=detectors, confidence=confidence)


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
    """End-to-end workflow: run -> grade -> metrics -> detect -> report (+finetune).

    GGUFs are resolved from --models-dir using each model's registered filename.
    """
    from . import detectors as det
    ds = data.load_dataset(a.dataset)
    models = a.models.split(",") if a.models else list(MODELS)
    os.makedirs(os.path.join(a.work, "graded"), exist_ok=True)
    graded_paths = []

    _banner(f"STAGE 1-2/5  run + grade  ({len(models)} model(s))")
    for n, m in enumerate(models, 1):
        print(f"\n-- model {n}/{len(models)}: {m} --", flush=True)
        rp = os.path.join(a.work, "runs", f"{m}.jsonl")
        try:
            runner.run(ds, m, rp, backend=a.backend, dtype=a.dtype, load_4bit=a.load_4bit,
                       models_dir=a.models_dir, samples=a.samples,
                       want_logprobs=a.logprobs, n_gpu_layers=a.n_gpu_layers,
                       limit=a.limit, progress=True)
        except Exception as e:                           # missing GGUF, OOM, etc.
            print(f"[pipeline] SKIPPING {m}: {e}", flush=True)
            continue
        gp = os.path.join(a.work, "graded", f"{m}.jsonl")
        graded = [grading.grade_row(r) for r in track(data.read_jsonl(rp), desc=f"grade:{m}")]
        data.write_jsonl(gp, graded)
        graded_paths.append(gp)

    if not graded_paths:
        print("\n[pipeline] No models ran successfully. Check --models-dir and that "
              "the GGUF filenames match the registry (see `seam list-models`).")
        return 1
    rows = _load_rows(graded_paths)
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    # Mechanistic + behavioural detectors. With --probe we extract the residual
    # stream (misleading for the probe, clean for separability) so internal and
    # behavioural aspects are both interrogated; the result feeds the SEAM score.
    _banner("STAGE 3/5  detectors + mechanistic")
    detector_res, mech_by_model = {}, {}
    if a.probe:
        from . import activations as act, mechanistic as mech
        import numpy as np
        if a.backend != "hf":
            print("[pipeline] NOTE: --probe extracts activations from the full-precision "
                  "hf_id checkpoint, but --backend llamacpp generated behaviour from the "
                  "quantized GGUF. For a confound-free probe (behaviour and activations from "
                  "one model) add --backend hf.", flush=True)
    for n, (m, mrows) in enumerate(by_model.items(), 1):
        separability = None
        if a.probe and MODELS.get(m, {}).get("activations"):
            hf_id = MODELS[m].get("hf_id")
            mis = [r for r in mrows if r["condition"] == "misleading"]
            cln = [r for r in mrows if r["condition"] == "clean"]
            try:
                print(f"[probe {n}/{len(by_model)}] {m}: residual stream", flush=True)
                Xm, idm, lyr = act.extract_activations(mis, hf_id, dtype=a.dtype, load_4bit=a.load_4bit)
                act.save(os.path.join(a.work, "acts", f"{m}_misleading.npz"), Xm, idm, lyr)
                detector_res[m] = det.compare(mrows, activations=Xm, activation_ids=idm)
                best = detector_res[m].get("best_layer")
                if cln and best is not None:
                    Xc, idc, _ = act.extract_activations(cln, hf_id, dtype=a.dtype, load_4bit=a.load_4bit)
                    act.save(os.path.join(a.work, "acts", f"{m}_clean.npz"), Xc, idc, lyr)
                    stack = np.vstack([Xc[:, best, :], Xm[:, best, :]])
                    separability = mech.activation_silhouette(stack, [0] * len(Xc) + [1] * len(Xm))
            except Exception as e:
                print(f"[pipeline] probe skipped for {m}: {e}", flush=True)
        if m not in detector_res:
            print(f"[detect {n}/{len(by_model)}] {m}: {len(mrows)} rows (text only)", flush=True)
            detector_res[m] = det.compare(mrows)
        if a.probe:
            mech_by_model[m] = mech.mechanistic_summary(detector_res[m], separability)

    _banner("STAGE 4/5  metrics")
    summaries = []
    for n, (m, mrows) in enumerate(by_model.items(), 1):
        print(f"[metrics {n}/{len(by_model)}] {m}: {len(mrows)} rows"
              + (f", bootstrap x{a.bootstrap}" if a.bootstrap else ""), flush=True)
        rcs = None
        if a.rcs:
            from . import semantic
            rcs = semantic.rcs_scores(mrows)
        summaries.append(metrics.summarize(mrows, m, rcs=rcs, ci=a.bootstrap,
                                           mechanistic=mech_by_model.get(m) or None))

    _banner("STAGE 5/5  report")
    from . import confidence as conf
    conf_analysis = conf.analyze(by_model)
    with open(os.path.join(a.work, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    with open(os.path.join(a.work, "detectors.json"), "w", encoding="utf-8") as f:
        json.dump(detector_res, f, ensure_ascii=False, indent=2)
    with open(os.path.join(a.work, "confidence.json"), "w", encoding="utf-8") as f:
        json.dump(conf_analysis, f, ensure_ascii=False, indent=2)
    report.generate(summaries, os.path.join(a.work, "report"),
                    detectors=detector_res, confidence=conf_analysis)

    if a.finetune:
        _banner("OPTIONAL  RCS fine-tuning")
        from . import semantic
        try:
            res = semantic.finetune(rows, out_dir=os.path.join(a.work, "models", "rcs-ft"))
            print(json.dumps(res, indent=2))
        except Exception as e:
            print(f"[finetune] skipped: {e}")

    _banner("DONE")
    print(report.table_accuracy(summaries))
    print("\nDetectors:\n" + report.table_detectors(detector_res))
    print(f"\nAll outputs in: {os.path.abspath(a.work)}")


def build_parser():
    p = argparse.ArgumentParser(prog="seam", description="SEAM evaluation harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-models").set_defaults(func=_cmd_models)

    r = sub.add_parser("run", help="collect model responses via llama.cpp (GGUF)")
    r.add_argument("--model", required=True)
    r.add_argument("--dataset", default=data.DEFAULT_DATASET)
    r.add_argument("--out", required=True)
    r.add_argument("--backend", default="llamacpp", choices=["llamacpp", "hf"],
                   help="llamacpp = quantized GGUF (fast); hf = the hf_id checkpoint "
                        "(same model as activation extraction, for a consistent probe)")
    r.add_argument("--dtype", default="float16", help="HF backend dtype")
    r.add_argument("--load-4bit", action="store_true", help="HF backend 4-bit load")
    r.add_argument("--gguf", default=None, help="path to a local .gguf")
    r.add_argument("--models-dir", default=None,
                   help="dir of GGUFs named per the model registry")
    r.add_argument("--samples", type=int, default=1,
                   help=">1 enables self-consistency + a confidence fallback")
    r.add_argument("--logprobs", action="store_true",
                   help="load with logits_all=True to record token-logprob confidence")
    r.add_argument("--n-gpu-layers", type=int, default=-1,
                   help="layers to offload to GPU (-1=all, 0=CPU); needs a CUDA build")
    r.add_argument("--items", default=None,
                   help="run a pre-built work-item JSONL (e.g. counterfactual) instead of the dataset")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--categories", default=None, help="comma-separated filter")
    r.set_defaults(func=_cmd_run)

    cfp = sub.add_parser("counterfactual", help="build the counterfactual condition (4th column)")
    cfp.add_argument("infiles", nargs="+", help="graded JSONL (needs clean + misleading)")
    cfp.add_argument("--model", default=None, help="restrict to one model's rows")
    cfp.add_argument("--out", required=True, help="counterfactual work-item JSONL")
    cfp.set_defaults(func=_cmd_counterfactual)

    g = sub.add_parser("grade", help="grade responses, detect flips/shortcuts")
    g.add_argument("infile"); g.add_argument("--out", required=True)
    g.set_defaults(func=_cmd_grade)

    m = sub.add_parser("metrics", help="compute behavioural metrics + SEAM score")
    m.add_argument("infiles", nargs="+")
    m.add_argument("--out", default="metrics.json")
    m.add_argument("--rcs", action="store_true", help="compute RCS (needs sentence-transformers)")
    m.add_argument("--rcs-model", default=None)
    m.add_argument("--bootstrap", type=int, default=0,
                   help="bootstrap resamples for 95%% CIs over base-problem IDs (0=off)")
    m.set_defaults(func=lambda a: _cmd_metrics(_with_rcs_default(a)))

    dt = sub.add_parser("detect", help="shortcut detectors (lexical/TF-IDF/residual) + AUROC")
    dt.add_argument("infiles", nargs="+")
    dt.add_argument("--out", default="detectors.json")
    dt.add_argument("--activations", action="append",
                    help="residual probe input as 'model=path.npy' (repeatable)")
    dt.set_defaults(func=_cmd_detect)

    sub.add_parser("det-selftest", help="run detectors on synthetic data"
                   ).set_defaults(func=_cmd_det_selftest)

    ex = sub.add_parser("extract", help="extract residual-stream activations (HF, for the probe)")
    ex.add_argument("infiles", nargs="+", help="graded JSONL")
    ex.add_argument("--model", required=True, help="registry key (provides hf_id)")
    ex.add_argument("--hf-id", default=None, help="override HF checkpoint")
    ex.add_argument("--condition", default="misleading")
    ex.add_argument("--layers", default=None, help="comma-separated layer indices (default: all)")
    ex.add_argument("--dtype", default="float16")
    ex.add_argument("--load-4bit", action="store_true", help="4-bit load for big checkpoints")
    ex.add_argument("--out", required=True, help="output .npz")
    ex.set_defaults(func=_cmd_extract)

    cf = sub.add_parser("confidence", help="confidence -> shortcut-susceptibility analysis")
    cf.add_argument("infiles", nargs="+")
    cf.add_argument("--out", default="confidence.json")
    cf.add_argument("--fig", default=None, help="optional PNG path")
    cf.add_argument("--proxy", default="auto", choices=["auto", "logprob", "consistency", "hedging"])
    cf.add_argument("--bins", type=int, default=3)
    cf.set_defaults(func=_cmd_confidence)

    rp = sub.add_parser("report", help="render tables + figures")
    rp.add_argument("--metrics", default="metrics.json"); rp.add_argument("--out-dir", default="report")
    rp.add_argument("--detectors", default=None, help="detectors.json for Table 3 / Fig 4-5")
    rp.add_argument("--confidence", default=None, help="confidence.json for Fig 6")
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
    pl.add_argument("--bootstrap", type=int, default=0, help="bootstrap resamples for CIs")
    pl.add_argument("--samples", type=int, default=1, help="self-consistency samples")
    pl.add_argument("--logprobs", action="store_true", help="record token-logprob confidence")
    pl.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers (-1=all, 0=CPU)")
    pl.add_argument("--backend", default="llamacpp", choices=["llamacpp", "hf"],
                    help="hf runs the hf_id checkpoint (consistent with the probe)")
    pl.add_argument("--dtype", default="float16")
    pl.add_argument("--load-4bit", action="store_true")
    pl.add_argument("--rcs", action="store_true", help="compute RCS in metrics")
    pl.add_argument("--finetune", action="store_true", help="also fine-tune the RCS model")
    pl.add_argument("--probe", action="store_true",
                    help="extract activations and run the residual probe for every model "
                         "(HF checkpoints; needs transformers + GPU)")
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
