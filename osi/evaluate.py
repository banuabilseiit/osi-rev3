"""Оценка проверок на синтетических сериях с известными нарушениями."""
import json
import random
import tempfile
from collections import defaultdict
from pathlib import Path

from .checks import check_reports
from .synth import INJECTIONS, make_series, render_pdf

ALLOWED_SHIFT = {"balance_savings": 1, "balance_current": 1, "jump": 1}


def _month_ok(kind, truth_month, finding_month, labels):
    shift = ALLOWED_SHIFT.get(kind, 0)
    allowed = {labels[truth_month]}
    if shift and truth_month + shift < len(labels):
        allowed.add(labels[truth_month + shift])
    return finding_month in allowed


def score_series(series, findings):
    """-> {kind: (tp, fp, fn)} для одной серии."""
    labels = [r.label for r in series.reports]
    out = {}
    for kind in INJECTIONS:
        truths = [t for t in series.truth if t["kind"] == kind]
        found = [f for f in findings if f.kind == kind]
        hit = [f for f in found if any(
            _month_ok(kind, t["month"], f.report, labels) for t in truths)]
        tp = min(len(truths), max(1, len(hit))) if hit else 0
        fp = len(found) - len(hit)
        fn = len(truths) - tp
        out[kind] = (tp, fp, fn)
    return out


def evaluate(n_series=200, seed=0, via_pdf=False, months=3):
    rng = random.Random(seed)
    totals = defaultdict(lambda: [0, 0, 0])
    tmp = Path(tempfile.mkdtemp())
    for i in range(n_series):
        kinds = [k for k in INJECTIONS if rng.random() < 0.3]
        series = make_series(seed * 100_000 + i, months, kinds)
        reports = series.reports
        if via_pdf:
            from .ingest import load_pdf
            from .report import build_report
            parsed = []
            for j, rep in enumerate(reports):
                path = render_pdf(rep, tmp / f"{i}_{j}.pdf")
                parsed.append(build_report(load_pdf(path), path.name))
            reports = parsed
        findings = check_reports(reports)
        for kind, (tp, fp, fn) in score_series(series, findings).items():
            totals[kind][0] += tp
            totals[kind][1] += fp
            totals[kind][2] += fn
    table = {}
    for kind, (tp, fp, fn) in totals.items():
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) \
            if precision + recall else 0.0
        table[kind] = {"tp": tp, "fp": fp, "fn": fn,
                       "precision": round(precision, 3),
                       "recall": round(recall, 3), "f1": round(f1, 3)}
    return table


if __name__ == "__main__":
    import sys
    pdf = "--pdf" in sys.argv
    result = evaluate(30 if pdf else 300, seed=1, via_pdf=pdf)
    name = "metrics_checks_pdf.json" if pdf else "metrics_checks.json"
    Path("docs").mkdir(exist_ok=True)
    Path("docs", name).write_text(json.dumps(result, indent=2,
                                             ensure_ascii=False))
    for kind, m in result.items():
        print(f"{kind:20s} P={m['precision']:.2f} R={m['recall']:.2f} "
              f"F1={m['f1']:.2f}  tp={m['tp']} fp={m['fp']} fn={m['fn']}")
