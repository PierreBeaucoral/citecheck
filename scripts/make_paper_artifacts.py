#!/usr/bin/env python3
"""Generate paper/tables/*.tex and paper/figures/*.pdf from data/eval/results.json.

Every number that appears in the paper comes through here — never hand-typed
in main.tex. After run_eval.py finishes, invoke this script and then re-render
the LaTeX.

Outputs (all idempotent overwrites):

    paper/tables/headline_metrics.tex      precision, recall, FPR, F1
    paper/tables/confusion_matrix.tex      TP / FN / FP / TN
    paper/tables/per_source_metrics.tex    Walters vs fixture
    paper/tables/per_pubtype_metrics.tex   articles / books / chapters / websites
    paper/figures/predicted_by_class.pdf   stacked bars: predicted verdict by true class
    paper/figures/per_layer_flag_rate.pdf  bar chart of each layer's flag rate

Usage:
    uv run python scripts/make_paper_artifacts.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = REPO_ROOT / "data" / "eval" / "results.json"
LABELS_CSV = REPO_ROOT / "data" / "eval" / "labels.csv"
TABLES_DIR = REPO_ROOT / "paper" / "tables"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"

POS_VERDICTS = {"suspicious", "likely_hallucinated"}
NEG_VERDICTS = {"real_high_confidence", "real_low_confidence"}


def _safe_div(num: int, den: int) -> float:
    return num / den if den else float("nan")


def _fmt(x: float, digits: int = 3) -> str:
    if x != x:  # NaN
        return "n/a"
    return f"{x:.{digits}f}"


def _metrics_from_confusion(tp: int, fn: int, fp: int, tn: int) -> dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall) > 0
        else float("nan")
    )
    return {"precision": precision, "recall": recall, "fpr": fpr, "f1": f1}


def _extract_pubtype(notes: str) -> str:
    """Walters' notes column has `type=A|B|C|W`. Return that letter or 'NA'."""
    m = re.search(r"type=([ABCW])", notes or "")
    return m.group(1) if m else "NA"


def _source_from_labeler(labeler: str) -> str:
    if labeler.startswith("walters"):
        return "walters"
    if "callaway" in labeler or "borusyak" in labeler:
        return "fixture"
    return "other"


def _load() -> tuple[list[dict], dict[str, dict]]:
    if not RESULTS_JSON.is_file():
        raise FileNotFoundError(
            f"missing: {RESULTS_JSON}.\nRun `uv run python scripts/run_eval.py` first."
        )
    data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    predictions = data["predictions"]
    with LABELS_CSV.open(encoding="utf-8") as fh:
        labels_by_id = {row["id"]: row for row in csv.DictReader(fh)}
    return predictions, labels_by_id


def _confusion(predictions: list[dict]) -> tuple[int, int, int, int]:
    tp = fn = fp = tn = 0
    for p in predictions:
        expected_pos = p["expected_verdict"] == "likely_hallucinated"
        predicted_pos = p["predicted_verdict"] in POS_VERDICTS
        if expected_pos and predicted_pos:
            tp += 1
        elif expected_pos and not predicted_pos:
            fn += 1
        elif not expected_pos and predicted_pos:
            fp += 1
        else:
            tn += 1
    return tp, fn, fp, tn


def _emit_headline_table(predictions: list[dict]) -> None:
    tp, fn, fp, tn = _confusion(predictions)
    m = _metrics_from_confusion(tp, fn, fp, tn)
    n = len(predictions)
    body = (
        "\\begin{tabular}{lr}\n\\toprule\nMetric & Value \\\\\n\\midrule\n"
        f"References evaluated & {n:,} \\\\\n"
        f"Precision & {_fmt(m['precision'])} \\\\\n"
        f"Recall & {_fmt(m['recall'])} \\\\\n"
        f"False-positive rate & {_fmt(m['fpr'])} \\\\\n"
        f"F1 score & {_fmt(m['f1'])} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    (TABLES_DIR / "headline_metrics.tex").write_text(body, encoding="utf-8")


def _emit_confusion_table(predictions: list[dict]) -> None:
    tp, fn, fp, tn = _confusion(predictions)
    body = (
        "\\begin{tabular}{lrr}\n\\toprule\n"
        " & Predicted hallucinated & Predicted real \\\\\n"
        "\\midrule\n"
        f"Truly hallucinated & {tp:,} & {fn:,} \\\\\n"
        f"Truly real         & {fp:,} & {tn:,} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    (TABLES_DIR / "confusion_matrix.tex").write_text(body, encoding="utf-8")


def _emit_per_source_table(predictions: list[dict], labels_by_id: dict[str, dict]) -> None:
    by_source: dict[str, list[dict]] = {}
    for p in predictions:
        row = labels_by_id.get(p["id"], {})
        src = _source_from_labeler(row.get("labeler", ""))
        by_source.setdefault(src, []).append(p)

    rows = []
    for src, preds in sorted(by_source.items()):
        tp, fn, fp, tn = _confusion(preds)
        m = _metrics_from_confusion(tp, fn, fp, tn)
        n_fab = tp + fn
        n_real = fp + tn
        rows.append(
            f"{src} & {len(preds):,} & {n_fab:,} & {n_real:,} & "
            f"{_fmt(m['precision'])} & {_fmt(m['recall'])} & "
            f"{_fmt(m['fpr'])} & {_fmt(m['f1'])} \\\\"
        )
    body = (
        "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
        "Source & N & Fab & Real & Precision & Recall & FPR & F1 \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (TABLES_DIR / "per_source_metrics.tex").write_text(body, encoding="utf-8")


def _emit_per_pubtype_table(predictions: list[dict], labels_by_id: dict[str, dict]) -> None:
    """Restricted to Walters rows (only source with type metadata)."""
    by_type: dict[str, list[dict]] = {}
    for p in predictions:
        row = labels_by_id.get(p["id"], {})
        if not row.get("labeler", "").startswith("walters"):
            continue
        t = _extract_pubtype(row.get("notes", ""))
        by_type.setdefault(t, []).append(p)

    type_names = {"A": "Articles", "B": "Books", "C": "Chapters", "W": "Websites", "NA": "Unknown"}
    rows = []
    for t in ["A", "B", "C", "W", "NA"]:
        preds = by_type.get(t, [])
        if not preds:
            continue
        tp, fn, fp, tn = _confusion(preds)
        m = _metrics_from_confusion(tp, fn, fp, tn)
        n_fab = tp + fn
        n_real = fp + tn
        rows.append(
            f"{type_names[t]} & {len(preds):,} & {n_fab:,} & {n_real:,} & "
            f"{_fmt(m['precision'])} & {_fmt(m['recall'])} & "
            f"{_fmt(m['fpr'])} & {_fmt(m['f1'])} \\\\"
        )
    body = (
        "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
        "Type & N & Fab & Real & Precision & Recall & FPR & F1 \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (TABLES_DIR / "per_pubtype_metrics.tex").write_text(body, encoding="utf-8")


def _emit_combined_headline(predictions: list[dict]) -> None:
    """Confusion matrix + classification metrics as one tabular, matching main.tex shape."""
    tp, fn, fp, tn = _confusion(predictions)
    m = _metrics_from_confusion(tp, fn, fp, tn)
    body = (
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "                            & Predicted real & Predicted fabricated \\\\\n"
        "\\midrule\n"
        f"True real                   & {tn:,}        & {fp:,}              \\\\\n"
        f"True fabricated             & {fn:,}        & {tp:,}              \\\\\n"
        "\\midrule\n"
        "\\multicolumn{3}{l}{\\textit{Classification metrics}} \\\\\n"
        "\\midrule\n"
        f"Precision                   & \\multicolumn{{2}}{{c}}{{{_fmt(m['precision'])}}} \\\\\n"
        f"Recall                      & \\multicolumn{{2}}{{c}}{{{_fmt(m['recall'])}}} \\\\\n"
        f"False-positive rate         & \\multicolumn{{2}}{{c}}{{{_fmt(m['fpr'])}}} \\\\\n"
        f"$F_1$                       & \\multicolumn{{2}}{{c}}{{{_fmt(m['f1'])}}} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    (TABLES_DIR / "eval_headline.tex").write_text(body, encoding="utf-8")


def _emit_combined_bysource(predictions: list[dict], labels_by_id: dict[str, dict]) -> None:
    """Per-source rows only. Publication type is broken out separately."""
    by_source: dict[str, list[dict]] = {}
    for p in predictions:
        src = _source_from_labeler(labels_by_id.get(p["id"], {}).get("labeler", ""))
        by_source.setdefault(src, []).append(p)

    rows = []
    walters_preds = by_source.get("walters", [])
    fixture_preds = by_source.get("fixture", [])
    if walters_preds:
        tp, fn, fp, tn = _confusion(walters_preds)
        m = _metrics_from_confusion(tp, fn, fp, tn)
        rows.append(
            f"\\citet{{walters_wilder_2023}}              & {len(walters_preds):,} & "
            f"{_fmt(m['precision'])} & {_fmt(m['recall'])} & {_fmt(m['fpr'])} & {_fmt(m['f1'])} \\\\"
        )
    if fixture_preds:
        _tp, _fn, fp, tn = _confusion(fixture_preds)
        fpr = _safe_div(fp, fp + tn)
        rows.append(
            f"arXiv econ augmentation                      & {len(fixture_preds):,} & "
            f"---            & ---            & {_fmt(fpr)} & ---            \\\\"
        )

    body = (
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        "Source                                       & N    & Precision      & "
        "Recall         & FPR             & $F_1$          \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    (TABLES_DIR / "eval_bysource.tex").write_text(body, encoding="utf-8")


def _emit_bypubtype_clean(predictions: list[dict], labels_by_id: dict[str, dict]) -> None:
    """Standalone per-publication-type table for the Walters slice.

    Each Walters citation carries a publication-type tag in the supplementary
    appendix (A=article, B=book, C=chapter, W=website). The eval table reports
    precision/recall/FPR/F1 per type so the reader can see where the detector
    works and where it does not. Articles are the design's primary target;
    books and grey literature are deliberately harder cases for an
    index-based detector.
    """
    walters_preds = [
        p
        for p in predictions
        if labels_by_id.get(p["id"], {}).get("labeler", "").startswith("walters")
    ]

    type_names = {
        "A": "Journal articles",
        "B": "Books",
        "C": "Chapters",
        "W": "Websites and grey lit",
    }
    by_type: dict[str, list[dict]] = {}
    for p in walters_preds:
        row = labels_by_id.get(p["id"], {})
        m_ = re.search(r"type=([ABCW])", row.get("notes", ""))
        if m_:
            by_type.setdefault(m_.group(1), []).append(p)

    rows: list[str] = []
    for code in ["A", "B", "C", "W"]:
        preds_t = by_type.get(code, [])
        if not preds_t:
            continue
        tp, fn, fp, tn = _confusion(preds_t)
        m = _metrics_from_confusion(tp, fn, fp, tn)
        n_fab = tp + fn
        n_real = fp + tn
        rows.append(
            f"{type_names[code]:<28}            & {len(preds_t):,} & {n_fab:,} & {n_real:,} & "
            f"{_fmt(m['precision'])} & {_fmt(m['recall'])} & {_fmt(m['fpr'])} & {_fmt(m['f1'])} \\\\"
        )

    body = (
        "\\begin{tabular}{lrrrcccc}\n\\toprule\n"
        "Publication type                          & N & Fab & Real & "
        "Precision & Recall & FPR & $F_1$ \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (TABLES_DIR / "eval_bypubtype.tex").write_text(body, encoding="utf-8")


def _emit_per_layer_firing_rate(predictions: list[dict]) -> None:
    """Per-layer flag rate broken out by true class.

    Requires results.json to carry `layer_flags` per row (added in run_eval.py).
    Falls back to an explanatory note when the field is absent (older eval).
    """
    if not predictions or "layer_flags" not in predictions[0]:
        body = (
            "\\begin{tabular}{lcc}\n\\toprule\n"
            "Layer & Real refs & Fabricated refs \\\\\n"
            "\\midrule\n"
            "\\multicolumn{3}{l}{\\textit{Per-layer firing rates not stored in this eval pass}} \\\\\n"
            "\\multicolumn{3}{l}{\\textit{(see scripts/run\\_eval.py layer\\_flags field)}} \\\\\n"
            "\\bottomrule\n\\end{tabular}\n"
        )
        (TABLES_DIR / "eval_per_layer.tex").write_text(body, encoding="utf-8")
        return

    real = [p for p in predictions if p["expected_verdict"] != "likely_hallucinated"]
    fab = [p for p in predictions if p["expected_verdict"] == "likely_hallucinated"]
    layers = ["L1_doi_integrity", "L2_cross_db", "L3_authors", "L4_venue", "L5_author_title"]
    nice_names = {
        "L1_doi_integrity": "L1 DOI integrity",
        "L2_cross_db": "L2 Cross-DB existence",
        "L3_authors": "L3 Author plausibility",
        "L4_venue": "L4 Venue plausibility",
        "L5_author_title": "L5 Author--title coherence",
    }

    rows = []
    for layer in layers:
        real_flag_rate = (
            sum(1 for p in real if p.get("layer_flags", {}).get(layer)) / len(real)
            if real
            else 0.0
        )
        fab_flag_rate = (
            sum(1 for p in fab if p.get("layer_flags", {}).get(layer)) / len(fab)
            if fab
            else 0.0
        )
        rows.append(
            f"{nice_names[layer]:<32} & {_fmt(real_flag_rate)} & {_fmt(fab_flag_rate)} \\\\"
        )

    body = (
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "Layer                          & Flag rate (real) & Flag rate (fab) \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (TABLES_DIR / "eval_per_layer.tex").write_text(body, encoding="utf-8")


def _emit_bypubtype_barchart(predictions: list[dict], labels_by_id: dict[str, dict]) -> None:
    """Grouped bar chart: precision / recall / FPR per publication type."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    walters = [
        p
        for p in predictions
        if labels_by_id.get(p["id"], {}).get("labeler", "").startswith("walters")
    ]
    by_type: dict[str, list[dict]] = {}
    for p in walters:
        row = labels_by_id.get(p["id"], {})
        m_ = re.search(r"type=([ABCW])", row.get("notes", ""))
        if m_:
            by_type.setdefault(m_.group(1), []).append(p)

    type_order = ["A", "B", "C", "W"]
    type_labels = ["Articles", "Books", "Chapters", "Websites"]
    precision: list[float] = []
    recall: list[float] = []
    fpr: list[float] = []
    for code in type_order:
        preds_t = by_type.get(code, [])
        if not preds_t:
            precision.append(0.0)
            recall.append(0.0)
            fpr.append(0.0)
            continue
        tp, fn, fp, tn = _confusion(preds_t)
        m = _metrics_from_confusion(tp, fn, fp, tn)
        precision.append(0.0 if m["precision"] != m["precision"] else m["precision"])
        recall.append(0.0 if m["recall"] != m["recall"] else m["recall"])
        fpr.append(0.0 if m["fpr"] != m["fpr"] else m["fpr"])

    x = list(range(len(type_order)))
    width = 0.27
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar([xi - width for xi in x], precision, width, label="Precision", color="#4c8bd9")
    ax.bar(x, recall, width, label="Recall", color="#67ad5b")
    ax.bar([xi + width for xi in x], fpr, width, label="FPR", color="#d9534f")
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels)
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.5, color="#999", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "metrics_by_pubtype.pdf")
    plt.close(fig)


def _emit_predicted_by_class_fig(predictions: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verdicts = ["real_high_confidence", "real_low_confidence", "suspicious", "likely_hallucinated"]
    # Count by true class by predicted verdict.
    counts = {"truly_real": [0] * 4, "truly_hallucinated": [0] * 4}
    for p in predictions:
        cls = (
            "truly_hallucinated" if p["expected_verdict"] == "likely_hallucinated" else "truly_real"
        )
        if p["predicted_verdict"] in verdicts:
            counts[cls][verdicts.index(p["predicted_verdict"])] += 1

    x = list(range(len(verdicts)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(
        [xi - width / 2 for xi in x],
        counts["truly_real"],
        width,
        label="Truly real",
        color="#4c8bd9",
    )
    ax.bar(
        [xi + width / 2 for xi in x],
        counts["truly_hallucinated"],
        width,
        label="Truly hallucinated",
        color="#d9534f",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["real_high", "real_low", "suspicious", "hallucinated"], rotation=15, ha="right"
    )
    ax.set_ylabel("Number of references")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "predicted_by_class.pdf")
    plt.close(fig)


def _emit_per_layer_flag_rate_fig(predictions: list[dict], labels_by_id: dict[str, dict]) -> None:
    """Show how often each layer fires on real vs hallucinated, per source."""
    # We don't ship layer-level signals in results.json yet — rely on the
    # red_flag_count we have. Display the distribution of red-flag counts.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = {"truly_real": [0, 0, 0, 0, 0], "truly_hallucinated": [0, 0, 0, 0, 0]}
    for p in predictions:
        cls = (
            "truly_hallucinated" if p["expected_verdict"] == "likely_hallucinated" else "truly_real"
        )
        k = min(p.get("red_flag_count") or 0, 4)
        counts[cls][k] += 1

    x = [0, 1, 2, 3, 4]
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(
        [xi - width / 2 for xi in x],
        counts["truly_real"],
        width,
        label="Truly real",
        color="#4c8bd9",
    )
    ax.bar(
        [xi + width / 2 for xi in x],
        counts["truly_hallucinated"],
        width,
        label="Truly hallucinated",
        color="#d9534f",
    )
    ax.set_xticks(x)
    ax.set_xlabel("Number of red flags (0--4)")
    ax.set_ylabel("Number of references")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "red_flag_distribution.pdf")
    plt.close(fig)


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        predictions, labels_by_id = _load()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    _emit_headline_table(predictions)
    _emit_confusion_table(predictions)
    _emit_per_source_table(predictions, labels_by_id)
    _emit_per_pubtype_table(predictions, labels_by_id)
    _emit_predicted_by_class_fig(predictions)
    _emit_per_layer_flag_rate_fig(predictions, labels_by_id)
    # Combined tables that main.tex \input{}s directly:
    _emit_combined_headline(predictions)
    _emit_combined_bysource(predictions, labels_by_id)
    _emit_bypubtype_clean(predictions, labels_by_id)
    _emit_per_layer_firing_rate(predictions)
    _emit_bypubtype_barchart(predictions, labels_by_id)

    tp, fn, fp, tn = _confusion(predictions)
    m = _metrics_from_confusion(tp, fn, fp, tn)
    print(
        f"OK. n={len(predictions)} precision={_fmt(m['precision'])} "
        f"recall={_fmt(m['recall'])} fpr={_fmt(m['fpr'])} f1={_fmt(m['f1'])}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
