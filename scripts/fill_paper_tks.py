#!/usr/bin/env python3
"""Replace [TK: ...] markers in paper/main.tex with values computed from results.json.

Also rewrites the writer-drafted inline tables in main.tex so they \\input{}
the code-generated table files in paper/tables/. The substitutions are
idempotent — running this twice has no further effect.

Usage:
    uv run python scripts/make_paper_artifacts.py    # produces tables + figures
    uv run python scripts/fill_paper_tks.py          # patches main.tex
    cd paper && latexmk main.tex                     # re-compile
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
MAIN_TEX = REPO_ROOT / "paper" / "main.tex"

POS_VERDICTS = {"suspicious", "likely_hallucinated"}


def _confusion(preds: list[dict]) -> tuple[int, int, int, int]:
    tp = fn = fp = tn = 0
    for p in preds:
        exp_pos = p["expected_verdict"] == "likely_hallucinated"
        pred_pos = p["predicted_verdict"] in POS_VERDICTS
        if exp_pos and pred_pos:
            tp += 1
        elif exp_pos:
            fn += 1
        elif pred_pos:
            fp += 1
        else:
            tn += 1
    return tp, fn, fp, tn


def _safe_div(a: int, b: int) -> float:
    return a / b if b else float("nan")


def _fmt(x: float, digits: int = 3) -> str:
    return "n/a" if x != x else f"{x:.{digits}f}"


def _metrics(tp: int, fn: int, fp: int, tn: int) -> dict[str, float]:
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    f1 = 2 * p * r / (p + r) if p == p and r == r and (p + r) > 0 else float("nan")
    return {"precision": p, "recall": r, "fpr": fpr, "f1": f1}


def _build_substitutions() -> dict[str, str]:
    if not RESULTS_JSON.is_file():
        raise FileNotFoundError(f"missing: {RESULTS_JSON}")
    data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    preds = data["predictions"]

    with LABELS_CSV.open(encoding="utf-8") as fh:
        labels = {row["id"]: row for row in csv.DictReader(fh)}

    # Overall.
    tp, fn, fp, tn = _confusion(preds)
    m = _metrics(tp, fn, fp, tn)

    subs: dict[str, str] = {
        "[TK: precision]": _fmt(m["precision"]),
        "[TK: recall]": _fmt(m["recall"]),
        "[TK: fpr]": _fmt(m["fpr"]),
        "[TK: f1]": _fmt(m["f1"]),
        "[TK: TP]": f"{tp:,}",
        "[TK: FN]": f"{fn:,}",
        "[TK: FP]": f"{fp:,}",
        "[TK: TN]": f"{tn:,}",
    }

    # Walters-only slice.
    walters_preds = [
        p for p in preds if labels.get(p["id"], {}).get("labeler", "").startswith("walters")
    ]
    tp_w, fn_w, fp_w, tn_w = _confusion(walters_preds)
    m_w = _metrics(tp_w, fn_w, fp_w, tn_w)
    subs.update(
        {
            "[TK: prec_w]": _fmt(m_w["precision"]),
            "[TK: rec_w]": _fmt(m_w["recall"]),
            "[TK: fpr_w]": _fmt(m_w["fpr"]),
            "[TK: f1_w]": _fmt(m_w["f1"]),
        }
    )

    # Fixture (augmentation) slice — all real, so FPR is the only meaningful metric.
    aug_preds = [
        p for p in preds if not labels.get(p["id"], {}).get("labeler", "").startswith("walters")
    ]
    _tp_a, _fn_a, fp_a, tn_a = _confusion(aug_preds)
    subs["[TK: fpr_aug]"] = _fmt(_safe_div(fp_a, fp_a + tn_a))

    # Per-publication-type metrics (Walters only).
    def _pubtype(row: dict) -> str:
        m_ = re.search(r"type=([ABCW])", row.get("notes", ""))
        return m_.group(1) if m_ else "NA"

    by_type: dict[str, list[dict]] = {}
    for p in walters_preds:
        t = _pubtype(labels.get(p["id"], {}))
        by_type.setdefault(t, []).append(p)

    for code, key_prefix in [("A", "art"), ("B", "bk")]:
        preds_t = by_type.get(code, [])
        n = len(preds_t)
        tp_t, fn_t, fp_t, tn_t = _confusion(preds_t)
        m_t = _metrics(tp_t, fn_t, fp_t, tn_t)
        subs[f"[TK: n_{key_prefix}]"] = f"{n:,}"
        subs[f"[TK: prec_{key_prefix}]"] = _fmt(m_t["precision"])
        subs[f"[TK: rec_{key_prefix}]"] = _fmt(m_t["recall"])
        subs[f"[TK: fpr_{key_prefix}]"] = _fmt(m_t["fpr"])
        subs[f"[TK: f1_{key_prefix}]"] = _fmt(m_t["f1"])

    return subs


def main() -> int:
    try:
        subs = _build_substitutions()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    text = MAIN_TEX.read_text(encoding="utf-8")
    n_replaced = 0
    for marker, value in subs.items():
        if marker in text:
            text = text.replace(marker, value)
            n_replaced += 1

    # Any TKs we didn't cover (e.g. layer-level numbers we don't yet expose)?
    leftover = re.findall(r"\[TK:[^\]]+\]", text)
    leftover_unique = sorted(set(leftover))

    MAIN_TEX.write_text(text, encoding="utf-8")
    print(f"replaced {n_replaced} TK markers", file=sys.stderr)
    if leftover_unique:
        print("  leftover (no value provided):", file=sys.stderr)
        for tk in leftover_unique:
            print(f"    {tk}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
