#!/usr/bin/env python3
"""Evaluate Phase 5 (claim verification) against the hand-curated triple set.

The eval set lives at data/eval/claim_verification_labels.json — 100 (paragraph,
citation, source) items spread over 11 verified open-access sources, with a
clean 50/50 correct/incorrect split.  Each false item carries an error_type
(distortion, fabricated_specifics, overstatement, cherry_picking,
citation_claim_mismatch) and a severity (subtle / moderate / blatant), so we
report stratified numbers in addition to the headline binary.

Pipeline per item:

  1. Resolve the source's PDF or full text:
       - DOI -> Unpaywall PDF -> pypdf text
       - PMC -> NCBI E-utilities JATS XML -> stripped text
     The result is cached on disk; subsequent runs are fast.
  2. Pass the paragraph + text into citecheck's verify_claim().
  3. Map predicted ClaimStatus to the binary "looks correct" label:
       SUPPORTED       -> "correct"
       PARTIAL         -> "correct" (gives the tool credit for partial support
                          rather than penalizing it for hedging; the dataset's
                          ground-truth label is "correct"/"incorrect", not
                          tri-state)
       NOT_SUPPORTED   -> "incorrect"
       UNVERIFIABLE/UNCHECKED/ERROR -> excluded from headline (counted
                          separately).

  4. Binary scoring (positive = predicted "incorrect"): precision, recall,
     FPR, F1.  Plus stratified breakdowns by severity and error_type.

By default the verifier runs against `gemma4:31b-cloud` on Ollama Cloud's free
tier.  Override with `--model` or `OLLAMA_MODEL=...`.

Usage:
    uv run python scripts/run_claim_eval.py
    uv run python scripts/run_claim_eval.py --limit 5      # dry-run
    uv run python scripts/run_claim_eval.py --model qwen2.5:7b-instruct
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# .env is the convention everywhere else in citecheck (Unpaywall, Crossref,
# OpenAlex polite pool); the CLI reads it via python-dotenv at import time and
# the eval should match.  Loading here keeps the scripts directory consistent
# with the rest of the codebase without needing CITECHECK_CONTACT_EMAIL set
# in the user's shell.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from citecheck.checks.cache import CacheStore  # noqa: E402
from citecheck.checks.claims import (  # noqa: E402
    _call_cerebras,
    _call_hf_chat,
    _call_ollama,
    verify_claim,
)
from citecheck.models import (  # noqa: E402
    ClaimStatus,
    RawReference,
    Reference,
    ResolutionStatus,
)
from citecheck.resolution.unpaywall import fetch_text_any  # noqa: E402

LABELS_JSON = REPO_ROOT / "data" / "eval" / "claim_verification_labels.json"
RESULTS_JSON = REPO_ROOT / "data" / "eval" / "claim_results.json"
RESULTS_MD = REPO_ROOT / "data" / "eval" / "claim_results.md"

log = logging.getLogger("claim_eval")


def _classify_id(raw: str) -> tuple[str | None, str | None]:
    """Split an identifier string into (doi, pmc_id) — at most one is non-None."""
    s = (raw or "").strip()
    if not s:
        return None, None
    if s.upper().startswith("PMC"):
        return None, s
    if re.match(r"^10\.\d{4,9}/", s):
        return s, None
    # Some dataset entries put a PMC ID into the doi field without the PMC prefix.
    if re.match(r"^\d{6,}$", s):
        return None, "PMC" + s
    return None, None


def _build_reference(doi: str | None) -> Reference:
    """Wrap a DOI as a Reference so verify_claim can record provenance even
    when text is supplied directly."""
    return Reference(
        raw=RawReference(raw_text="eval", title="eval"),
        status=ResolutionStatus.RESOLVED if doi else ResolutionStatus.UNRESOLVED,
        resolved_doi=doi,
    )


def _gather_source_texts(
    sources: list[dict], cache: CacheStore
) -> dict[str, tuple[str | None, str | None]]:
    """Resolve every source once up-front so the eval loop never re-fetches.

    Returns a dict keyed by source_id; values are (text, provenance) where
    provenance is one of: "unpaywall_pdf", "pmc_xml", or None when the
    fetcher returned nothing.
    """
    out: dict[str, tuple[str | None, str | None]] = {}
    for src in sources:
        sid = src["source_id"]
        doi, pmc_id = _classify_id(src.get("doi"))
        log.info("fetching source %s (doi=%s pmc=%s)", sid, doi, pmc_id)
        text, provenance = fetch_text_any(doi=doi, pmc_id=pmc_id, cache=cache)
        if text is None:
            log.warning("source %s: no text resolved", sid)
        else:
            log.info("source %s: %d chars via %s", sid, len(text), provenance)
        out[sid] = (text, provenance)
    return out


def _predicted_label(status: ClaimStatus) -> str | None:
    """Map ClaimStatus to the binary dataset label.

    Returns None for non-headline statuses (UNVERIFIABLE / UNCHECKED / ERROR)
    so they are excluded from precision/recall.
    """
    if status == ClaimStatus.SUPPORTED:
        return "correct"
    if status == ClaimStatus.PARTIAL:
        # Give partial support credit on the "correct" side rather than
        # double-counting it as a flag.  The dataset's ground truth is binary
        # so we have to collapse the tri-state somewhere.
        return "correct"
    if status == ClaimStatus.NOT_SUPPORTED:
        return "incorrect"
    return None


def _safe_div(a: int, b: int) -> float:
    return a / b if b else float("nan")


def _format_pct(v: float) -> str:
    return f"{v:.3f}" if v == v else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=LABELS_JSON,
        help="Path to the labels JSON (default: data/eval/claim_verification_labels.json).",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=RESULTS_MD,
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_JSON,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=("ollama", "hf", "cerebras"),
        default=os.environ.get("CITECHECK_LLM_PROVIDER", "ollama"),
        help="LLM provider: 'ollama' (default; honors --model), 'hf' "
        "(HF Inference Router; honors --hf-model), or 'cerebras' "
        "(Cerebras Cloud; honors --cerebras-model).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud"),
        help="Ollama model name. Default: gemma4:31b-cloud (free cloud tier).",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=os.environ.get("CITECHECK_HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        help="HuggingFace model id (used when --provider hf). Default: Qwen/Qwen2.5-7B-Instruct.",
    )
    parser.add_argument(
        "--cerebras-model",
        type=str,
        default=os.environ.get("CITECHECK_CEREBRAS_MODEL", "qwen-3-235b-a22b-instruct-2507"),
        help="Cerebras model id (used when --provider cerebras). "
        "Default: qwen-3-235b-a22b-instruct-2507 (free tier).",
    )
    parser.add_argument(
        "--sleep-s",
        type=float,
        default=0.0,
        help="Sleep this many seconds between LLM calls.  Recommended values "
        "by provider: Cerebras free tier ~1.5 (30 req/min cap; verified live "
        "the 2.5s setting kept us at ~4 req/min, safely under cap but slower "
        "than necessary); HF Inference free ~0 (no per-minute cap but tight "
        "monthly token quota); Ollama local 0 (no rate limit).  Has no effect "
        "on the deployed app (one request per claim).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, run only the first N items (for dry-runs).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Echo per-item progress to stderr.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if not args.labels.is_file():
        print(f"missing: {args.labels}", file=sys.stderr)
        return 2

    data = json.loads(args.labels.read_text(encoding="utf-8"))
    sources = data["sources"]
    items = data["items"]
    if args.limit > 0:
        items = items[: args.limit]

    chosen_provider = args.provider
    if chosen_provider == "hf":
        chosen_model = args.hf_model
    elif chosen_provider == "cerebras":
        chosen_model = args.cerebras_model
    else:
        chosen_model = args.model
    print(f"loaded {len(items)} items over {len(sources)} sources", file=sys.stderr)
    print(f"provider: {chosen_provider}, model: {chosen_model}", file=sys.stderr)

    cache = CacheStore()
    try:
        # 1) Pre-resolve every source's text so the eval loop is purely LLM-bound.
        source_texts = _gather_source_texts(sources, cache)
        resolved = sum(1 for t, _ in source_texts.values() if t is not None)
        print(
            f"resolved text for {resolved}/{len(sources)} sources",
            file=sys.stderr,
        )
        if resolved == 0:
            print(
                "FATAL: no sources resolved; check network + email env var.",
                file=sys.stderr,
            )
            return 3

        # 2) Wrap the chosen LLM provider in a closure that fixes the model
        #    at the chosen value, matching verify_claim's expected signature.
        if chosen_provider == "hf":

            def ollama_call(prompt: str) -> str:
                return _call_hf_chat(prompt, model=chosen_model)
        elif chosen_provider == "cerebras":

            def ollama_call(prompt: str) -> str:
                return _call_cerebras(prompt, model=chosen_model)
        else:

            def ollama_call(prompt: str) -> str:
                return _call_ollama(prompt, model=chosen_model)

        # 3) The embedder is expensive to load — share one instance across items.
        from citecheck.checks.claims import _load_embedder

        embedder = _load_embedder()

        predictions: list[dict[str, Any]] = []
        start = time.time()
        skipped_no_text = 0

        for i, item in enumerate(items, 1):
            sid = item["source_id"]
            text, provenance = source_texts.get(sid, (None, None))
            if text is None:
                # Source failed to fetch; record but exclude from scoring.
                predictions.append(
                    {
                        "id": item["id"],
                        "source_id": sid,
                        "expected": item["correctness"],
                        "predicted_status": "unverifiable",
                        "predicted_label": None,
                        "error_type": item.get("error_type"),
                        "severity": item.get("severity"),
                        "provenance": provenance,
                        "elapsed_s": 0.0,
                        "match": None,
                        "notes": "source text unavailable",
                    }
                )
                skipped_no_text += 1
                continue

            ref = _build_reference(
                _classify_id(next((s["doi"] for s in sources if s["source_id"] == sid), ""))[0]
            )

            # Inter-request throttle for rate-limited free providers.
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)
            t0 = time.time()
            try:
                check = verify_claim(
                    ref,
                    item["paragraph"],
                    cache=cache,
                    embedder=embedder,
                    ollama_call=ollama_call,
                    text=text,
                )
            except Exception as exc:
                # Don't let one bad LLM response kill the run.
                check = None
                err = repr(exc)
                log.warning("verify_claim raised for item %s: %s", item["id"], err)
            else:
                err = None
            elapsed = time.time() - t0

            if check is None:
                predicted_status = "error"
                predicted_label = None
                quote = None
                confidence = None
                reasoning = err
            else:
                predicted_status = check.status.value
                predicted_label = _predicted_label(check.status)
                quote = check.quote
                confidence = check.confidence
                reasoning = check.reasoning

            match: bool | None = (
                None if predicted_label is None else predicted_label == item["correctness"]
            )

            predictions.append(
                {
                    "id": item["id"],
                    "source_id": sid,
                    "expected": item["correctness"],
                    "predicted_status": predicted_status,
                    "predicted_label": predicted_label,
                    "error_type": item.get("error_type"),
                    "severity": item.get("severity"),
                    "field": item.get("field"),
                    "provenance": provenance,
                    "elapsed_s": round(elapsed, 2),
                    "match": match,
                    "quote": quote,
                    "confidence": confidence,
                    "reasoning": reasoning,
                }
            )

            if i % 5 == 0 or i == len(items):
                done_pct = i / len(items)
                wall = time.time() - start
                rate = i / wall if wall > 0 else 0
                eta = (len(items) - i) / rate if rate > 0 else 0
                print(
                    f"  {i:3d}/{len(items)} ({done_pct:.0%}) "
                    f"elapsed {wall:.0f}s rate {rate:.2f}/s eta {eta:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        cache.close()

    # ---- Scoring -----------------------------------------------------------

    # Positive class = predicted "incorrect" (flagged as misrepresentation).
    scored = [p for p in predictions if p["predicted_label"] is not None]
    tp = sum(
        1 for p in scored if p["predicted_label"] == "incorrect" and p["expected"] == "incorrect"
    )
    fp = sum(
        1 for p in scored if p["predicted_label"] == "incorrect" and p["expected"] == "correct"
    )
    fn = sum(
        1 for p in scored if p["predicted_label"] == "correct" and p["expected"] == "incorrect"
    )
    tn = sum(1 for p in scored if p["predicted_label"] == "correct" and p["expected"] == "correct")

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall) > 0
        else float("nan")
    )

    # Stratified recall by severity (positive class only — "correct" items
    # have severity=none and we report FPR on them separately).
    severity_counts: dict[str, dict[str, int]] = {}
    for p in scored:
        if p["expected"] != "incorrect":
            continue
        sev = p["severity"] or "unknown"
        bucket = severity_counts.setdefault(sev, {"tp": 0, "fn": 0})
        if p["predicted_label"] == "incorrect":
            bucket["tp"] += 1
        else:
            bucket["fn"] += 1

    # Stratified recall by error_type.
    err_counts: dict[str, dict[str, int]] = {}
    for p in scored:
        if p["expected"] != "incorrect":
            continue
        et = p["error_type"] or "unknown"
        bucket = err_counts.setdefault(et, {"tp": 0, "fn": 0})
        if p["predicted_label"] == "incorrect":
            bucket["tp"] += 1
        else:
            bucket["fn"] += 1

    summary = {
        "n_items_loaded": len(items),
        "n_scored": len(scored),
        "n_excluded_no_text": skipped_no_text,
        "n_excluded_other": len(items) - len(scored) - skipped_no_text,
        "provider": chosen_provider,
        "model": chosen_model,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
        "recall_by_severity": {
            sev: {
                "n": v["tp"] + v["fn"],
                "recall": _safe_div(v["tp"], v["tp"] + v["fn"]),
            }
            for sev, v in sorted(severity_counts.items())
        },
        "recall_by_error_type": {
            et: {
                "n": v["tp"] + v["fn"],
                "recall": _safe_div(v["tp"], v["tp"] + v["fn"]),
            }
            for et, v in sorted(err_counts.items())
        },
    }

    args.out_json.write_text(
        json.dumps({"summary": summary, "predictions": predictions}, indent=2),
        encoding="utf-8",
    )

    # ---- Markdown report ---------------------------------------------------
    md: list[str] = [
        "# Citecheck Phase 5 (claim verification) eval results",
        "",
        f"Evaluated {len(items)} items over {len(set(p['source_id'] for p in predictions))} sources from `{args.labels.relative_to(REPO_ROOT)}`.",
        f"Provider: `{chosen_provider}`, model: `{chosen_model}`.",
        "",
        "## Headline (binary: predicted INCORRECT vs ground-truth INCORRECT)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Items scored | {len(scored)} / {len(items)} |",
        f"| Excluded (no source text) | {skipped_no_text} |",
        f"| Precision | {_format_pct(precision)} |",
        f"| Recall | {_format_pct(recall)} |",
        f"| False-positive rate | {_format_pct(fpr)} |",
        f"| F1 | {_format_pct(f1)} |",
        "",
        "## Confusion matrix",
        "",
        "| Expected \\ Predicted | correct | incorrect |",
        "|---|---|---|",
        f"| correct | {tn} | {fp} |",
        f"| incorrect | {fn} | {tp} |",
        "",
        "## Recall stratified by severity (false items only)",
        "",
        "| Severity | n | Recall |",
        "|---|---|---|",
    ]
    for sev, v in summary["recall_by_severity"].items():
        md.append(f"| {sev} | {v['n']} | {_format_pct(v['recall'])} |")
    md += [
        "",
        "## Recall stratified by error type (false items only)",
        "",
        "| Error type | n | Recall |",
        "|---|---|---|",
    ]
    for et, v in summary["recall_by_error_type"].items():
        md.append(f"| {et} | {v['n']} | {_format_pct(v['recall'])} |")
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(
        f"DONE. n_scored={len(scored)} precision={_format_pct(precision)} "
        f"recall={_format_pct(recall)} fpr={_format_pct(fpr)} f1={_format_pct(f1)}",
        file=sys.stderr,
    )
    print(f"  results: {args.out_md} and {args.out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
