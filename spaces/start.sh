#!/usr/bin/env bash
# spaces/start.sh — single-container entrypoint for HF Spaces.
#
# Launches GROBID in the background on 127.0.0.1:8070, polls its
# /api/isalive endpoint until it responds, then execs uvicorn on
# 0.0.0.0:7860 (the port HF Spaces routes public traffic to).
#
# Why exec uvicorn instead of letting the script linger?  Exec replaces
# the shell with the Python process so signal handling (SIGTERM from
# the orchestrator when the Space restarts) reaches uvicorn cleanly.

set -euo pipefail

# ---------------------------------------------------------------------------
# Sanity-check required env vars before doing anything else — fail fast
# with an actionable message rather than mid-pipeline.
# ---------------------------------------------------------------------------
if [[ -z "${CITECHECK_CONTACT_EMAIL:-}" ]]; then
    echo "FATAL: CITECHECK_CONTACT_EMAIL not set." >&2
    echo "  Set it via the HF Space Secrets UI (Settings -> Variables and secrets)." >&2
    echo "  Crossref / OpenAlex / Unpaywall all require this for polite-pool access." >&2
    exit 1
fi

if [[ -z "${CEREBRAS_API_KEY:-}" ]] && [[ "${CITECHECK_WEB_LLM_PROVIDER:-cerebras}" == "cerebras" ]]; then
    echo "WARN: CEREBRAS_API_KEY not set but provider is cerebras." >&2
    echo "  Phase 5 (claim verification) will fail at call-time." >&2
    echo "  Set CEREBRAS_API_KEY via HF Space Secrets, or override CITECHECK_WEB_LLM_PROVIDER." >&2
fi

# ---------------------------------------------------------------------------
# Launch GROBID — the base image's expected start command is
# /opt/grobid/grobid-service/bin/grobid-service.  We send logs to a file
# and tail them in parallel so HF Space build logs show GROBID startup
# while we wait for it.
# ---------------------------------------------------------------------------
GROBID_LOG=/tmp/grobid.log
echo "[start.sh] Launching GROBID on 127.0.0.1:8070 (log: $GROBID_LOG)..."
# The base image's CMD was `./grobid-service/bin/grobid-service` from
# WorkingDir /opt/grobid; the launcher script uses relative paths to
# locate its config files, so we cd there before exec'ing.
(cd /opt/grobid && ./grobid-service/bin/grobid-service) > "$GROBID_LOG" 2>&1 &
GROBID_PID=$!

# Wait up to 120s for /api/isalive to respond 200.  GROBID 0.8.2-crf on
# a 2-vCPU HF Space typically warms up in ~30-45s.
GROBID_READY=0
for i in $(seq 1 120); do
    if curl -fsS --max-time 2 http://127.0.0.1:8070/api/isalive >/dev/null 2>&1; then
        echo "[start.sh] GROBID ready after ${i}s."
        GROBID_READY=1
        break
    fi
    # Surface the most recent GROBID log lines once every 10s while we wait,
    # so HF build logs don't go silent during the long warm-up.
    if (( i % 10 == 0 )); then
        echo "[start.sh] Still waiting for GROBID (${i}s elapsed)..."
        tail -n 3 "$GROBID_LOG" 2>/dev/null | sed 's/^/[grobid] /' || true
    fi
    sleep 1
done

if [[ "$GROBID_READY" != "1" ]]; then
    echo "FATAL: GROBID did not become ready within 120s." >&2
    echo "----- last 50 lines of $GROBID_LOG -----" >&2
    tail -n 50 "$GROBID_LOG" >&2 || true
    exit 1
fi

# ---------------------------------------------------------------------------
# Boot uvicorn on 0.0.0.0:7860.  exec replaces the shell so SIGTERM is
# routed to the Python process directly.
# ---------------------------------------------------------------------------
echo "[start.sh] Starting uvicorn on 0.0.0.0:7860..."
exec uvicorn citecheck.web.app:app \
    --host 0.0.0.0 \
    --port 7860 \
    --workers 1 \
    --log-level info \
    --proxy-headers \
    --forwarded-allow-ips='*'
