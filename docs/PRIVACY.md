# Privacy & data retention

This document describes how citecheck handles uploaded PDFs and generated reports. It applies to the public web app (Phase 6) and to any third-party deployment that follows the default configuration.

## CLI usage

The CLI runs locally. No data is sent to citecheck servers. External lookups (Crossref, OpenAlex, Unpaywall) are made directly from your machine and do not include the PDF contents — only metadata (DOIs, titles, author names) parsed from the references.

If you set `CITECHECK_CONTACT_EMAIL`, it is included in API requests to Crossref/OpenAlex as a User-Agent — this is required courtesy for high-volume API users and identifies you to the provider, not to citecheck.

## Web app — uploaded PDFs

**Default policy:** uploaded PDFs are deleted **24 hours** after upload. Reports generated from them persist by UUID and remain accessible at their public-by-URL location until the operator deletes them.

Rationale:
- 24h gives users time to re-fetch the report or re-run with different options.
- Eliminating PDF storage limits the operator's data-protection liability.
- Text-only reports do not contain the original PDF content, only metadata about its references.

## Web app — reports

- Reports are stored as JSON + HTML by UUID.
- Reports are **public-by-URL but not listed** (no directory, no search). Sharing the URL shares the report.
- Reports are not indexed by search engines (robots.txt disallows `/reports/*`).
- Users with the URL can delete their report via a "delete" link.

## What we never store

- The text body of uploaded PDFs (beyond the temporary 24h window required for processing).
- IP addresses beyond what is required for rate limiting (kept ≤30 days, hashed).
- Account information — there are no accounts.

## What we may log

- Aggregate API usage (request counts per hour) for capacity planning.
- Errors and stack traces (sanitized to remove PDF content references).

## Operators running their own instance

If you fork and deploy citecheck on your own infrastructure, you set the retention policy via configuration. The defaults above are recommendations, not contractual obligations of forks.

## Changes

This policy can change. Material changes will be announced via the project changelog and the web app landing page.

---

Last updated: 2026-05-20
