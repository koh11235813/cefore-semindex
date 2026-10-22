# CLAUDE.md

Distributed Semantic Index over Cefore. Vocabulary: CONTEXT.md. Decisions: docs/adr/. Current effort and spec: docs/wayfinder/semantic-index/map.md. PLAN.md is ephemeral scratch.

ReCefore (`/home/lab_shared/cefore-emu-sslab`) is the Mininet testbed only; it must not learn about synchronization (ADR-0001).

## Program shape

- Unix philosophy: small programs that do one thing, compose, and speak text streams (JSON lines).
- The sync core is transport-independent; cefpyco is reached only through the transport adapter.
- The agent is single-threaded: cefpyco `receive()` holds the GIL and shares one static buffer.
- Check every payload size before `send_data`; cefpyco does not.

## Comments

Comment intent, not mechanics. Record fix provenance inline with a date (`# 2026-10-22 fix: ...`). Write docstrings for every function and class. Do not reference tickets, PRs or callers in comments.

## Development

- `uv sync`, then `.venv/bin/python3 -m pytest`.
- Mininet runs need root: `sudo .venv/bin/python3 ...` (never bare `python3`).
- Assert diagnostics by exact match, not substring.
- Signatures are not implemented; never describe validated records as authenticated.
