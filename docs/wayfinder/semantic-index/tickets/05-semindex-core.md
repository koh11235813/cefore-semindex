---
status: closed
type: task
claimed-by: orchestrator (worker: sonnet)
blocked-by: []
---

## Task

Transport-independent core of the Sync agent: Index record schema and canonical JSON codec, SQLite store (WAL, record + progress in one transaction, seq allocation surviving restart, conflict detection), Holder state codec with the 8 KiB / 64-extra limits, missing-set computation with the three distinct quantities, retry/backoff and in-flight limits, name construction/parsing. A fake in-memory multi-node transport with partitions drives the unit tests. Fixes the wire names in the map.

## Acceptance

Unit tests cover: gaps (1,2,4 ⇒ 3 missing), reordering, duplicates, conflict, restart without seq collision, A→B→C relay with A partitioned, oversize rejection, truncation. Exact-match asserts per repo testing policy.

## Resolution (2026-10-22)

- semindex PR #1 で main に反映（merge `a00ad63`）。
- unit 98 passed。Fable の phase review で各コミットの試験と16種類の mutation を確認し、生き残った4種類（1024 上限、holder 照合の2分岐、1トランザクション）は follow-up の試験で潰した。
- ワイヤ名・Index record・Holder state の JSON は map の「Wire names (fixed)」で確定。
