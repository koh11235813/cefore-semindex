---
status: open
type: task
claimed-by:
blocked-by: []
---

## Task

Transport-independent core of the Sync agent: Index record schema and canonical JSON codec, SQLite store (WAL, record + progress in one transaction, seq allocation surviving restart, conflict detection), Holder state codec with the 8 KiB / 64-extra limits, missing-set computation with the three distinct quantities, retry/backoff and in-flight limits, name construction/parsing. A fake in-memory multi-node transport with partitions drives the unit tests. Fixes the wire names in the map.

## Acceptance

Unit tests cover: gaps (1,2,4 ⇒ 3 missing), reordering, duplicates, conflict, restart without seq collision, A→B→C relay with A partitioned, oversize rejection, truncation. Exact-match asserts per repo testing policy.
