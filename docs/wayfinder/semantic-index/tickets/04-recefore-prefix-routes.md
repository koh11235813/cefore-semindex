---
status: closed
type: task
claimed-by:
blocked-by: [03]
---

## Task

ReCefore: `prefix_routes: {prefix: host_idx}` merged into the existing FIB generation (`compute_fib_for_uris`), independent of put/pubsub_pub publishers. Branch + PR from the personal fork.

## Acceptance

Unit tests on the computed FIB for a 3-host line with and without publication events; smoke shows the routes in `cefstatus`.

## Resolution (2026-10-23)

- ReCefore（cefore-emu-sslab）PR #39 で main に反映（merge `f42ec63`）。
- `prefix_routes: {prefix: host_idx}` は既存の `compute_fib_for_uris` に足し込まれ、put/pubsub_pub の publisher の有無とは独立に経路が張られる。
