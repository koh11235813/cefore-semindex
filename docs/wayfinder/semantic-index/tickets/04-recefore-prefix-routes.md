---
status: open
type: task
claimed-by:
blocked-by: [03]
---

## Task

ReCefore: `prefix_routes: {prefix: host_idx}` merged into the existing FIB generation (`compute_fib_for_uris`), independent of put/pubsub_pub publishers. Branch + PR from the personal fork.

## Acceptance

Unit tests on the computed FIB for a 3-host line with and without publication events; smoke shows the routes in `cefstatus`.
