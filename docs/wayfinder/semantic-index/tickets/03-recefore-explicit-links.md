---
status: open
type: task
claimed-by:
blocked-by: []
---

## Task

ReCefore: disaster scenario accepts `links: [[a, b], ...]` and builds exactly that topology; absent, the seeded mesh is unchanged (seeded golden snapshots stay byte-identical). `link_down`/`link_up` on those pairs work. Branch + PR from the personal fork.

## Acceptance

Unit tests on topology construction and validation; a 3-host line smoke with a link_down/link_up and a get across it.
