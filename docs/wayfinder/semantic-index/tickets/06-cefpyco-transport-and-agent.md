---
status: open
type: task
claimed-by:
blocked-by: [01, 05]
---

## Task

cefpyco transport adapter and the single-threaded agent loop: one `register` per agent, sync timer, generate/probe schedules relative to t0, probe = local query then chunked body fetch, `fault.drop_rec`, JSONL events and byte counters, clean shutdown on SIGTERM.

## Acceptance

Two-host Mininet run under ReCefore `host_apps` shows records syncing both ways and the expected events.
