---
status: open
type: task
claimed-by:
blocked-by: []
---

## Task

ReCefore: add `host_apps:` — per-host long-running processes with argv/env/cwd placeholders (node name, node dir, cefnetd.conf dir, scenario t0 epoch ms). Start after cefnetd via `CommandRunner.start`; stop inside `shutdown_runtime_resources` before daemon teardown, each host independently (DaemonFleet shape). The scheduler's `at: 0` must equal the t0 passed to the apps. No sync knowledge in ReCefore. Branch + PR from the personal fork.

## Acceptance

Unit tests with the recording fake runner; a smoke config in `config/examples/` whose app writes a marker file per host; `cefore-run-tests` stays green.
