---
status: closed
type: task
claimed-by: orchestrator (worker: sonnet)
blocked-by: []
---

## Task

ReCefore: add `host_apps:` — per-host long-running processes with argv/env/cwd placeholders (node name, node dir, cefnetd.conf dir, scenario t0 epoch ms). Start after cefnetd via `CommandRunner.start`; stop inside `shutdown_runtime_resources` before daemon teardown, each host independently (DaemonFleet shape). The scheduler's `at: 0` must equal the t0 passed to the apps. No sync knowledge in ReCefore. Branch + PR from the personal fork.

## Acceptance

Unit tests with the recording fake runner; a smoke config in `config/examples/` whose app writes a marker file per host; `cefore-run-tests` stays green.

## Resolution (2026-10-22)

- ReCefore PR #37 で main に反映（merge `ab655e2`）。
- smoke `min_host_apps` を cefore-run-tests に登録し、root で `[OK]`。3台の marker の t0_ms が同一（`1792661585524`）、停止後に `sleep 1000` が残らないことも確認。
- unit 1670 passed / 8 skipped、mypy、ruff clean（PR 時点）。
