---
status: open
type: task
claimed-by:
blocked-by: [02, 04]
---

## Task

ReCefore: a generic scheduled event `{at, type: host_app_restart, host, downtime_s}` that stops every host_apps process of one host through the existing stop path (SIGTERM, 2 s grace, SIGKILL), waits `downtime_s`, and starts them again with the argv/env/cwd/log expanded at first start (same `{t0_ms}`). The previous life's log must survive the restart. One results.json outcome row per event, in the shape link_down/link_up rows use. `host_down` is unchanged. Branch + PR from the personal fork, stacked on prefix_routes.

## Acceptance

Unit tests with the recording fake (restart, downtime, a restart racing shutdown leaves no orphan). Smoke `min_host_app_restart` shows two marker lines on the restarted host with the same t0 and different pids, one line on the others, and a successful restart row.
