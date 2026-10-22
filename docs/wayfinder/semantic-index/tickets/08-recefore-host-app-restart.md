---
status: closed
type: task
claimed-by:
blocked-by: [02, 04]
---

## Task

ReCefore: a generic scheduled event `{at, type: host_app_restart, host, downtime_s}` that stops every host_apps process of one host through the existing stop path (SIGTERM, 2 s grace, SIGKILL), waits `downtime_s`, and starts them again with the argv/env/cwd/log expanded at first start (same `{t0_ms}`). The previous life's log must survive the restart. One results.json outcome row per event, in the shape link_down/link_up rows use. `host_down` is unchanged. Branch + PR from the personal fork, stacked on prefix_routes.

## Acceptance

Unit tests with the recording fake (restart, downtime, a restart racing shutdown leaves no orphan). Smoke `min_host_app_restart` shows two marker lines on the restarted host with the same t0 and different pids, one line on the others, and a successful restart row.

## Resolution (2026-10-23)

- ReCefore（cefore-emu-sslab）PR #41 で main に反映（merge `be78de6`）。
- `host_app_restart` イベントを追加: 既存の停止経路（SIGTERM、2 秒グレース、SIGKILL）で対象ホストの host_apps を止め、`downtime_s` 後に最初の起動と同じ argv/env/cwd/`{t0_ms}` で立て直す。前のライフのログは残したまま、新しいライフのログを別ファイルで続ける。
- 実機 smoke（root）で `min_host_apps` と `min_host_app_restart` の両方が通過した。`min_host_app_restart` の再起動対象ホストでは `h1.log` と `h1.log.life2` が両方残っていることを確認した（前のライフのログが再起動で消えていない）。
