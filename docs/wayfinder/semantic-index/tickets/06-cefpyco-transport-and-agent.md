---
status: closed
type: task
claimed-by:
blocked-by: [01, 05]
---

## Task

cefpyco transport adapter and the single-threaded agent loop: one `register` per agent, sync timer, generate/probe schedules relative to t0, probe = local query then chunked body fetch, `fault.drop_rec`, JSONL events and byte counters, clean shutdown on SIGTERM.

## Acceptance

Two-host Mininet run under ReCefore `host_apps` shows records syncing both ways and the expected events.

## Resolution (2026-10-23)

- semindex PR #3 で main に反映（merge `e1c7a39`）。
- cefpyco transport adapter と単一スレッドの Sync agent ループ（register、同期タイマー、t0 起点の generate/probe スケジュール、`fault.drop_rec`、JSONL イベントとバイトカウンタ、SIGTERM での正常終了）を追加。2ホスト Mininet（ReCefore `host_apps`）での両方向同期は map.md「実機で確かめたこと（2026-10-22）」に記録済み。
