---
status: closed
type: prototype
claimed-by: orchestrator (worker: sonnet)
blocked-by: []
---

## Task

Build upstream cefpyco v0.10.1 unmodified against the installed Cefore v0.12.0 into this repository's venv (cmake from PyPI, not apt), then prove on ReCefore-provisioned Mininet hosts with their real `cefnetd.conf` (CS_MODE=2, csmgrd present on cache nodes):

1. `create_handle(ceforedir=<node conf dir>)` binds to that node's cefnetd.
2. An app that `register`s `ccnx:/semidx/g1/hN` receives an Interest for a longer name under it, on a csmgrd node and on a non-csmgrd node.
3. `send_data` with no chunk TLV answers it; the requester's `receive` gets it within `lifetime=2000`.
4. State-style names that change per request are never answered from a Content Store with an older object.
5. `send_interest(chunk_num=k)` fetches all chunks of a cefputfile-published object.

Patch only what fails, on a branch of `/home/lab_shared/cefpyco`. Keep the smoke code as the binding's regression test.

## Acceptance

Each of 1–5 has recorded command output. A negative result on 2 or 4 stops the effort and goes back to the user.

## Resolution (2026-10-22)

- cefpyco は個人フォーク koh11235813/cefpyco のブランチ `v0.12-compat`（`6434fde`）。上流 v0.10.1（`854fb86`）からの修正は3点: `register_route()` の存在しない opcode 名、Data 解析の `chunk_num_f` 未参照（chunk 無しが chunk 0 に見えた）、`receive(timeout_ms)` が 1 秒単位でしか戻らない件。単体試験 `semindex_smoke` 33 passed。
- 1台構成（root 実行）: Mode A（CS_MODE=2、`Cache Mode : Excache`）と Mode B（CS_MODE=0、`Cache Mode : None`）で項目 1〜4 が PASS、Mode A で項目 5（cefputfile の chunk 取得、5 chunks / 5000 bytes 一致）も PASS。直す前の初回 Mode A では、配送は通ったが chunk_num の解析不具合で consumer/state_freshness が FAIL していた。
- 同じ名前の2回目: Mode A では CS が答え（producer は1回だけ）、Mode B では producer が2回答えた。`cache_time=0` は「CS に置かない」ではない（map の「実機で確かめたこと」）。
- 2台構成: ReCefore 上の h0（CS_MODE=1）と h1（CS_MODE=2）で、それぞれのノードの cefnetd.conf を使う別々の daemon に `create_handle(ceforedir=...)` で結び、FIB 経路（prefix_routes）を通って両方向に Interest/Data が届いた（チケット 06 の2台同期で確認）。
- 記録: 単一ホストのスクリプトと出力は `docs/wayfinder/semantic-index/evidence/01/`（`mode_a_before_fix.out` は直す前、`mode_a.out` と `mode_b.out` は直した後）。
