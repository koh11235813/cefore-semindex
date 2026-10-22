---
status: closed
type: task
claimed-by: orchestrator (worker: sonnet)
blocked-by: []
---

## Task

ReCefore: disaster scenario accepts `links: [[a, b], ...]` and builds exactly that topology; absent, the seeded mesh is unchanged (seeded golden snapshots stay byte-identical). `link_down`/`link_up` on those pairs work. Branch + PR from the personal fork.

## Acceptance

Unit tests on topology construction and validation; a 3-host line smoke with a link_down/link_up and a get across it.

## Resolution (2026-10-22)

- ReCefore PR #38 で main に反映（merge `06533e5`。host_apps との衝突はマージ `705c5b5` で解消）。
- smoke `min_explicit_line`: 初回は autotest モードの warmup が `during` を先取りしていて FAIL。`warmup_gets` で先取りを `before` に限り、put を `at: 0` に寄せて組み直した後、root で `[OK]`（eval の get: before True / during False / after True）。h1 の `h1-eth0 192.168.1.2`、`h1-eth1 192.168.2.2` も確認。
- host_apps とのマージ後に両方の smoke を再実行して `[OK]`。unit 1721 passed / 8 skipped。
- 後続: connect での `links` 拒否を validator（`OptionSpec.block`）へ移す、meta.json に `links` を記録する。
