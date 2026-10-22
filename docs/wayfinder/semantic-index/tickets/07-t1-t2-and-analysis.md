---
status: closed
type: task
claimed-by:
blocked-by: [02, 03, 04, 06, 08]
---

## Task

Experiment generator (one description → ReCefore YAML + per-node agent configs), T1 and T2 scenarios, and the analysis script (discovery latency, all-nodes-consistent time, control bytes, discovery vs retrieval).

## Acceptance

T1 and T2 pass as defined in the map, with the run directories and analysis output recorded in the Resolution.

## Resolution (2026-10-23)

- semindex PR #4 で main に反映（merge `19484bd`）。実験記述（`experiments/<t>.json`）から ReCefore YAML とノードごとのエージェント設定を1つの生成スクリプト（`semindex.gen`）で出すようにし、T1・T2a（欠落回復）・T2b（公開者自身の再起動）・T2c（中継ノードの再起動）の4シナリオと解析スクリプト（`semindex.analyze`、発見遅延・全ノード一致時刻・制御通信量）を追加した。
- 実行コマンドと証跡は `docs/wayfinder/semantic-index/evidence/07/README.md` と同ディレクトリの `check-*.out` / `analyze_ex992*.json` / `t1/t1.yaml` ほかに記録。
- T1 / T2a / T2b / T2c は 2 秒バックオフでいずれも `exit 0`。T2a は `(h0, 3)` への `fetch_timeout` をちょうど 2 回（4.0 秒間隔）記録してから格納に成功した。
- T1 の発見遅延（切断前 seq 1〜5）: `h1` 1150〜1408 ms、`h2` 1959〜2301 ms（`analyze_ex9921.json`）。
- T2b の再起動から準備完了の証拠（ほかのノードが h0 から受けた `state_rx`）まで: 再起動の完了時刻（`results.json` の `ts`）から 159 ms（ex9923）。
- 初回取得のバックオフを 1 秒始まりから 2 秒始まりに変更した（2026-10-23、ユーザー決定）。1 秒始まりだと T2a で Cefore の PIT（lifetime 2000 ms + CefC_Pit_CleaningTime 1 秒）が同名 Interest の最初の再送を吸収し、3 回の `fetch_timeout`（`exit 1`）になっていた。1 秒バックオフ時の旧ラン（ex9911〜9914_seed42）は `tests/test_check.py` の実機リグレッション試験のフィクスチャとして `/home/lab_shared/semindex-fixtures/` に保存されている（元の実行場所は ReCefore 側 worktree の `logs/semindex/`、保存したのは checker/analyze が読む `results.json` と `debug/node_dirs/hN/semindex/events.jsonl` のみ。残りは Cefore の鍵材料を含みうるため保存していない。semindex ブランチ `test/durable-live-fixtures`、別 PR）。
- T2b のグレース判定の基準点（grace anchor）は「再起動時刻 + 10 秒」で頭打ちにしている。
