# チケット 07 の実行証跡

T1 / T2a / T2b / T2c を実機（mininet-debian、root）で通した check/analyze の出力と、生成されたシナリオ YAML。

## 実行コマンド

```
# 生成（cefore-semindex checkout）
.venv/bin/python3 -m semindex.gen experiments/<t>.json <outdir>

# 実行（ReCefore = cefore-emu-sslab checkout、root）
sudo setsid timeout 300 .venv/bin/python3 -m src disaster --config <outdir>/<t>/<t>.yaml --no-cli

# 確認（cefore-semindex checkout）
.venv/bin/python3 -m semindex.check <t> experiments/<t>.json <run_dir>

# 解析（cefore-semindex checkout）
.venv/bin/python3 -m semindex.analyze <run_dir>
```

`<t>` は `t1` / `t2a` / `t2b` / `t2c`。生成のときに実験番号 `num` を 991x から 992x に変えたので、1 秒バックオフの古い実行（下記）を上書きしていない。

## 実行したラン

実行場所はもともと ReCefore 側の worktree `/home/lab_shared/wt/recefore-host-app-restart/logs/semindex/ex<N>_seed42` だった。その worktree は撤去されるため、checker/analyze が読む入力（`results.json` と `debug/node_dirs/hN/semindex/events.jsonl` のみ）を `/home/lab_shared/semindex-fixtures/ex<N>_seed42/` にコピーして恒久的な場所にした。`node_dirs` の残り（cefnetd/csmgrd のログなど、Cefore の鍵材料を含みうるもの）はコピーしていない。

- 2 秒バックオフ（本採用）: `/home/lab_shared/semindex-fixtures/ex9921_seed42`〜`ex9924_seed42`（t1, t2a, t2b, t2c の順）。4 件とも `exit 0`。
- 1 秒バックオフ（旧設定、比較用）: `/home/lab_shared/semindex-fixtures/ex9911_seed42`〜`ex9914_seed42`。t2a は `exit 1` で、`fetch_timeout` が 3 回記録された。原因は Cefore の PIT が同名 Interest のエントリを lifetime（2000 ms）+ CefC_Pit_CleaningTime（1 秒）の間保持し、1 秒後の最初の再送をそのエントリが吸収して producer に届かなかったこと（詳細は map.md「実機で確かめたこと」）。この旧ランは `tests/test_check.py` の実機リグレッション試験が `/home/lab_shared/semindex-fixtures/` をフィクスチャとして参照しており（semindex ブランチ `test/durable-live-fixtures`、別 PR）、無ければその試験はスキップする。移設後も checker/analyze の exit code は移設前と同一（991x: 0, 1, 0, 0 / 992x: 0, 0, 0, 0）。本 README では 1 秒バックオフ側の check/analyze 出力はコピーしていない（チケットの指示で 2 秒バックオフ側の4ファイルのみを証跡として残す）。

## このディレクトリの内容

- `check-t1.out` / `check-t2a.out` / `check-t2b.out` / `check-t2c.out`: 2 秒バックオフのランに対する `semindex.check` の出力。
- `analyze_ex9921.json`〜`analyze_ex9924.json`: 同じランに対する `semindex.analyze` の出力（t1=9921, t2a=9922, t2b=9923, t2c=9924）。
- `t1/t1.yaml` / `t2a/t2a.yaml` / `t2b/t2b.yaml` / `t2c/t2c.yaml`: `semindex.gen` が生成した ReCefore 用シナリオ YAML。

生成された YAML の `host_apps.argv` にはエージェント設定ファイルの絶対パスが入っており、これは実行当時のスクラッチパス（`/tmp/claude-1001/.../scratchpad/wave3d/<t>/<t>/{node}.json`）を指している。このパスは実行時にしか存在せず、再現時は `semindex.gen` を自分の環境で再実行して得られる新しいパスに置き換わる。鍵やトークンなど秘匿情報は含まれていない（確認済み）。

## 結果（チケット 07 の受け入れ事項）

- T1 / T2a / T2b / T2c はいずれも 2 秒バックオフで `exit 0`。
- T2a: `h1` が `(h0, 3)` について `fetch_timeout` をちょうど 2 回記録し、間隔は 4.0 秒（`1792689074429` → `1792689078441`、実測 4012 ms）。その後 `(h0, 3)` を格納している（`check-t2a.out`）。
- T1 の発見遅延（`analyze_ex9921.json` の `discovery_latency_ms`、切断前の seq 1〜5 分）: `h1` は 1150〜1408 ms、`h2` は 1959〜2301 ms。
- T2b の再起動から準備完了の証拠まで: checker と同じ選び方（ほかのノードが h0 から受けた `state_rx`）で、再起動の完了時刻（`results.json` の `ts`）から ex9923 で 159 ms（1 秒バックオフの ex9913 では 185 ms）。T2b の判定で、猶予の基準を「再起動 + 10 秒」で頭打ちにしている上限は、これより十分に大きい。
- 初回取得のバックオフは 1 秒始まりから 2 秒始まりに変更した（2026-10-23、ユーザー決定）。T2b のグレース判定の基準点（grace anchor）は「再起動時刻 + 10 秒」で頭打ちにしている。
