---
status: closed
type: task
claimed-by: orchestrator
blocked-by: []
---

## Task

5 ノードの直線（h0–h1–h2–h3–h4、全ホストでエージェント）を、切断なしで1回だけ実機で通す。手元の ReCefore の checkout を origin/main に fast-forward してから走らせる。生成器は今のまま使い、試験の記述を1つ手で書く（h0 と h4 を origin、h2 から probe）。

## Acceptance

5 台すべてのエージェントが起動して `counters` 行を書き、h0 と h4 のレコードが全ノードに届く。`semindex.analyze` が exit 0 で、発見までの時間と通信量を出す。1 回の run にかかった実時間（Mininet の起動と片づけを含む）を Resolution に記録し、map の全体時間の見積もりを更新する。

## Resolution (2026-10-23)

- ReCefore の main の checkout を origin/main（`be78de6`）に fast-forward してから、root で1回走らせた。試験の記述は手で書いたもの（h0 と h4 が各 30 件を 2 秒ごとに作り、h2 から 10 秒ごとに probe、切断なし、`duration: 125`）を、今の `semindex.gen` にかけた。
- 5 台すべてのエージェントが起動し、最後の行に `counters` を書いた。h0 と h4 は相手の 30 件を、h1〜h3 は 60 件を保存した。`semindex.analyze` は exit 0。
- 発見までの時間（中央値、30 件ずつ）: h0 のレコードは h1 で 739 ms、h2 で 1827 ms、h3 で 2527 ms、h4 で 4095 ms。h4 のレコードは h3 で 880 ms、h2 で 2103 ms、h1 で 2607 ms、h0 で 3766 ms。1 ホストごとにおよそ 0.7〜0.9 秒延びる。
- 制御の通信量（送信）: 端の h0 が 18 KB、中継の h1〜h3 がそれぞれ 36〜37 KB。
- 1 回の run の実時間は 149 秒、`mn -c` の片づけは 1 秒。
- run の入力（`results.json`、各ノードの `events.jsonl`）、試験の記述、生成した設定、analyze の出力は `/home/lab_shared/semindex-eval/smoke/e01/42/` に置いた。
