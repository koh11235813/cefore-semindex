---
status: open
type: task
claimed-by:
blocked-by: []
---

## Task

5 ノードの直線（h0–h1–h2–h3–h4、全ホストでエージェント）を、切断なしで1回だけ実機で通す。手元の ReCefore の checkout を origin/main に fast-forward してから走らせる。生成器は今のまま使い、試験の記述を1つ手で書く（h0 と h4 を origin、h2 から probe）。

## Acceptance

5 台すべてのエージェントが起動して `counters` 行を書き、h0 と h4 のレコードが全ノードに届く。`semindex.analyze` が exit 0 で、発見までの時間と通信量を出す。1 回の run にかかった実時間（Mininet の起動と片づけを含む）を Resolution に記録し、map の全体時間の見積もりを更新する。
