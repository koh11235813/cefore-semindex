---
status: closed
type: task
claimed-by: orchestrator
blocked-by: [02, 04]
---

## Task

集計スクリプト。保存先の run を全部読み、map の指標の定義どおりに表と図を出す。発見までの時間は打ち切りを残し、届いた割合の時間変化と打ち切りの割合を並べる。全ノードが一致するまでの時間、probe の発見率、レコード1件あたりの制御の通信量を、条件ごとに出す。印つきの run は集計から外し、条件ごとの数を表に載せる。`decision` 行は、種類ごとの件数と、状態の欄の分布だけを出す（段 2 の設計の材料）。

`semindex.analyze` の関数を部品として使い、同じ計算を書き直さない。

## Acceptance

手で作った小さな run の集合で、各指標の値（打ち切りの扱いを含む）を完全一致で確かめる。出力は図にそのまま使える形の CSV で、コマンド1つで作り直せる。図そのものはチケット 06 で報告を作るときに決める（2026-10-23 変更。venv に matplotlib がなく、描画のための依存を足すかは利用者の判断なので）。

## Resolution (2026-10-23)

- `semindex.evalagg` を `feat/eval-aggregate` で実装し、統合ブランチに入れた。出力は `runs`・`discovery`・`survival`・`consistency`・`summary`・`decisions` の 6 つの CSV で、コマンド 1 つで作り直せる。
- 手で作った小さな run の集合で、各指標（打ち切りの扱い、Kaplan–Meier、Wilson 区間を含む）を完全一致で確かめた。
- `DONE` の無い run と判定の無い run は、既定では集計から外して数だけ出す（`--include-undone`、`--include-unknown`）。
- pilot2 で、probe の発見率が答えの無い probe も取りこぼしに数えることが分かった。正解による分類は `feat/eval-probe-truth` で足す（map の「指標の定義」）。
