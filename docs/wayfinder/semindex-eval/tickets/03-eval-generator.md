---
status: open
type: task
claimed-by:
blocked-by: []
---

## Task

計測用の生成器。切断率 p と seed から、1 回の run 分のシナリオとエージェント設定を作る。

- リンクごとに 5 秒の区切りで確率 p の切断を引き、`link_down` と `link_up` のイベントとして並べる。
- h0 と h4 のレコード生成の予定（2 秒に1件、各 30 件）を作る。属性は seed から各語彙の一様分布で引く。
- h2 の probe（10 秒ごと、`zone` は一様に引く）を作る。
- run の長さはおよそ 120 秒。

既存の `semindex.gen` を部品として使い、同じことを書き直さない。乱数は seed だけから決まり、同じ seed なら同じ出力になる。

## Acceptance

同じ (p, seed) で出力がバイト単位で一致し、seed が違えば切断の予定と属性が変わることを、単体試験で確かめる。p = 0 で切断イベントがゼロになり、p を大きくすると落ちている時間の割合が p に近づくこと（多数の seed での平均）も確かめる。生成した YAML が ReCefore の validator を通る。
