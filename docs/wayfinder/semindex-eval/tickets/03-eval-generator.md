---
status: closed
type: task
claimed-by: orchestrator
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

## Resolution (2026-10-23)

- `semindex.evalgen`（`--p --seed --stage --out`）を `feat/eval-gen` で実装し、統合ブランチに入れた。
- 同じ (p, seed) でバイト単位で一致すること、seed が違えば切断の予定と属性が変わること、p = 0 で切断イベントがゼロになること、多数の seed の平均で落ちている時間の割合が p に近づくことを、単体試験で確かめた。
- レコードと probe の乱数は seed だけから引く（`0670bdb`）。p を変えても予定が同じになり、条件の比較が対応のある比較になる。pilot2 で、同じ seed の設定が `group` の名前以外一致することを確かめた。
- 生成した YAML は、trial の 3 回、pilot2 の 50 回とも ReCefore が受け付けて最後まで走った。
