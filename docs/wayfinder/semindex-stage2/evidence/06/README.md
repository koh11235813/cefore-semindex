# index.db の `synchronous` を NORMAL にしたときの、受け取りの速さ

段 2 の本番の集計（`evidence/05/` の 4.）で、取りに行く側のエージェントが受け取った 1 件を処理する区間に 33〜74 ms かかり、その大半は 1 件ごとの SQLite の commit の fsync（WAL、`synchronous` は SQLite の既定の FULL）だと推定した。その推定を確かめるために、`--store-synchronous` を足した版で FULL と NORMAL を並べて回した計測（2026-10-29）。この結果をもとに、エージェントの既定を NORMAL にした（`docs/adr/0003-*`）。

## 回した段

どれも直線 5 ノード、seed 1〜3、15 回すべて有効。semindex は `0e67181`、ReCefore は `be78de6`（どちらも dirty でない）。回し方は `micro24.sh`（root で `sudo bash micro24.sh`）。2026-10-29 05:23〜06:02（UTC）に続けて回した。

| 段 | 条件 | `synchronous` |
|---|---|---|
| `s3fk8` | k = 8、既定の方針（2 秒 ± 20%）、p = 0 | フラグなし（そのときのコードでは SQLite の既定の FULL。`meta.json` は null） |
| `s3nk8` | 同じ | NORMAL |
| `s3fmix` | 4 つの間隔（0.5・1.359・3.695・10.043 秒）を一様に混ぜる、j = 0.1、k = 4、p = 0 | フラグなし（FULL） |
| `s3nmix` | 同じ | NORMAL |
| `s3nc32` | k = 4、既定の方針、隣ごとの同時取得の上限 32、p = 0.4 | NORMAL |

`s3nc32` の比べる相手は、試走の `t2c32k4`（同じ条件で FULL、2026-10-26 に `99fb5fe` で回した 3 回、`evidence/04/`）。日も版も違うので、この組の比べは、ほかの 2 組ほど条件がそろっていない。

## 結果

`micro24_an.txt`（`micro24_an.py`）と `micro24_cap.txt`（`evidence/04/capcheck.py`）と `micro24_summary.csv`（evalagg の `summary.csv`）から。

| 組 | 1 件の処理の区間（中央値／99 パーセンタイル／最大） | 取得 2〜4 件のときの受け取り待ち | 取得の平均（空き待ち + 受け取り待ち） | 発見の中央値 |
|---|---|---|---|---|
| k = 8、FULL | 89 ／ 288 ／ 528 ms | 469 ms | 1063 ms（212 + 851） | 6.08 s |
| k = 8、NORMAL | 0 ／ 1 ／ 1 ms | 2 ms | 4 ms（0 + 4） | 2.32 s |
| 混合、FULL | 55 ／ 125 ／ 539 ms | 108 ms | 663 ms（298 + 365） | 10.03 s |
| 混合、NORMAL | 0 ／ 1 ／ 1 ms | 2 ms | 5 ms（2 + 3） | 8.65 s |
| 上限 32、p = 0.4、FULL（試走 `t2c32k4`） | 33 ／ 92 ／ 1693 ms | 88 ms | 852 ms（112 + 740） | 10.5 s |
| 上限 32、p = 0.4、NORMAL | 0 ／ 1 ／ 1 ms | 2 ms | 8 ms（2 + 6） | 7.55 s |

- 1 件の処理の区間は、同じループの中の `record_stored`（処理の前）と、すぐ後の同じ隣への `fetch_plan`（処理の後）の時刻の差（`evidence/05/commit_time.py` と同じ数え方）。続けて届く間隔（同じ隣からの `record_stored` どうしの間隔、2 秒未満）も、NORMAL ではどの組も中央値 0 ms。
- 受け取り待ちは、出ている取得の数によらず、NORMAL では 1〜9 ms（上限 32 の 9 件以上で 9 ms）。FULL では、1 件増えるごとにおよそ 50 ms 延びていた。
- 上限 32 の組の時間切れは、FULL の試走で 159 件（リンクがつながっている間に 93 件、落ちている間に 66 件）、NORMAL では 0 件。
- 制御の通信量／レコードは、p = 0 の 2 組では FULL と NORMAL でほぼ同じ（どれも 1.38 KB）。

## 読めること

- 受け取った 1 件を処理する区間は、`synchronous` を NORMAL にするだけで、測れたものの中央値が 0 ms、99 パーセンタイルと最大が 1 ms になった。測れるのは、保存のすぐ後に同じ隣への取得の計画が続き、2 秒未満のものだけで、NORMAL の 9 回の 11,520 件のうち 2,691 件。時刻は整数 ms なので、1 ms 未満とまでは言えない。全件について見ても、受け取り待ちの平均は 3〜6 ms（`micro24_an.txt` の `receive_wait`）。`evidence/05/` で推定した「処理の区間の大半は commit の fsync」は当たっていた。区間に入るほかの処理（検証、ハッシュ、INSERT、進捗の更新と読み出し、時間切れの確認）は、測れた区間では合わせて 1 ms ほど以下。
- WAL の checkpoint の発生と所要時間は、直接は測っていない。測れた区間には長い止まりは出ていないが、2 秒以上の区間は数え方から外れるので、まれな長い止まりが無いとまでは言えない。
- 取得の速さが上がると、発見も早くなる。k = 8 では中央値が 6.08 秒から 2.32 秒になった。混合の組は 10.03 秒から 8.65 秒で、縮み方が小さい。混合の組では、長い間隔を選んだ状態確認の待ちが遅れの多くを占めるため（`evidence/05/` の 2. の待ちと取得の内訳）と見ている。上限 32 の組の 10.5 秒は、日も版も違う試走の値。
- どれも 3 seed ずつの結果で、ばらつきは見ていない。

## ファイル

- `micro24.sh`: 計測の台本。
- `micro24_an.py`・`micro24_an.txt`: 段ごとの 1 件の処理の区間、続けて届く間隔、取得の空き待ちと受け取り待ち（出ている取得の数ごと）。`evidence/05/fetchsplit.py` をそのまま使う。
- `micro24_cap.txt`: `evidence/04/capcheck.py t2k4 t2c32k4 s3nc32` の出力（p = 0.4 の取得の時間切れを、リンクの状態で分けたもの）。
- `micro24_summary.csv`: 5 つの段の evalagg の `summary.csv` をまとめたもの。

## 作り直し方

```bash
ST=/home/lab_shared/semindex-eval; D=docs/wayfinder/semindex-stage2/evidence
.venv/bin/python3 $D/06/micro24_an.py s3fk8:0 s3nk8:0 s3fmix:0 s3nmix:0 t2c32k4:0.4 s3nc32:0.4
.venv/bin/python3 $D/04/capcheck.py --store $ST t2k4 t2c32k4 s3nc32
for s in s3fk8 s3nk8 s3fmix s3nmix s3nc32; do
  MPLCONFIGDIR=$(mktemp -d) PYTHONPATH=src .venv/bin/python3 -m semindex.evalagg --store $ST --stage $s --out <出力先>/$s
done
```
