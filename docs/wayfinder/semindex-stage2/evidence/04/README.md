# 段 2 の試走（チケット 04）

間隔の選択肢の値と k を決めるために回した短い試走の集計（2026-10-26）。決めた値と根拠は map の「行動と確率」「負荷」「選択肢の間隔のゆらぎ」、試走の経過はチケット 04 の Resolution。

## 回した段

どれも直線 5 ノード、seed 1〜3。p は段ごとに下のとおり。84 回すべて有効（印つき・判定なし・未完了は 0 本）で、ReCefore はすべて `be78de6`（dirty でない）。

| 段 | 方針 | k | p | semindex | 回し方 |
|---|---|---|---|---|---|
| `t2i0p5`・`t2i1`・`t2i2`・`t2i4`・`t2i8` | 間隔を 0.5・1・2・4・8 秒に固定、ゆらぎ無し | 1 | 0、0.4 | `51716ab` | `trial04.sh` |
| `t2k1`・`t2k2`・`t2k4`・`t2k8` | 既定（2 秒 ± 20%） | 1・2・4・8 | 0.4 | `51716ab` | `trial04.sh` |
| `t2mix` | {0.5, 1, 2, 4, 8} を一様、ゆらぎ無し | 1 | 0、0.4 | `51716ab` | `trial04.sh` |
| `t2j0p5`・`t2j1p359`・`t2j3p695`・`t2j10p043` | 0.5・1.359・3.695・10.043 秒に固定、j = 0.1 | 1 | 0、0.4 | `1efd237` | `trial04b.sh` |
| `t2jmix` | {0.5, 1.359, 3.695, 10.043} を一様、j = 0.1 | 1 | 0、0.4 | `1efd237` | `trial04b.sh` |
| `t2c32k4`・`t2c32k8` | 既定、隣ごとの同時取得の上限 32 | 4・8 | 0.4 | `99fb5fe` | `trial04c.sh` |

- semindex の版は各 run の `meta.json` の `semindex_git_commit`（どれも dirty でない）。3 つの回し手は、それぞれ別の worktree の `src` を `PYTHONPATH` にして回した。
- `meta.json` に `max_in_flight_per_neighbor` があるのは `t2c32*` だけ（値は 32）。ほかの段はこの欄ができる前の版で回したので欄が無く、エージェントの既定の 8 で動いている。`capcheck.py` は、欄が無いか null なら 8 と読む。
- `t2i*`・`t2mix` は、選択肢の方針にゆらぎを入れる前の版で回した（`state_poll_choice_jitter_frac` の欄が無い）。ゆらぎの無い選択肢では状態確認の時刻がそろい続けるので、この 2 つの段の遅れは大きめに出ている。間隔ごとの遅れの基準は `t2j*`。
- 比べる基準の `s1sub` は、段 1 の本番 `s1` の p = 0 と 0.4 の seed 1〜3（6 回）。`trial04.py` が出力先の `s1store/` に、ストアの run への symlink を作って集計する。

## ファイル

- `summary.md`: `trial04.py` が書いた表の全体（英語）。表 1〜4 と定義。
- `interval_table.csv`: 表 1 のもと。(段, p) ごとの run の数、発見までの時間と全ノード一致の Kaplan–Meier 中央値、打ち切りの組の数、1 ホストあたりの遅れの内訳、制御の送信量／レコード、状態確認の数と `last_poll_status` の頻度。
- `k_table.csv`: 表 2 のもと。(段, p) ごとの `fetch_plan` の数、1 度に取る件数の分布（中央値・四分位・最大・2 件以上の割合・度数）、run 1 回の保存量。
- `sync_table.csv`: 表 3 のもと。状態確認の時刻のそろい方を、窓 δ = 10 ms と 50 ms で（`summary.md` は 50 ms だけ）。
- `choice_table.csv`: 表 4 のもと。選ばれた間隔の度数と一様からのずれ（χ²）、`action_prob` の値ごとの数。
- `run_info.csv`: run ごとの保存量（`du` と同じ数え方）、`events.jsonl` の合計、窓の中の判断の数。
- `trial04.py`・`breakdown2.py`: 集計のスクリプト。`breakdown2.py` は段 1 の `evidence/s1-breakdown/breakdown.py` を段 2 の名前と `meta.json` に合わせて直したもので、遅れの分け方は同じ（何を変えたかは docstring）。
- `capcheck.py`: 隣ごとの同時取得の上限の確かめ（下の「上限の確かめ」）。
- `trial04.sh`・`trial04b.sh`・`trial04c.sh`: 試走の回し手。回したときのまま置いている。`trial04c.sh` の最初の数行は、ひとつ前の `trial04b.sh` のログに終わりの印が出るまで待つためのもので、ログの道のりはそのときのセッションの作業ディレクトリ。

`deliveries.csv`（4.6 MB）、`bytes.csv`、`sync_pairs.csv`、`agg/`（evalagg の出力、12 MB）、`s1store/` は置いていない。下のコマンドで作り直せる。

## 定義（`summary.md` の Definitions の要約）

- 集める run は、evalagg が有効とした run（`runs.csv` の `valid=valid`）だけ。`DONE` の無い run は未完了として数える。
- 発見までの時間と全ノード一致の中央値は、evalagg の `summary.csv` の Kaplan–Meier の値。`disc mean*` は打ち切られていない組だけの単純平均で、打ち切りがあると小さく出る。
- 1 ホストあたりの遅れの内訳は段 1 と同じ分け方。(レコード, ノード) の組ごとに、上流が保存した時刻 U、そのレコードを初めて載せた返事のもとになった状態確認の時刻 P、このノードが保存した時刻 S を取る。[U, P] のうちリンクが落ちていた時間が「link-down」、つながっていた間の待ちが「poll wait」（そのうち最後にリンクが戻ってから P までが「after-up」）、残りが「other」（状態確認の往復と取得）。`state_rx` には状態確認の epoch が無いので、P は返事の直前の状態確認とした。状態確認がいくつも返事待ちのときは、poll wait を大きめに読むことがある（`poll rtt` が ms の桁なら取り違えは無い）。
- `B/rec med` は evalagg の制御の送信量／レコード（窓の中）。`B/rec norm` は、Interest と Data の名前のグループ名を 15 文字にそろえ直した送信量を、窓の中のレコードの数で割った run ごとの平均。段によってグループ名の長さが違い、その差が状態確認の回数に比例して効くため。
- `fetched` は、窓の中の `fetch_plan` 1 回で取りに行く件数（`action.fetched` の長さ）。
- 状態確認のそろい方: ある送り手からある相手への窓の中の状態確認の時刻を 1 本の流れとし、同じホルダーを確かめる 2 つの隣の流れ（`same_holder`）と、1 本のリンクの両向きの流れ（`link_reverse`）を比べる。`coinc` は X の状態確認のうち ±δ 以内に Y の状態確認がある割合。`chance(shift)` は Y を [0, W) の一様な量だけ巡回的にずらして（200 回、seed つき）同じものを数えた値で、Y 自身の周期は残る。`excess` = `coinc` / `chance(shift)`（1 で偶然と同じ、1 よりずっと大きければそろっている）。`persist` は、この状態確認が重なったときに次も重なる割合。`offset drift` は、X の状態確認から見たいちばん近い Y の状態確認までの符号つきのずれが、続く 2 回でどれだけ変わるかの中央値で、0 に近いと位相が固定されている（ずれが δ より大きくても）。処理の遅れがたまって絶対の位相は動くので、そろい続けているかは、相手との相対のずれ（`offset drift` と `persist`）で見る。
- 選択肢の確かめは、run 全体の `state_poll` の判断で、`interval_s` ごとの度数を一様と比べる（χ² の自由度は選択肢の数 − 1。5% の臨界値は自由度 2 で 5.99、4 で 9.49）。

## 上限の確かめ（`capcheck.py`、p = 0.4、seed 1〜3 の合計、run 全体）

| 段 | 上限 | `fetch_plan` | 上限に届いた割合 | 計画の前の in_flight（Q1 / 中央値 / Q3） | 取得のタイムアウト | そのうち直前 2 秒ずっとリンクがつながっていた | リンクが落ちていた | 送ってからタイムアウトまで（最小〜最大、ms） | 保存したレコード | h1〜h3 で続く `record_stored` の間隔の中央値（300 ms 未満だけ） |
|---|---|---|---|---|---|---|---|---|---|---|
| `t2k4` | 8 | 1775 | 0.903 | 7 / 7 / 7 | 205 | 24 | 181 | 2003〜2101 | 2874 | 50 ms |
| `t2c32k4` | 32 | 604 | 0.642 | 0 / 31 / 31 | 159 | 93 | 66 | 2001〜2458 | 2880 | 34 ms |
| `t2k8` | 8 | 4355 | 0.962 | 7 / 7 / 7 | 1233 | 82 | 1151 | 2000〜2876 | 5152 | 75 ms |
| `t2c32k8` | 32 | 2056 | 0.832 | 30 / 31 / 31 | 2916 | 1343 | 1573 | 2000〜3579 | 5583 | 58 ms |

- 上限に届いた計画は、`state.in_flight` + 取りに行く件数 ≥ 上限のもの。
- リンクが落ちていたかは、`results.json` の `link_down`／`link_up` から作った区間（`breakdown2.link_intervals`）が、タイムアウトの前の 2 秒（取得の Interest の lifetime）と少しでも重なるか。
- 送ってからタイムアウトまでは、その (origin, seq) をその隣に取りに行った `fetch_plan` のうち、タイムアウトより前でいちばん新しいものから測った。4 段 4513 件のうち 2.75 秒を超えたのは 13 件（`t2k8` で 3 件、`t2c32k8` で 10 件）だけで、99 パーセンタイルは 2.1〜2.6 秒。
- 同じスクリプトを `t2k1 t2k2` で回すと、取得のタイムアウトは 0 回、`record_stored` の間隔の中央値はどちらも 34 ms。
- `fetch_backoff` の待ちは 2・4・8・16 秒（4 段の合計で 2679・1009・446・379 回）。

## 作り直し方

`trial04.py` は `--out` に `deliveries.csv`（4.6 MB）と evalagg の出力（12 MB）、`s1store/`（ストアへの symlink）を書くので、出力先はリポジトリの外にする。ストアは読むだけ。

```
cd <worktree>/docs/wayfinder/semindex-stage2/evidence/04
PYTHONPATH=<worktree>/src MPLCONFIGDIR=<scratch>/mpl \
  /home/lab_shared/cefore-semindex/.venv/bin/python3 trial04.py \
  --store /home/lab_shared/semindex-eval --out <scratch>/out
PYTHONPATH=<worktree>/src /home/lab_shared/cefore-semindex/.venv/bin/python3 capcheck.py \
  --store /home/lab_shared/semindex-eval t2k4 t2c32k4 t2k8 t2c32k8
```

- `<worktree>` は `max_in_flight_per_neighbor` を集計できる版（`99fb5fe` 以降）の checkout。`PYTHONPATH` を付けないと、venv に editable で入っている main の semindex が使われる。
- `trial04.py` の段は、既定でストアの `t2*` のディレクトリ全部（`--stages t2k4,t2k8` のように絞れる）。段 1 の基準 `s1sub` は `--s1-seeds 1,2,3 --s1-ps 0,0.4`（既定）で足す。evalagg を回し直さずに表だけ作り直すなら `--skip-evalagg`。
- このディレクトリの表は、root で `trial04.py` を回した出力（`out2/`）からコピーした。`capcheck.py` の表は、ストアの run が利用者から読めたので root なしで回した。
