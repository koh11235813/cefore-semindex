# 失敗に反応する二つの方針の実測と、段 A・B・C の取り直し（`s5*`）

状態確認の決め直し（`state_poll_policy`、`evidence/08/`）を入れたコードで段 A・B・C を回し直し、同じコードで、失敗に反応する二つの方針（短縮と 5 秒待ち）を回した（2026-11-03〜11-05）。方針の 4 ホップの遅れ、通信量、状態確認の数を固定の腕と比べ、方針を選んだ replay の予測との差と、全ノードの再試行が同じ時刻にそろうかを数えた。

## 回したもの

`main4.sh` を root で回した（2026-11-03 15:21〜11-05 16:55）。semindex は 7d5cd80（決め直しを入れた変更のマージ）を誰も触らない写し（detached）に置き、ReCefore は 25ad4e9。1120 本すべて valid で、印つきの run は無い。meta.json は 1120 個すべて `semindex_git_dirty` と `recefore_git_dirty` が false、`store_synchronous` は NORMAL。

| 段 | 方針 | p | seed |
|---|---|---|---|
| `s5c` | 段 1 の方針（2 秒 ± 20%） | 0〜0.4 の 5 つ | 1〜20 |
| `s5b0p5`・`s5b1p359`・`s5b3p695`・`s5b10p043` | 選択肢を 1 つに固定（ゆらぎ 0.1） | 0〜0.4 の 5 つ | 1〜20 |
| `s5a` | 4 つの選択肢を一様に選ぶ（ゆらぎ 0.1） | 0〜0.4 の 5 つ | 1〜100 |
| `s5short` | 短縮: 答えの後は 10.043 秒、失敗の後は 3.695 秒（どちらもゆらぎ 0.1） | 0・0.2・0.4 | 1〜20 |
| `s5wait` | 5 秒待ち: 答えの後は 10.043 秒（ゆらぎ 0.1）、失敗の後はちょうど 5 秒 | 0・0.2・0.4 | 1〜20 |

- どの段も k = 4、隣ごとの同時取得の上限は既定の 8。段 A・B・C の設定は `evidence/07/` の `s4*`（`main3.sh`）と同じ。
- `s5short` と `s5wait` の `state_poll_policy` は、それぞれ `{"intervals_s": [10.043, 3.695], "jitter_frac": [0.1, 0.1]}` と `{"intervals_s": [10.043, 5.0], "jitter_frac": [0.1, 0.0]}`。失敗の後とは、続けて返事の来なかった回数が 1 以上のとき（map の「返事と期限切れでの決め直し」）。
- 決め直しは、最新の状態確認に返事が来たか lifetime が過ぎたときに、その状態確認を送った時刻から次の時刻を決め直す。replay の `ideal` の形に当たる（`replay.py` の docstring）。送るときに間隔を決める古い形（replay の `engine`、`policy_table.csv` の `shorten-engine`）ではない。
- 二つの方針は、`s4a` の 100 本の切断の予定の上で方針を比べた replay（このディレクトリの `replay.py`・`policies.py`・`policy_table.csv`、出どころは下の「replay の写し」）から選んだもの。ADR 0004 の、A の判断の部品から続く判断を組み立てるモデルではない（そのモデルはまだ作っていない）。
- 切断の予定は (p, seed) だけで決まる。p 0.2・0.4 × seed 1・7 で、`s5short`・`s5wait`・`s5b10p043`・`s5b0p5`・`s5c`・`s4a` の `*.experiment.json` が一致した。同じ seed の段どうしは、同じ切断の上で比べている。

## 作り直し方

1. 集計（evalagg）。出力はおよそ 47 MB あり、run から作り直せるので、ここには置いていない。

   ```bash
   W=/home/lab_shared/cefore-semindex        # このリポジトリ（7d5cd80 以降）
   AGG=<作業用のディレクトリ>/main4-agg
   for st in s5c s5b0p5 s5b1p359 s5b3p695 s5b10p043 s5a s5short s5wait; do
     PYTHONPATH=$W/src $W/.venv/bin/python3 -m semindex.evalagg \
       --store /home/lab_shared/semindex-eval --stage $st --out $AGG/$st
   done
   ```

2. 解析。このディレクトリで回す。`evidence/05/` の `attrib.py`・`run_all.py`、`evidence/04/` の `trial04.py`・`breakdown2.py`、`evidence/07/` の `seq_vs_B.csv`・`per_hop_B.csv`・`concat_summary.py` を、中身を変えずに使う。`replay.py` はストア `/home/lab_shared/semindex-eval` の `s4a` と `s5*` の切断の予定を読む。

   ```bash
   cd $W/docs/wayfinder/semindex-stage2/evidence/09
   export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=<作業用のディレクトリ>/mpl AGG
   PY=$W/.venv/bin/python3
   PYTHONPATH=$W/src:../05:../07 $PY perrun.py              # perrun.csv
   $PY replay_runs.py                                        # replay_perseed.csv
   $PY ana.py > /dev/null                                    # live_main.csv ほか 4 つと tables.md（標準出力は tables.md と同じ）
   $PY sens4.py > sens4.md
   PYTHONPATH=$W/src $PY handcheck.py > handcheck.txt
   PYTHONPATH=$W/src $PY phase.py --agg $AGG --out . --workers 3   # sync_table.csv ほか 3 つと retry_pairs.csv
   gzip -n9 retry_pairs.csv                                  # 再試行 1 本ごとの行（330 KB）を retry_pairs.csv.gz に
   PYTHONPATH=$W/src $PY retry_recount.py > retry_recount.txt
   PYTHONPATH=$W/src $PY retry_offsets.py > retry_offsets.txt
   $PY recheck.py > recheck.txt
   $PY ../07/concat_summary.py $AGG s5c s5b0p5 s5b1p359 s5b3p695 s5b10p043 s5a s5short s5wait > main4_summary.csv
   ```

   全部で 1 分ほど。`perrun.py`・`replay_runs.py`・`phase.py` は 3 プロセスで回す。

2026-11-09 にこのディレクトリの写しで 2. を回し直し、解析を最初に回した作業用のディレクトリの出力と、CSV 10 個（`perrun.csv`、`replay_perseed.csv`、`live_main.csv`、`calibration_s5b_vs_s4b.csv`、`live_vs_replay.csv`、`front4.csv`、`sync_table.csv`、`retry_table.csv`、`retry_phase_hist.csv`、`poll_phase_hist.csv`）と `tables.md`・`sens4.md`・`handcheck.txt` の 13 個がバイト単位で同じことを確かめた。`retry_pairs.csv` も、作業用のディレクトリで最初に書いたものと同じだった。`retry_recount.txt`・`retry_offsets.txt`・`main4_summary.csv`・`recheck.txt` は、このとき初めてファイルに書いた。台本で変えたのは、作業用のディレクトリを指していた道のり（`AGG` と `STORE` を環境変数に、`evidence/04/`・`evidence/07/`・`replay.py` をこのリポジトリの中に）と、使っていなかった import、名前を変えた 2 本（元は `mycheck.py` と `offs.py`）の import だけ。出力に効く行は変えていない。

## 結果

### 1. 主な数字（実測、seed 20 本）

遅れは、4 ホップ先（h0→h4 と h4→h0）の打ち切られていない組の平均で、組をまとめて平均した値（`evidence/07/` の `seq_model.direct`。遅れは evalagg の `discovery.csv` の値）。± は seed を単位にした bootstrap の SE。バイトは、レコード・ホップあたりの状態確認の Interest と返事のバイトで、状態の名前の長さを `s4a` の平均のグループ名（14.92 文字、replay の `INT_B` = 50.9 B の基準）にそろえてある。

| 段 | p | 4 ホップの遅れ ms | 打ち切り | B/レコード・ホップ | 状態確認/run | 無駄 |
|---|---|---|---|---|---|---|
| s5short | 0 | 20379 ± 661 | 0 % | 17.9 | 96.0 | −0.001 |
| s5short | 0.2 | 24693 ± 615 | 0 % | 17.5 | 112.4 | 0.226 |
| s5short | 0.4 | 29757 ± 938 | 1.08 % | 16.8 | 137.7 | 0.474 |
| s5wait | 0 | 20742 ± 661 | 0 % | 17.9 | 96.1 | 0.000 |
| s5wait | 0.2 | 24847 ± 598 | 0 % | 17.1 | 106.7 | 0.191 |
| s5wait | 0.4 | 31561 ± 1389 | 0.92 % | 15.9 | 121.4 | 0.406 |
| s5b10p043 | 0 | 20476 ± 677 | 0 % | 17.9 | 96.0 | 0.000 |
| s5b10p043 | 0.2 | 29643 ± 934 | 0.08 % | 15.2 | 96.0 | 0.183 |
| s5b10p043 | 0.4 | 42872 ± 2298 | 6.58 % | 12.7 | 96.0 | 0.397 |
| s5b3p695 | 0.2 | 12053 ± 322 | 0 % | 42.3 | 256.1 | 0.193 |
| s5b3p695 | 0.4 | 19702 ± 1067 | 0.17 % | 34.2 | 256.1 | 0.406 |

- 状態確認/run は、窓 [t0 + 10 秒, t0 + 130 秒] の中で 8 本の向きに送った状態確認の数（evalagg の `polls.csv` の `polls_sent`）。無駄は返事の来なかった割合（1 − 返事の数 / 状態確認の数、seed をまとめて計算）。s5short p 0 の −0.001 は、seed 9 の 1 run で窓の中の返事が状態確認より 1 本多かったため（窓の直前に送った状態確認の返事が窓の中に入ったと見ている）。
- 決め直しを「いま送る」に丸めた数と、遅れて届いた返事の数は、全段で 0。
- 全部の段・p・腕の表は `tables.md` と `live_main.csv`。段ごとの evalagg の要約は `main4_summary.csv`（`evidence/07/main3_summary.csv` と同じ形）。

### 2. 固定 10.043 秒の腕に比べた遅れ

| p | 実測 短縮 | 実測 5 秒待ち | replay（同じ 20 seed）短縮 | replay 公表値（100 seed）短縮 / 5 秒待ち | replay 公表値 古い形の短縮 |
|---|---|---|---|---|---|
| 0.2 | −16.7 % | −16.2 % | −19.2 % | −20.6 % / −20.5 % | −9.2 % |
| 0.4 | −30.6 % | −26.4 % | −27.0 % | −28.9 % / −27.5 % | −19.4 % |

- 実測は 1. の値（組をまとめた平均）から計算した。p 0.4 を seed ごとの平均で計算すると、短縮 −32.7 %、5 秒待ち −28.8 %（p 0.2 は変わらない）。
- p 0.2 の短縮では、同じ 20 seed の replay が予測した得（5614 ms）のうち 88 %（4950 ms）が実測で出た。古い形（送るときに決める）の公表値の得（2774 ms）の 1.8 倍。
- p 0.4 では、固定 10.043 秒の 4 ホップの組の 6.58 % が打ち切られている（方針は 1 % ほど）。打ち切られるのは遅い組なので、固定の側の遅れは速めに出ていて、方針の得は小さめに見えている。

### 3. replay の予測との差（遅れ）

| 段 | p | 実測 | 同じ切断の予定の replay（差、z） | 公表値（差、z） |
|---|---|---|---|---|
| s5short | 0.2 | 24693 | 23646（+4.4 %、z 1.7） | 23973（+3.0 %、z 1.0） |
| s5short | 0.4 | 29787 | 30838（−3.4 %、z −1.9） | 31062（−4.1 %、z −1.2） |
| s5wait | 0.2 | 24847 | 24273（+2.4 %、z 1.1） | 24007（+3.5 %、z 1.4） |
| s5wait | 0.4 | 31534 | 31211（+1.0 %、z 0.4） | 31657（−0.4 %） |

- p 0.4 の実測は、replay と同じく seed ごとの平均。1. の値（組をまとめた平均）とは 30 ms ほどずれる。
- 「同じ切断の予定の replay」は、実測と同じ (p, seed) の切断とレコードの予定で `replay.py` を回した値（seed 1〜20、ゆらぎの引き直し 10 回）。「公表値」は `policy_table.csv`（`s4a` の seed 1〜100）。`replay_runs.py` で公表値と同じ設定を回し直すと、比べた方針のすべてで `policy_table.csv` の遅れ・バイト・無駄・打ち切りと同じ値になった（`tables.md` の「Replay: published ... vs our re-run」）。
- p 0.2・0.4 では、方針のバイト（名前の長さをそろえた値）は replay と ±1.2 % で合い、状態確認/run は実測のほうが 1.0〜1.6 % 多い。p 0 では、バイトは実測が replay より 1.6〜1.7 % 少なく、状態確認/run は 0.8〜1.0 % 多い。固定の腕（10.043 秒と 3.695 秒）の replay と実測は |z| ≤ 1.4。表は `live_vs_replay.csv`。

### 4. 固定の腕の線との差（[4] と [c]）

| 方針 | p | 実測 [4] | replay [4]、同じ 20 seed | replay [4]、公表値 | replay [c]、公表値 |
|---|---|---|---|---|---|
| s5short | 0.2 | −3481（95 % CI −4538〜−2423） | −4288 | −4691 | −3169 |
| s5wait | 0.2 | −3588（−5317〜−1953） | −3902 | −4946 | −3711 |
| s5short | 0.4 | −9851（−15309〜−5959） | −7780 | −8546 | −4919 |
| s5wait | 0.4 | −9159（−14414〜−4857） | −8378 | −8975 | −6265 |

- [4] は、固定の 4 つの腕の（バイト, 遅れ）の下側の凸包を方針と同じバイトで読んだ遅れと、方針の遅れの差（ms、負なら方針が線より速い）。seed ごとの平均で数え、CI は seed の bootstrap 400 回。replay の `frontier.py` と同じ数え方。[c] は、replay だけにある 11 点の固定の間隔（0.5〜15 秒）の線との差。
- 実測で出せるのは [4] だけで、[c] に当たる実測は無い。実測の −3481 ms と並べるのは replay の [4]（−4691 ms、同じ 20 seed では −4288 ms）。
- 公表値の [4] との差 1.2 秒のうち、およそ 0.4 秒は seed を 1〜20 に絞ったことによる。残りのおよそ 0.8 秒が実機と replay のモデルの差。
- p 0.4 の [4] は、seed ごとに平均するか組をまとめるかで 1.2 秒ほど動く（s5short −9851 → −8661）。固定 10.043 秒の打ち切りが seed ごとに偏るため。表は `front4.csv`。

### 5. 数え方で結論が変わるところ

数え方ごとの [4] は `sens4.md`。

1. **p 0 のバイト。** 名前の長さをそろえないと、方針（18.6 B）が固定 10.043 秒（19.0 B、段の名前が 2 文字長い）より左に出て凸包の外になり、[4] が決まらない。そろえると [4] は −92 / +275 ms で、p 0 では方針と固定 10.043 秒は同じ（失敗が無ければ同じ動きなので、そうなるはず）。
2. **返事のバイトは腕によって違う。** 名前の長さをそろえた返事 1 本のバイトは、p 0 で 0.5 秒の腕が 138.4 B、10.043 秒の腕と二つの方針が 127.1〜127.3 B で、replay は p ごとに一定の `REP_B`（p 0 で 132.5 B）を使う（`recheck.txt`）。遅い腕ほど返事に載る状態が小さいためと見ている（推定）。同じ切断の予定の replay に比べた実測のバイト（`live_vs_replay.csv` の `rel`）は、0.5 秒の腕で +3.3〜+7.6 %、1.359 秒で +3.0〜+5.4 %、3.695 秒で +1.3〜+2.1 %、10.043 秒で −2.1〜−3.5 %、二つの方針で p 0 が −1.6〜−1.7 %、p 0.2 が −0.5〜−0.8 %、p 0.4 が +1.1〜+1.2 %。それでも [4] は、バイトの数え方（そのまま、名前の長さをそろえた値、`s4a` の定数）を替えても、名前の長さをそろえた値からの差が p 0.2 で 360 ms 以内、p 0.4 で 630 ms 以内に収まり、符号は変わらない（`sens4.md`）。
3. **状態確認の数。** 実測は replay より 1.0〜1.6 % 多い（p 0.4 の 0.5 秒の腕では 1 run あたり 1746 回と 1702 回）。replay は 1 回ごとのループの遅れを 0.05 + 0.035p 秒と置くが、実測では p が上がってもループの遅れが延びていないと見ている（推定）。
4. **無駄の割合。** seed をまとめて計算するか run ごとに平均するか（replay は run ごと）で 0.002〜0.003 違う。

### 6. 位相（全ノードの再試行がそろうか）

詳しくは `phase.md`。尺度は map の「選択肢の間隔のゆらぎ」と同じ（`evidence/04/` の `trial04.py`）。

- どの方針・p でもそろわなかった。重なりは偶然の 0.9〜2.4 倍、相手とのずれの変わり方は 1 回あたり 410〜4113 ms で、ゆらぎのある `t2j*`（0.65〜2.0 倍、41〜1406 ms）と同じ帯にあり、ゆらぎの無い `t2i*`（2.8〜100 倍、1〜35 ms で位相が固定）とは違う。再試行が 100 ms の幅に 3 本以上入ったのは、全 run で 1 か所だけ（s5short p 0.4）。
- 5 秒待ちは、停止のあいだ 5 秒の格子の上の位相を持ち越す（続く再試行の位相の差が ±50 ms に収まる割合 0.86〜0.88）。同じリンクの両端が重なったまま次の状態確認に進む割合は、p 0.4 で 0.32 と、map の「ゆらぎあり」の帯（0〜0.27）を少し超えた。ただし重なりは 2428 回のうち 66 回と少ない。再試行だけで見ると、重なったまま次の再試行に進んだのは 14 回のうち 6 回で、そのうち 4 回は seed 14 の 1 run（`retry_pairs.csv.gz` の `rev_pers_num`・`rev_pers_den`）。
- 位相は 5 秒の区切りの 0〜1.5 秒に寄る。全エージェントが t0 + 515 ms ごろに一斉に最初の状態確認を出すことと、10.043 秒がほぼ 5 秒の 2 倍なことによると見ている（推定）。固定 10.043 秒の段にも同じ寄りがあり、方針によるものではない。
- リンクが実際に戻った時刻（results.json）から 0.5 秒以内に送った再試行は 215 回で、失敗は 0 回。replay はこの窓の状態確認が 2.7 % 失敗すると置くが、実測では出なかった。名目の link_up（t0 + `at`）から数えると 4 回が失敗しているが、どれもリンクが実際に戻る（約 9.6 ms 後）より前に送ったもの（`retry_offsets.txt`）。

### 7. `s5b*` と `s4b*`（`evidence/07/`）の比べ

`calibration_s5b_vs_s4b.csv`。間には、h2 のプローブの本体を止まらずに取る変更（ADR 0004 の Consequences）と、決め直しの変更が入っている。決め直しは段 A・B・C の送信を変えない（`evidence/08/`）。

- 4 ホップの遅れは、4 つの腕 × 5 つの p の 20 か所すべてで |z| ≤ 1.06。
- バイト（レコード・ホップあたりの状態確認のバイト、ホップの平均。段の名前は同じ長さ）は、0.5 秒と 1.359 秒の腕で 10 か所すべて `s5b*` のほうが多く、p とともに +0.1 % から +2.7 % まで増える（0.5 秒・p 0.4 で 240.8 → 247.4 B）。1 か所ずつでは、差は SE の 0.7〜2.1 倍ではっきりしない（`s4b*` の SE はここに無いので、`s5b*` と同じと置いた見積もり）。それでも 10 か所すべてで向きがそろい、p とともに増えるので、偶然の揺れとは見ていない。3.695 秒と 10.043 秒の腕は ±0.7 % 以内。
- 推定: プローブの本体の取得で h2 のループが止まらなくなり、短い腕で遅れて出ていた状態確認が予定どおり出て、窓の中の状態確認が増えたぶん（確かめていない）。
- だから `s4b*` の遅れは `s5b*` と同じに扱えるが、短い腕のバイトは最大 2.7 % 違う。

## 測ったもの、推したもの

- 測った: 遅れ、打ち切り、バイト（名前の長さの差し引きは決まった計算）、状態確認と返事の数、決め直しの数、実測の [4]、切断の予定の一致、位相の尺度。
- replay のモデルの値: 公表値、同じ切断の予定での予測、replay の [4] と [c]。
- 推した: ループの遅れが p で延びないこと、返事のバイトが腕で違う理由、[4] の差のうち seed の標本による部分、位相が寄る原因、`s5b*` と `s4b*` のバイトの差の原因。

## 確かめたこと

- 解析の台本は、指標（`perrun.py`〜`sens4.py`）と位相（`phase.py`）の 2 つに分けて別々に書き、書いたのとは別の場所で回し直して出力が同じことを確かめた。上の「作り直し方」の回し直しも同じ。
- 4 ホップの遅れと打ち切りの割合を、`attrib.py` を通さずに evalagg の `discovery.csv` から直接数え直し（`recheck.py`、`recheck.txt`）、`live_main.csv` の 31 か所（固定の 4 つの腕と `s5c` は p 5 つずつ、二つの方針は p 3 つずつ）すべてで一致した。
- 名前のバイトを生のイベントから手で作り直した（`handcheck.py`、`handcheck.txt`）。s5short p 0.2 seed 1 の窓の中の状態確認 116 本の名前はどれも 54 B で、`polls.csv` と合う。
- [4] を 1. の値から手で内挿して、p 0.2 で −3457 / −3563 ms（seed ごとの平均の値は −3481 / −3588 ms）。
- 位相の再試行の数、復帰直後の割合、最初の状態確認の時刻を、`phase.py` とは別の台本（`retry_recount.py`、`retry_offsets.py`）で数え直した。違いは数え方の違いで説明がつく（`phase.md` の「別の台本での数え直し」）。
- 同じ切断の予定の replay は `ideal` の形（`Table((10.043, 3.695))` の既定と `SlotBlind(10.043)`）で回している。

## replay の写し

`replay.py`・`policies.py`・`policy_table.csv` は、`/home/lab_shared/taf-grant-2026/analysis/value-check/`（git の管理の外）から、2026-11-09 に中身を変えずに写した。

| ファイル | 元の更新時刻 | sha256 |
|---|---|---|
| `replay.py` | 2026-11-02 17:16 | `4e5d610dd0ee8587c8a007197db3b74dc4ff9d0507a11d0884de46d65e418170` |
| `policies.py` | 2026-11-02 17:16 | `a379e41c0b3d1c391489c048861e5a4f26ab779a5799fc5a485d73f5d23c7fe6` |
| `policy_table.csv` | 2026-11-02 17:21 | `a20558e95f6eaebf472715b50ee05b92e9f22787bc8df88bd80f9ecb25f6afa0` |

- `replay.py`: 記録した切断の予定の上で方針を決定的に回すモデル（測ったものではない）。`replay_runs.py` と `ana.py` が import する。
- `policy_table.csv`: 公表値の表。`ana.py` が「公表値」の列に使う。
- `policies.py`: 公表値の方針の定義と回し方を読むためだけに置いた。同じディレクトリの `dp.py` を import し、`policy_table.csv` は別の `frontier.py` が `policies.py` の出力から書く。この 2 つは写していないので、`policies.py` はここでは動かない。

## ファイル

- `main4.sh`: 回した台本（root で回す）。
- `perrun.py` → `perrun.csv`: run ごとの 4 ホップの組、ホップごとのレコード・ホップとバイト、名前の長さ、`polls.csv` の数。
- `replay_runs.py` → `replay_perseed.csv`: replay の (方針, p, seed) ごとの値（同じ切断の予定と、公表値の設定の回し直し）。
- `ana.py` → `live_main.csv`（実測の表）、`calibration_s5b_vs_s4b.csv`（7.）、`live_vs_replay.csv`（3.）、`front4.csv`（4.）、`tables.md`（全部の表）。
- `sens4.py` → `sens4.md`: [4] の数え方ごとの値（5.）。
- `handcheck.py` → `handcheck.txt`: 名前のバイトの手計算。
- `phase.py` → `sync_table.csv`、`retry_table.csv`、`retry_phase_hist.csv`、`poll_phase_hist.csv`、`retry_pairs.csv.gz`（再試行の、run と送り手ごとの行）。`phase.md` はその読み方と結果。
- `retry_recount.py` → `retry_recount.txt`、`retry_offsets.py` → `retry_offsets.txt`: 位相の数え直し。
- `recheck.py` → `recheck.txt`: 4 ホップの遅れと打ち切りの数え直しと、返事 1 本のバイト。
- `main4_summary.csv`: 8 つの段の evalagg の `summary.csv` をまとめたもの（`evidence/07/concat_summary.py`）。
- `replay.py`、`policies.py`、`policy_table.csv`: replay の写し（上）。
