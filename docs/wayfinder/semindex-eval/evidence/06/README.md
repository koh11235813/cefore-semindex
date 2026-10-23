# チケット 06 の証跡（段 1 の本番 `s1`）

直線 5 ノードで、切断率 p = 0・0.1・0.2・0.3・0.4 の 5 条件を、条件ごとに seed 1〜100 で回した 500 回の集計。

## 回したもの

- 回した期間: 2026-10-23 10:25 ごろ〜2026-10-24 08:30 ごろ（JST）。1 回はどの p でもおよそ 158 秒。
- コード: 500 回すべての `meta.json` が、semindex `7321c71`・ReCefore `be78de6` で、どちらも dirty でないことを記録している。
- 保存先: `/home/lab_shared/semindex-eval/s1/p<切断率>/<seed>/`（全体で 244 MB）。run ごとの生データはそちらにあり、このディレクトリには入れていない。

回したコマンド（オーケストレータが root で実行）:

```
sudo env PYTHONPATH=<feat/eval-stage1 の checkout>/src \
  /home/lab_shared/cefore-semindex/.venv/bin/python3 -m semindex.evalrun \
  --stage s1 --p 0 0.1 0.2 0.3 0.4 --seeds 1-100 \
  --store /home/lab_shared/semindex-eval --recefore /home/lab_shared/cefore-emu-sslab
```

## 集計のコマンド

main `75211a4`（図を描く `semindex.evalfig` を含む）で、一般の利用者として実行した。

```
MPLCONFIGDIR=<書き込める一時ディレクトリ> \
  .venv/bin/python3 -m semindex.evalagg --store /home/lab_shared/semindex-eval --stage s1 --out <出力先>
```

`MPLCONFIGDIR` は、`~/.config/matplotlib` に書き込めない環境で matplotlib が標準エラーに出す警告を止めるためだけのもので、出力の中身には関係しない。

## このディレクトリの内容

- `summary.csv`: 条件ごとの指標（run の判定の数、発見までの時間と全ノード一致の時間の KM 中央値と四分位、打ち切りの数、レコード 1 件あたりの制御の通信量、probe の発見率と Wilson 区間、probe の正解による分類）。
- `runs.csv`: run ごとの 1 行（判定、t0 の出し方、通信量など）。
- `survival.csv`: 発見の Kaplan–Meier の曲線（条件 × ホップ数）。
- `decisions.csv`: `decision` 行の種類ごとの件数と、状態と行動の欄の分布。
- `fig_discovery_survival`・`fig_consistency`・`fig_probe_established`・`fig_control_bytes`（PNG と PDF）。

集計はほかに `discovery.csv`（組ごと、3.6 MB）・`consistency.csv`（レコードごと）・`probes.csv`（probe ごと）も出すが、大きいのと上のコマンドで作り直せるので、ここには入れていない。

## 有効な run の数

| p | valid | flagged | unknown | incomplete |
|---|---|---|---|---|
| 0 | 100 | 0 | 0 | 0 |
| 0.1 | 100 | 0 | 0 | 0 |
| 0.2 | 100 | 0 | 0 | 0 |
| 0.3 | 100 | 0 | 0 | 0 |
| 0.4 | 100 | 0 | 0 | 0 |

印つきの run は各条件 0%（受け入れの上限は 5%）。probe の正解は 500 回すべてで組み立てられ、分類できなかった probe は 0 本。
