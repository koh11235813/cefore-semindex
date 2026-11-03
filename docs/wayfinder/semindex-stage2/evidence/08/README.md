# 次回の状態確認の決め直しを入れても、段 A・B・C のイベント列が変わらないことの確かめ

`state_poll_policy`（返事が来たとき／期限が切れたときに次回の状態確認の時刻を表から決め直す、ADR 0004 の Consequences）を足した変更で、そのキーを設定しない段 A・B・C の動きが main と同じかを、決まった偽のネットワークで回したイベント列をバイト単位で比べて確かめた（2026-11-03）。

## 回したもの

`identity.py`。3 ノードの直線（h0–h1–h2）を、時計を 0.1 秒刻みで 120 秒進めて回す。seed 5。各エンジンの乱数は build_agent と同じ作り方（`_poll_jitter_rng(node)`、seed があるときは `_poll_choice_rng(seed, node)` と `_poll_choice_jitter_rng(seed, node)`）。

| 段 | 設定 |
|---|---|
| A | 選択肢 0.5・1.359・3.695・10.043 秒、ゆらぎ 0.1 |
| B | 選択肢 3.695 秒だけ、ゆらぎ 0.1 |
| C | 段 1 の方針（2 秒 ± 20%） |

- h0–h1 のリンクは、ふだん返事が 10 ms で届く。30〜40 秒は状態確認の返事が 2.5 秒後に届く（lifetime 2 秒を過ぎた遅い返事）。50〜62 秒と 80〜100 秒は返事が来ない（時間切れと、lifetime + grace を超えて黙る区間）。
- h0 と h2 が 7 秒ごとにレコードを作るので、取得の計画と保存も入る。
- 出力は 1 行 1 JSON（キーは並べ替え）。送った Interest（名前と時刻）とエンジンのイベントを、起きた順に並べる。
- 偽のネットワークはスクリプトの中に持つ（`git archive` した `src` には `tests/fakes.py` が無いので、両側で同じ道具を使うため）。使っている API は main 3d89cfe にもあるものだけ。

main の側の 3 段で、状態確認の判断に出た直前の確認の状態は answered 400 回、timed_out 51 回、pending 25 回（残りは最初の 1 回の None）。

## 回し方

```bash
S=/tmp/claude-1001/-home-lab-shared-cefore-emu-sslab/4677c249-e96f-46cd-a827-e50302f7cbe5/scratchpad/redecide
W=/home/lab_shared/wt/semindex-redecide
PY=/home/lab_shared/cefore-semindex/.venv/bin/python3
mkdir -p $S/base && git -C $W archive 3d89cfe src | tar -x -C $S/base
cd $S
PYTHONPATH=$S/base/src $PY $W/docs/wayfinder/semindex-stage2/evidence/08/identity.py > identity.base.jsonl
PYTHONPATH=$W/src      $PY $W/docs/wayfinder/semindex-stage2/evidence/08/identity.py > identity.branch.jsonl
cmp identity.base.jsonl identity.branch.jsonl && echo IDENTICAL
```

## 結果

**同じ。** `cmp` は差を出さず、どちらも 2082 行、sha256 `fcdda5984a266e5b803396ccfd213051742662a919a98fd91948da0d2b721050`。変更後の側を 2 回回しても同じだった（スクリプト自体が決定的）。

言えるのは、この決まったシナリオで段 A・B・C の送信とイベントが main とバイト単位で同じということまで。Mininet 上では時刻が決定的でないので、同じイベント列にはならない。単体テストの側では、方針が無いときに `_PollReplies.snapshot` が送信 1 回につき 1 回しか呼ばれないこと、乱数を引く回数が変わらないことを `tests/test_engine.py` の `test_without_a_policy_nothing_new_runs` で固定している。

## 実験・集計の側: キーを渡さない段 A・B・C の生成ファイルと集計が変わらないこと

`state_poll_policy` を gen（実験記述のトップレベル）、evalgen／evalrun（`--poll-policy '<json>'`）、meta.json、evalagg（runs.csv の末尾 1 列と新しい polls.csv）に通した変更で、キーを渡さないときの出力が main 3d89cfe と同じかを確かめた（2026-11-03）。スクリプトはこのディレクトリの `gen_identity.sh` と `agg_identity.sh`。どちらも作業用のディレクトリを環境変数 `P` で受け取り、その下の `main/src` に置いた main のソースと、このリポジトリの `src` を比べる（2026-11-03 は scratchpad の `redecide/plumb` で回した）。

### 生成ファイル

段 A（`--interval-choices 0.5 1.359 3.695 10.043 --choice-jitter 0.1`）、B（`--interval-choices 0.5 --choice-jitter 0.1`）、C（選択肢なし）を `--p 0.2 --seed 1 --k 4` で evalgen にかけ、`experiments/t1.json`・`t2a.json`・`t2b.json`・`t2c.json` を gen にかけた。両側とも同じ `--out` に書いてから別の名前に移すので、YAML に入る `output_dir` と設定の置き場所の道のりは同じになる。

```bash
# リポジトリの根で
export P=<作業用のディレクトリ>
mkdir -p $P/main && git archive 3d89cfe src | tar -x -C $P/main
sh docs/wayfinder/semindex-stage2/evidence/08/gen_identity.sh   # 各段・各実験を両側で生成し、diff -r
```

**同じ。** A・B・C は 7 ファイル（experiment.json、YAML、h0〜h4.json）、t1・t2b・t2c は 4 ファイル、t2a は 3 ファイルで、`diff -r` は差を出さなかった。ただし YAML の 1 行だけは比べる前に置き換えている。`host_apps` の `env.PYTHONPATH` には gen.py の場所から決まる `SEMINDEX_SRC_DIR` が入り、main の側では `git archive` したコピーの `src` を指すので、それを worktree の `src` に `sed` で置き換えた（YAML 1 つにつき 1 か所）。JSON ファイルには手を加えていない。

### 集計

ストア `/home/lab_shared/semindex-eval` の段 `t6c`（p0 と p0.4 が 3 seed ずつ、6 run）を、読むだけで両側の evalagg にかけた。

```bash
sh docs/wayfinder/semindex-stage2/evidence/08/agg_identity.sh t6c   # 出力は $P/agg/t6c/{main,branch}
```

**runs.csv の末尾の列と polls.csv のほかは同じ。** consistency・decisions・discovery・probes・summary・survival の 6 つの CSV は `cmp` で同じ。runs.csv は変更後の側の最後の列を csv モジュールで落とすと main と同じで、落とした列はヘッダが `state_poll_policy`、6 行とも空だった（JSON の欄に引用符と区切りが入るので、`sed` ではなく csv で切った）。polls.csv は新しいファイルで 48 行（6 run × 有向の隣の組 8 つ）。h0→h1（p0 seed1）は窓の中で `polls_sent` 60、`state_rx` 60 で、events.jsonl 全体の `state_poll` 65 行のうち窓の外の 5 行が落ちている。方針を使っていないので決め直しと遅い返事の列はすべて 0。
