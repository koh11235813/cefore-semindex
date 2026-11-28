# The s12 prediction: the commands, in order

The held-out prediction is run once, by hand, in this order. Steps 0 to 3 touch no seed above 100 and may be repeated. Steps 4 to 7 are the held-out run; steps 4, 6 and 7 are run under `set -C` (noclobber), so a repeated line refuses before it can truncate `predict.txt`, `lpredict.log` or `predict_report.txt`. A step that failed is restarted only after its output is looked at and removed on purpose.

The GPU side is run dry (step 3) before the CPU side's once-only step (step 4). `predict_rows.csv` carries the digests of `predict.py` and `heldout.py`, so a fix to either after step 4 would make step 7 refuse the rows.

```sh
# zsh does not word-split a variable, so the interpreter path is kept alone in PY.
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/lab_shared/wt/semindex-ev25/src
PY=/home/lab_shared/cefore-semindex/.venv/bin/python3
E=/home/lab_shared/wt/semindex-ev25/docs/wayfinder/semindex-stage2/evidence/25
DRY=/tmp/ev25-dry                                   # outside evidence/, for the dry files
H=hpc-debian
R=Development/semindex-learn                        # ROOT on $H, relative to its home
RD=$R/repo/docs/wayfinder/semindex-stage2/evidence/25
FAR='env PYTHONDONTWRITEBYTECODE=1 STORE=$PWD/store PYTHONPATH=$PWD/repo/src HF_HUB_OFFLINE=1 ./.venv/bin/python -B repo/docs/wayfinder/semindex-stage2/evidence/25/lpredict.py $PWD'
```

`PYTHONPATH` is the worktree's own `src`, not `semindex-decider/src`: `../20/predict.py` refuses any other. Gate 0 of `predict.py` holds the two trees byte-equal.

## 0. The gates, and the dry input for the GPU side (about 70 s)

```sh
mkdir -p $DRY && cd $E && $PY -B predict.py --dry --write-l-input $DRY/dry_input.json > $DRY/predict_dry.txt; echo rc $?
DSHA=$(sha256sum $DRY/dry_input.json | cut -d' ' -f1); echo $DSHA
```

Repeating this step needs `$DRY/dry_input.json` removed first: `heldout.write_runs` opens it exclusively and refuses otherwise. `predict_dry.txt` itself is simply overwritten by the shell redirect.

## 1. Push the L code and the dry input

The card must be free. Stop the t25 model server first if it is up and no L testbed stage is running (`bash $E/lrun.sh stop`): `lpredict.py` refuses a card that has another compute process on it.

```sh
ssh -o BatchMode=yes $H "mkdir -p ~/$RD ~/$R/ev25predict"
scp -q -o BatchMode=yes $E/heldout.py $E/lpredict.py $E/lserver.py $H:$RD/
scp -q -o BatchMode=yes $DRY/dry_input.json $H:$R/ev25predict/
(cd $E && sha256sum heldout.py lpredict.py lserver.py; sha256sum $DRY/dry_input.json)
ssh -o BatchMode=yes $H "cd ~/$RD && sha256sum heldout.py lpredict.py lserver.py ~/$R/ev25predict/dry_input.json"
```

The two digest lists must agree line by line.

## 2. The GPU side, dry (24 check runs, about 2 min)

```sh
ssh -o BatchMode=yes $H "cd ~/$R && $FAR --input \$PWD/ev25predict/dry_input.json --input-sha $DSHA \
  --out \$PWD/ev25predict/l_dry.csv --dry --k 4 > ev25predict/l_dry.log 2> ev25predict/l_dry.err"; echo rc $?
ssh -o BatchMode=yes $H "tail -4 ~/$R/ev25predict/l_dry.log"
```

The log must say that all 24 check runs equal the scoring's b rows. Repeating this step, whether it failed or just needs redoing, needs `ev25predict/l_dry.csv` removed on `$H` first (`lpredict.py` refuses an `--out` that already exists); `ev25predict/l_dry.csv.tmp` also stays behind with the worker logs after every run, a successful one included, and must be removed before the next try.

## 3. The dry rows, back through the report's checks

```sh
scp -q -o BatchMode=yes $H:$R/ev25predict/l_dry.csv $DRY/
cd $E && $PY -B predict.py --dry --check-l-rows $DRY/l_dry.csv --l-input $DRY/dry_input.json | grep "^L rows\|STOP"
```

**Before step 4: commit and push.** `spec.md`, and every script steps 4 to 7 run (`predict.py`, `heldout.py`, `lpredict.py`, this file), must already be committed and pushed: that is commit (1) of `spec.md`'s "Order of commits, fixed here."

## 4. The CPU side, once (seeds 121 to 160, about 3 min)

```sh
set -C
cd $E && $PY -B predict.py --held-out-once > predict.txt 2> predict.err; echo rc $?
SHA=$(sha256sum predict_runs.json | cut -d' ' -f1); echo $SHA
```

This writes `predict_runs.json` (the GPU side's input: run dicts, the 24 check runs with the scoring's own strings, the 1600 held-out (seed, stream) pairs) and `predict_rows.csv` (9600 rows of the six CPU stages). It refuses to start if either file exists.

## 5. Push the held-out input

```sh
scp -q -o BatchMode=yes $E/predict_runs.json $H:$R/ev25predict/
ssh -o BatchMode=yes $H "sha256sum ~/$R/ev25predict/predict_runs.json"      # must be $SHA
```

## 6. The GPU side, once (seeds 121 to 160, about 80 min)

It runs under `nohup`, because a foreground ssh that holds the card for over an hour dies with its connection. It has no resume: a worker that dies loses the whole held-out phase. A failed run may leave partial rows in the directory `predict_l.csv.tmp`; nothing reads that directory, and it is looked at before it is removed.

```sh
ssh -o BatchMode=yes $H "set -C; cd ~/$R && (nohup $FAR --input \$PWD/ev25predict/predict_runs.json --input-sha $SHA \
  --out \$PWD/ev25predict/predict_l.csv --held-out-once --k 4 \
  > ev25predict/lpredict.log 2> ev25predict/lpredict.err < /dev/null &)"
ssh -o BatchMode=yes $H "tail -3 ~/$R/ev25predict/lpredict.log"     # until "wrote ... predict_l.csv"
```

## 7. Fetch, then the full prediction (about 90 s)

```sh
set -C
scp -q -o BatchMode=yes $H:$R/ev25predict/predict_l.csv $H:$R/ev25predict/lpredict.log $E/
cd $E && $PY -B predict.py --report --l-rows predict_l.csv > predict_report.txt; echo rc $?
```

`--report` re-runs the gates on seeds 1 to 100 and simulates no held-out seed. It reads the three files and holds L's rows to four things: the input file's digest, candidate b's adapter digest, the digest of the local copies of the L code, and the 24 check runs against `../23/eval_dry.csv` and `../23/eval.csv`. `predict_report.txt` holds the tables `spec.md` needs.

**After step 7: fill "Predicted values" and commit.** Copy the six-stage rows from `predict.txt` and the seven-stage tables from `predict_report.txt` into `spec.md`'s "Predicted values" section, and commit that change alone, nothing above it moving: that is commit (3) of `spec.md`'s "Order of commits, fixed here," and `MAIN12_RUN.md` step 1's `DOCS_COMMIT` is this commit.
