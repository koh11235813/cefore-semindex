# The s12 sweep: the commands, in order

`main12.sh` loops over the 280 (stage, seed) runs. It refuses to start if the testbed is already
busy, if its checkout does not descend from the merged src commit `c333655`, if that checkout's
`src` differs from `c333655`'s, or if the checkout holds any change or untracked file. The
checkout, the L server and the relay are set up here by hand, in this order.

## 0. Before anything: confirm nothing is already running

```sh
pgrep -fa '[m]ininet|[c]efnetd|[c]smgrd'   # must print nothing
```

## 1. The one run checkout, at the docs commit that holds this file

The docs commit sits on top of `c333655`, so its `src` is the merged one, and the decider files
(`mdecider.py`, `ldecider.py`, `../24/coef_a.npy`, `lrun.sh`) and `main12.sh` itself are the ones
that commit records. `DOCS_COMMIT` is the commit that also holds `spec.md` with its predicted
values filled in.

```sh
DOCS_COMMIT=<the docs commit that holds spec.md with its predicted values filled in>
git -C /home/lab_shared/cefore-semindex fetch fork
git -C /home/lab_shared/cefore-semindex worktree add --detach \
  /home/lab_shared/wt/semindex-main12 "$DOCS_COMMIT"
git -C /home/lab_shared/wt/semindex-main12 diff --stat c333655a52f52a36b0eb4ae3a56c0970317b9e9d HEAD -- src   # must print nothing
LDIR=/home/lab_shared/wt/semindex-main12/docs/wayfinder/semindex-stage2/evidence/25
```

Nothing is written inside this checkout while the sweep runs, the sweep log included: an
untracked file there makes `main12.sh` refuse the next start.

## 2. Start the L server and the relay (`main12.sh` never does this itself)

```sh
bash $LDIR/lrun.sh push
ssh -o BatchMode=yes hpc-debian "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"   # must print nothing: no lpredict.py worker may still hold the card
bash $LDIR/lrun.sh start
bash $LDIR/lrun.sh forward
bash $LDIR/lrun.sh probe
```

`main12.sh sweep` probes again before every `s12l` run and stops with a message the moment the
probe fails; it starts or restarts nothing by itself.

## 3. Review the 280 commands before running any of them

```sh
bash $LDIR/main12.sh --print 121      # one seed, 7 lines
bash $LDIR/main12.sh --print | grep -c '^env '   # must print 280
```

Nothing is run, started or connected to in this mode; it hashes the two decider files under the
checkout, loads `../22/port.py` once to check that `SEND_START` equals its `ARMS[1]`, and prints the
evalrun command lines.

## 4. Run the sweep

`tee` opens its target file before `main12.sh` has run a single stage (the first `s12m`/`s12l`
node is what would otherwise create `/home/lab_shared/t25rec-s12`), so the directory is made by
hand first, as the user, not as root. `-a` is required too: a resumed sweep runs this exact line
again, and without it `tee` would truncate the log's own first line, which step 6 below keeps
for RUN.txt.

```sh
mkdir -p /home/lab_shared/t25rec-s12
sudo bash $LDIR/main12.sh sweep 2>&1 | tee -a /home/lab_shared/t25rec-s12/sweep.log
```

Resumable: running the same command again skips every (stage, seed) already marked DONE in the
store, because `semindex.evalrun` checks that marker itself. The first line of the log records
the checkout's commit and dirty count, that its `src` equals `c333655`, the recefore commit and
its dirty count, and the start time; keep that line for RUN.txt.

## 5. After the sweep: stop the relay and the server

```sh
bash $LDIR/lrun.sh stop
```

## 6. Record RUN.txt

From the sweep log's first and last line: the checkout's commit, that its `src` equals
`c333655a52f52a36b0eb4ae3a56c0970317b9e9d`, the recefore commit and its dirty count, and the
start and finish times, as evidence/20's RUN.txt records main10's.

Each `s12m` and `s12l` run's recorder and, for `s12l`, the client's own failure counters land in
`/home/lab_shared/t25rec-s12/<stage>/<seed>/` (one directory per run, `<node>.jsonl` from the
recorder and, for `s12l` only, `<node>.lclient.json` from the client); `parity.py` reads these as
it reads `t25rec`'s. A `(stage, seed)` retaken because its store run had no `DONE` marker has its
old directory moved aside first, to `<seed>.attempt-N`, never deleted; `main12.sh` does this
itself, so `parity.py` and `arms.py` always read only the current attempt's directory.
