#!/bin/bash
# Start, reach and stop the model server that answers the L stage's decisions.
#
# The server runs on the GPU machine and the nodes of the emulator reach it through one
# unix socket on this machine, forwarded over ssh. Neither end is described in the
# emulator's scenario file: what a node needs at run time is one socket path in its own
# config, and a server that outlives a single run cannot be a per-run host process.
#
#   push     copy lserver.py to the GPU machine's checkout and compare the two digests
#   start    start the server there and wait for its READY line (it refuses to start
#            unless the code, the adapter, the library versions and three archived logits
#            are the ones the scoring recorded)
#   forward  open the ssh forward of the local socket to the server's socket
#   probe    ask the server, through the forward, whether it is ready
#   stop     close the forward, then stop the server with SIGTERM
#
# The local socket lives under /home/lab_shared and not under /tmp: a node process reaches
# a socket on the shared file system, and whether a node sees this machine's /tmp was never
# tested because it never had to be. Both socket paths are short and fixed, since a unix
# socket path longer than 108 bytes cannot be opened, which rules out a per-run directory.
#
# Nothing here reads or looks for an access token: the model is loaded from disk with the
# hub offline, and the three token variables are removed from the server's environment.
set -u
HOST=hpc-debian
RROOT=Development/semindex-learn            # relative to the home directory on $HOST
RDIR=$RROOT/repo/docs/wayfinder/semindex-stage2/evidence/25
HERE=$(cd "$(dirname "$0")" && pwd)
LDIR=/home/lab_shared/t25sock
LSOCK=$LDIR/l.sock
CTL=$LDIR/ssh.ctl
PY=/home/lab_shared/cefore-semindex/.venv/bin/python3
SSH="ssh -o BatchMode=yes"

probe() {
  # 2026-11-28 crash fix: this used to print whatever came back and exit 0 regardless, so
  # a live ssh forward over a dead lserver.py (the far end closes the stream, recv returns
  # b"") printed a blank line and still looked like success. Now it exits non-zero on
  # anything but the server's own {"probe": 1, "ready": true, ...} reply, an empty reply
  # included.
  "$PY" -B -c "
import json
import socket
import sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(3); s.connect('$LSOCK')
s.sendall(b'{\"probe\": 1}\n')
reply = s.recv(4096).decode().strip()
print(reply)
try:
    ok = json.loads(reply).get('ready') is True
except ValueError:
    ok = False
sys.exit(0 if ok else 1)"
}

case "${1:-}" in
  push)
    $SSH $HOST "mkdir -p ~/$RDIR" && scp -q -o BatchMode=yes "$HERE/lserver.py" "$HOST:$RDIR/lserver.py" || exit 1
    here=$(sha256sum "$HERE/lserver.py" | cut -d' ' -f1)
    there=$($SSH $HOST "sha256sum ~/$RDIR/lserver.py" | cut -d' ' -f1)
    echo "lserver.py here $here there $there"
    [ "$here" = "$there" ]
    ;;
  start)
    # nohup and a background job on the far side: a foreground ssh that holds a model for
    # hours dies with its connection. The pid file is what stop reads.
    # 2026-11-27: a pid file whose pid is alive but is NOT an lserver.py process (e.g. the
    # pid got reused by an unrelated process after a server died without cleaning up) is
    # stale, not "a server already running"; it is removed here so a start proceeds
    # instead of refusing on a pid file nothing related is actually holding.
    $SSH $HOST "cd ~/$RROOT && mkdir -p t25 && if [ -f t25/lserver.pid ] && kill -0 \$(cat t25/lserver.pid) 2>/dev/null; then \
        if ps -p \$(cat t25/lserver.pid) -o args= 2>/dev/null | grep -q lserver.py; then echo 'a server is already running'; exit 1; \
        else echo 'stale pid file (not an lserver.py process); removing it'; rm -f t25/lserver.pid; fi; \
      fi; \
      (env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HF_API_TOKEN ROOT=\$PWD STORE=\$PWD/store PYTHONPATH=\$PWD/repo/src \
        PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=\$PWD/hf-home \
        nohup ./.venv/bin/python -B repo/docs/wayfinder/semindex-stage2/evidence/25/lserver.py \
        --socket \$PWD/t25/l.sock --adapter \$PWD/adapters/b --log \$PWD/t25/lserver.log \
        > t25/lserver.out 2> t25/lserver.err < /dev/null & echo \$! > t25/lserver.pid); \
      for i in \$(seq 1 180); do grep -q '^READY' t25/lserver.out 2>/dev/null && break; \
        kill -0 \$(cat t25/lserver.pid) 2>/dev/null || { echo 'the server exited before READY:'; tail -3 t25/lserver.err; exit 1; }; sleep 1; done; \
      grep -q '^READY' t25/lserver.out || { echo 'no READY line after 180 s'; exit 1; }; echo pid \$(cat t25/lserver.pid); cat t25/lserver.out"
    ;;
  forward)
    mkdir -p "$LDIR"
    # 2026-11-27: refuse rather than start a second master over this control socket; a
    # second `-M` here would leave two ssh processes both claiming the forward and `stop`
    # would only ever close one of them.
    if [ -S "$CTL" ] && $SSH -S "$CTL" -O check $HOST >/dev/null 2>&1; then
      echo "forward: a master is already live on $CTL; refusing to start a second one"
      exit 1
    fi
    # A control socket, so that stop closes exactly this forward and no other ssh of the user.
    $SSH -M -S "$CTL" -f -N -o ExitOnForwardFailure=yes -o StreamLocalBindUnlink=yes \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
      -L "$LSOCK:/home/$(whoami)/$RROOT/t25/l.sock" $HOST && probe
    ;;
  probe)
    probe
    ;;
  stop)
    [ -S "$CTL" ] && $SSH -S "$CTL" -O exit $HOST 2>&1
    # 2026-11-27: the pid in t25/lserver.pid can outlive the server it named (pid reuse
    # by an unrelated process, or the server itself already gone) -- SIGTERM is sent only
    # after confirming, on the far side, that the pid is both running and still names
    # lserver.py; otherwise nothing is signalled and this says so instead of guessing. The
    # pid file is removed in every case where it no longer names a live lserver.py process
    # (no such pid running at all, or that pid now belongs to something else), not left
    # for the next start to trip over, and also once a signalled lserver is confirmed gone.
    $SSH $HOST "cd ~/$RROOT && if [ -f t25/lserver.pid ]; then p=\$(cat t25/lserver.pid); \
      if ! kill -0 \$p 2>/dev/null; then \
        echo \"pid \$p in t25/lserver.pid is not running; removing stale pid file\"; rm -f t25/lserver.pid; \
      elif ! ps -p \$p -o args= 2>/dev/null | grep -q lserver.py; then \
        echo \"pid \$p in t25/lserver.pid is running but is not an lserver.py process; not signalling it\"; rm -f t25/lserver.pid; \
      else \
        kill -TERM \$p 2>/dev/null; \
        for i in \$(seq 1 20); do kill -0 \$p 2>/dev/null || break; sleep 0.5; done; \
        if kill -0 \$p 2>/dev/null; then echo 'still running'; \
        else echo stopped; tail -1 t25/lserver.log; rm -f t25/lserver.pid; fi; \
      fi; \
    fi"
    ;;
  *)
    echo "usage: bash lrun.sh push|start|forward|probe|stop"; exit 2
    ;;
esac
