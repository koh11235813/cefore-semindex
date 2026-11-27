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
  "$PY" -B -c "
import socket
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(3); s.connect('$LSOCK')
s.sendall(b'{\"probe\": 1}\n'); print(s.recv(4096).decode().strip())"
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
    $SSH $HOST "cd ~/$RROOT && mkdir -p t25 && if [ -f t25/lserver.pid ] && kill -0 \$(cat t25/lserver.pid) 2>/dev/null; then echo 'a server is already running'; exit 1; fi; \
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
    $SSH $HOST "cd ~/$RROOT && if [ -f t25/lserver.pid ]; then p=\$(cat t25/lserver.pid); kill -TERM \$p 2>/dev/null; \
      for i in \$(seq 1 20); do kill -0 \$p 2>/dev/null || break; sleep 0.5; done; \
      kill -0 \$p 2>/dev/null && echo 'still running' || { echo stopped; tail -1 t25/lserver.log; }; fi"
    ;;
  *)
    echo "usage: bash lrun.sh push|start|forward|probe|stop"; exit 2
    ;;
esac
