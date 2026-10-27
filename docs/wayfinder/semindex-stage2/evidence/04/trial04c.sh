#!/bin/bash
# In-flight cap trial: k=4,8 with cap 32 at p=0.4; cap 8 is t2k4/t2k8. Waits for trial04b (Mininet runs one at a time).
S=/tmp/claude-1001/-home-lab-shared-cefore-emu-sslab/4677c249-e96f-46cd-a827-e50302f7cbe5/scratchpad
until grep -q "== trial04b finished" $S/trial04b.log; do sleep 30; done
W=/home/lab_shared/wt/semindex-s2cap
run() { env PYTHONPATH=$W/src /home/lab_shared/cefore-semindex/.venv/bin/python3 -m semindex.evalrun \
  --store /home/lab_shared/semindex-eval --recefore /home/lab_shared/cefore-emu-sslab "$@"; echo "== exit $? : $*"; }
for k in 4 8; do run --stage "t2c32k$k" --p 0.4 --seeds 1-3 --k $k --max-in-flight 32; done
echo "== trial04c finished"
