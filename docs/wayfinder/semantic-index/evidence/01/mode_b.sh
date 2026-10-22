#!/bin/bash
# RUNBOOK Mode B (CS_MODE=0, no csmgrd) + same-name-twice observation
cd /home/lab_shared/cefpyco
export SMOKE_CONF=$(mktemp -d /tmp/semindex-smoke-conf.XXXXXX)
cp -a /usr/local/cefore/. "$SMOKE_CONF"/; : > "$SMOKE_CONF"/cefnetd.fib
PY=/home/lab_shared/cefore-semindex/.venv/bin/python3
sed -i 's/^#*CS_MODE=.*/CS_MODE=0/' "$SMOKE_CONF"/cefnetd.conf
grep -n '^CS_MODE' "$SMOKE_CONF"/cefnetd.conf
PRODUCER_PID=""
cleanup() { [ -n "$PRODUCER_PID" ] && sudo kill "$PRODUCER_PID" 2>/dev/null; sudo cefnetdstop -d "$SMOKE_CONF" 2>/dev/null; }
trap cleanup EXIT
sudo cefnetdstart -d "$SMOKE_CONF"; sleep 1
ls -l /tmp/cef_9695.0
sudo cefstatus -d "$SMOKE_CONF" | grep "Cache Mode"
sudo "$PY" semindex_smoke/producer.py --prefix ccnx:/semidx/smoke/a --ceforedir "$SMOKE_CONF" \
    --expiry-ms 1000 --cache-time-ms 0 > /tmp/semindex-smoke-producer-b.log 2>&1 &
PRODUCER_PID=$!; sleep 1
echo "=== consumer"; sudo "$PY" semindex_smoke/consumer.py ccnx:/semidx/smoke/a/hello --ceforedir "$SMOKE_CONF"; echo "consumer exit: $?"
echo "=== freshness"; sudo "$PY" semindex_smoke/state_freshness.py --prefix ccnx:/semidx/smoke/a --ceforedir "$SMOKE_CONF" --count 20 | tail -1; echo "state_freshness exit: ${PIPESTATUS[0]}"
echo "=== same name twice"; for i in 1 2; do sudo "$PY" semindex_smoke/consumer.py ccnx:/semidx/smoke/a/same --ceforedir "$SMOKE_CONF"; echo "exit $?"; done
sudo kill "$PRODUCER_PID"; wait "$PRODUCER_PID" 2>/dev/null; PRODUCER_PID=""
echo "--- producer log (tail)"; grep -v '/sv/' /tmp/semindex-smoke-producer-b.log
