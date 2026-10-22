#!/bin/bash
# RUNBOOK setup + Mode A, producer at state-Data policy (expiry 1s, cache_time 0)
set -u
cd /home/lab_shared/cefpyco
export SMOKE_CONF=$(mktemp -d /tmp/semindex-smoke-conf.XXXXXX)
cp -a /usr/local/cefore/. "$SMOKE_CONF"/
: > "$SMOKE_CONF"/cefnetd.fib
echo "CACHE_TYPE=memory" >> "$SMOKE_CONF"/csmgrd.conf
echo "SMOKE_CONF=$SMOKE_CONF"
PY=/home/lab_shared/cefore-semindex/.venv/bin/python3
sed -i 's/^#*CS_MODE=.*/CS_MODE=2/' "$SMOKE_CONF"/cefnetd.conf
grep -n '^CS_MODE' "$SMOKE_CONF"/cefnetd.conf || echo "CS_MODE=2" >> "$SMOKE_CONF"/cefnetd.conf
grep -n '^CS_MODE' "$SMOKE_CONF"/cefnetd.conf
PRODUCER_PID=""; CEFPUTFILE_PID=""
cleanup() {
    [ -n "$PRODUCER_PID" ] && sudo kill "$PRODUCER_PID" 2>/dev/null
    [ -n "$CEFPUTFILE_PID" ] && sudo kill "$CEFPUTFILE_PID" 2>/dev/null
    sudo cefnetdstop -d "$SMOKE_CONF" 2>/dev/null
    sudo csmgrdstop -d "$SMOKE_CONF" 2>/dev/null
}
trap cleanup EXIT
sudo csmgrdstart -d "$SMOKE_CONF"
sudo cefnetdstart -d "$SMOKE_CONF"
sleep 1
ls -l /tmp/cef_9695.0
sudo csmgrstatus | head -20
sudo cefstatus -d "$SMOKE_CONF" | grep "Cache Mode"
echo "=== item1-3 producer"
sudo "$PY" semindex_smoke/producer.py --prefix ccnx:/semidx/smoke/a --ceforedir "$SMOKE_CONF" \
    --expiry-ms 1000 --cache-time-ms 0 > /tmp/semindex-smoke-producer-a.log 2>&1 &
PRODUCER_PID=$!
sleep 1
echo "=== consumer"
time sudo "$PY" semindex_smoke/consumer.py ccnx:/semidx/smoke/a/hello --ceforedir "$SMOKE_CONF"
echo "consumer exit: $?"
echo "=== item4 freshness"
time sudo "$PY" semindex_smoke/state_freshness.py --prefix ccnx:/semidx/smoke/a --ceforedir "$SMOKE_CONF" --count 20
echo "state_freshness exit: $?"
echo "=== extra: same name twice (CS answer check, cache_time 0)"
sudo "$PY" semindex_smoke/consumer.py ccnx:/semidx/smoke/a/same --ceforedir "$SMOKE_CONF"; echo "exit $?"
sudo "$PY" semindex_smoke/consumer.py ccnx:/semidx/smoke/a/same --ceforedir "$SMOKE_CONF"; echo "exit $?"
sudo kill "$PRODUCER_PID"; wait "$PRODUCER_PID" 2>/dev/null; PRODUCER_PID=""
echo "--- producer log"; cat /tmp/semindex-smoke-producer-a.log
echo "=== item5 chunk fetch"
head -c 5000 /dev/urandom > /tmp/semindex-smoke-src.bin
sudo cefputfile ccnx:/semidx/smoke/file -f /tmp/semindex-smoke-src.bin -d "$SMOKE_CONF" -b 1024 &
CEFPUTFILE_PID=$!
sleep 1
sudo "$PY" semindex_smoke/chunk_fetch.py ccnx:/semidx/smoke/file /tmp/semindex-smoke-src.bin --ceforedir "$SMOKE_CONF"
echo "chunk_fetch exit: $?"
sudo kill "$CEFPUTFILE_PID" 2>/dev/null; CEFPUTFILE_PID=""
