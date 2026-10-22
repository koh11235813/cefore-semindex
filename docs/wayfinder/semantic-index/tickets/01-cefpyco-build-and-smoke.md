---
status: open
type: prototype
claimed-by:
blocked-by: []
---

## Task

Build upstream cefpyco v0.10.1 unmodified against the installed Cefore v0.12.0 into this repository's venv (cmake from PyPI, not apt), then prove on ReCefore-provisioned Mininet hosts with their real `cefnetd.conf` (CS_MODE=2, csmgrd present on cache nodes):

1. `create_handle(ceforedir=<node conf dir>)` binds to that node's cefnetd.
2. An app that `register`s `ccnx:/semidx/g1/hN` receives an Interest for a longer name under it, on a csmgrd node and on a non-csmgrd node.
3. `send_data` with no chunk TLV answers it; the requester's `receive` gets it within `lifetime=2000`.
4. State-style names that change per request are never answered from a Content Store with an older object.
5. `send_interest(chunk_num=k)` fetches all chunks of a cefputfile-published object.

Patch only what fails, on a branch of `/home/lab_shared/cefpyco`. Keep the smoke code as the binding's regression test.

## Acceptance

Each of 1–5 has recorded command output. A negative result on 2 or 4 stops the effort and goes back to the user.
