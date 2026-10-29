# Agents open index.db with synchronous=NORMAL by default

Each Sync agent commits every received Index record in its own transaction on a WAL SQLite database. Under SQLite's default `synchronous=FULL` every commit costs an fsync, and with five agents on one virtual disk the fetching agent spent 33–89 ms handling each received record; that handling time, not the link or Cefore (a lone fetch returns in 1–2 ms), shaped fetch delays and the waits behind the per-neighbor in-flight cap (`docs/wayfinder/semindex-stage2/evidence/05/` section 4). With `synchronous=NORMAL` the same handling took 0 ms median and at most 1 ms (p99 1 ms) over 15 runs, fetch receive waits fell to 2–9 ms, link-up fetch timeouts at cap 32 fell from 93 to 0, and discovery medians dropped (e.g. 6.08 s → 2.32 s at k = 8, p = 0) (`evidence/06/`). We therefore made NORMAL the agents' default: we trade per-record durability for throughput, because the sync protocol re-fetches a lost record from any neighbor that still holds it, so an fsync per record buys little. The Store always issues the PRAGMA; there is no "leave SQLite's default" state.

## Considered Options

- **Keep FULL as the default and pass NORMAL in experiments**: rejected because the default would then describe a configuration no experiment uses, and every run would depend on remembering a flag whose absence silently changes the measured fetch timing.
- **Batch commits per loop step to keep FULL's guarantee at amortised cost**: the agent handles one Data per loop step, so batching would need a restructured receive path; not worth it while NORMAL already brings handling to ≤ 1 ms.
- **Put the database on tmpfs in experiments only**: would remove the fsync cost on the testbed but measure a storage setup no deployment has, and still leave FULL as the default the code describes.
- **Replace SQLite with another database or a custom store**: not needed now, since per-record handling is ≤ 1 ms under NORMAL. Revisit if the tail of per-record handling grows at larger scale or WAL checkpoint stalls appear; even then, the single-threaded design that handles one Data per loop step would likely limit throughput before the database does.

## Consequences

- An OS crash (not an agent-process crash, which loses nothing under WAL + NORMAL) may lose a node's own just-created records that no neighbor has fetched yet; lost records it had received are fetched again from any neighbor that still holds them.
- Runs recorded before this change ran with FULL; reproducing them needs `--store-synchronous FULL`. Their `meta.json` has `store_synchronous` null or missing, which aggregation keeps as an empty cell.
- `semindex.evalrun` always records the effective mode in `meta.json` (the given one, or NORMAL), never null. Generated agent configs still omit the key when no mode is given, so generated files are unchanged; only what they mean at run time changed.
