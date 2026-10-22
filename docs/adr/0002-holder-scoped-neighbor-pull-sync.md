# Holder-scoped, neighbor-only pull synchronization over plain Interest/Content Object

The initial Sync agent synchronizes only with configured direct neighbors, by pull: it periodically asks each neighbor for its holder state and then fetches the Index records it lacks, using names scoped by the answering holder (illustrative: `ccnx:/semidx/<group>/<holder>/sv/<requester>/<epoch_ms>` and `ccnx:/semidx/<group>/<holder>/rec/<origin>/<seq>`). It uses no Interest payload and no hop-by-hop header piggyback. We chose this because it needs no Cefore modification, makes explicit which node is asked and under which name it answers (a Replica holder can serve records while their Index origin is unreachable), and keeps the required cefpyco surface to prefix registration, Interest/Data send and receive, and expiry/cache_time. This is a staged design decision, not a measured claim that it beats header piggyback.

## Considered Options

- **Header piggyback (option B)**: small update hints in `hdr_org`/`msg_org` on ordinary traffic. Kept as a later comparison; it still needs an independent sync opportunity when there is no ordinary traffic, and the hop-by-hop area is limited to 247 bytes and outside signature coverage.
- **Origin-scoped record names** (`.../rec/<origin>/<seq>` routed toward the origin): would let any cache on the path answer, but assumes Interests toward an origin reach a Replica holder, which we do not assume.

## Consequences

- Requester identity and a per-request time component in the state name keep Cefore's PIT from aggregating different requesters and keep Content Stores from answering with an old state.
- Propagation beyond one hop costs about one sync period per hop, and a record fetched from one holder is not reused from a Content Store under another holder's name.
