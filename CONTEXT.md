# Cefore Semantic Index

Disaster-time, intermittently connected multi-hop ICN (Cefore / CCNx 1.0) in which every node keeps a distributed Semantic Index, so that a node that does not know a content name in advance can discover it from semantic conditions and then retrieve the content by name. This glossary fixes the project-specific language; general programming and ICN concepts are excluded.

## Language

### Index

**Semantic Index**:
The per-node collection of Index records that maps semantic attributes to real content names. Each node holds its own copy; there is no central index.
_Avoid_: catalog, directory, search DB

**Index record**:
One immutable entry of the Semantic Index: an origin-assigned update ID, the real content name, the semantic attributes, the observation time, the schema version and the model identity. A correction is a new Index record that names the one it supersedes, never an overwrite.
_Avoid_: index entry, metadata row, Record (unqualified)

**Holder state**:
A holder's summary of which Index records it holds, per Index origin: the highest sequence held contiguously from the first, plus the sparse sequences held above it. It is what a neighbor compares against its own holdings to know what to fetch; it says nothing about records the holder has merely heard of.
_Avoid_: state vector (NDN SVS carries the highest known sequence, which is a different quantity), digest

### Roles

**Index origin**:
The node that created an Index record and assigned its update ID. Distinct from the Publisher of the content body, even though one sensor node is usually both.
_Avoid_: producer, producer ID, source node

**Publisher**:
The host that serves the content body named by an Index record. Same meaning as in ReCefore.
_Avoid_: origin (for the content body)

**Replica holder**:
A node that holds Index records whose Index origin is another node, and serves them on to others.
_Avoid_: cache node (that is a Content Store role), relay

**Sync agent**:
The per-node application that exchanges state with neighbors and fetches missing Index records over Cefore.
_Avoid_: daemon, sync server
