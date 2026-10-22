# Separate repository; ReCefore gains only generic testbed features

The Semantic Index and its Sync agent live in this repository, not inside ReCefore. ReCefore is used only as the Mininet testbed, and the changes it receives for this work are generic testbed features that carry no knowledge of synchronization: per-host long-running processes (`host_apps`), an explicit link list for the disaster scenario, and prefix-to-host route announcements fed into its existing FIB generation. We chose this over importing ReCefore's internal setup/teardown API from here (which would break on every ReCefore refactor) and over a standalone Mininet script (which would re-implement link events, results and log collection).

## Consequences

- An experiment is a ReCefore YAML plus this repository's agent; anything the agent needs at run time must be expressible through those generic features.
- ReCefore changes go through its own branch and PR; this repository depends on a ReCefore revision that has them.
