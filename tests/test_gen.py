"""Tests for the semindex.gen experiment generator."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from semindex import gen, schema
from semindex.gen import (
    ExperimentError,
    generate,
    main,
    parse_experiment,
    write_experiment,
)


def _minimal_raw(**overrides) -> dict:
    """A minimal valid experiment description, for tests that only need a baseline to mutate."""
    raw = {
        "name": "mini",
        "num": 1,
        "seed": 1,
        "group": "g",
        "hosts": 3,
        "links": [[0, 1], [1, 2]],
        "duration": 10,
        "output_dir": "logs/mini",
        "agents": [0, 1],
        "publisher": {
            "host": 0,
            "uri": "ccnx:/content/h0/body",
            "file": "./sample-putfile",
            "expiry": 3000,
            "cache_time": 3000,
        },
    }
    raw.update(overrides)
    return raw


# -- parse_experiment: shape and required/unknown keys -----------------------


def test_parse_experiment_accepts_the_minimal_raw():
    """Parse experiment accepts the minimal raw."""
    spec = parse_experiment(_minimal_raw())
    assert spec.name == "mini"
    assert spec.agents == (0, 1)
    assert spec.schedule == {}
    assert spec.probes == {}
    assert spec.fault == {}
    assert spec.events == ()


def test_parse_experiment_rejects_an_unknown_top_level_key():
    """Parse experiment rejects an unknown top level key."""
    raw = _minimal_raw()
    raw["bogus"] = 1
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(raw)
    assert str(exc_info.value) == "unknown experiment key: 'bogus'"


def test_parse_experiment_rejects_a_missing_required_key():
    """Parse experiment rejects a missing required key."""
    raw = _minimal_raw()
    del raw["hosts"]
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(raw)
    assert str(exc_info.value) == "missing required experiment key: 'hosts'"


def test_parse_experiment_rejects_hosts_below_three():
    """Parse experiment rejects hosts below three."""
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(_minimal_raw(hosts=2, links=[[0, 1]], agents=[0, 1]))
    assert str(exc_info.value) == "hosts must be >= 3"


def test_parse_experiment_rejects_an_absolute_path_as_name():
    """2026-10-22 fix: `name` becomes a path component verbatim in
    write_experiment, with no sanitizing there -- an absolute path must be rejected before it
    reaches that point."""
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(_minimal_raw(name="/etc/passwd"))
    assert str(exc_info.value) == "name must not be a path: '/etc/passwd'"


def test_parse_experiment_rejects_a_name_containing_a_path_separator():
    """2026-10-22 fix: a name like "../escaped" would otherwise let
    write_experiment write <outdir>/escaped.yaml and <outdir>/escaped/<node>.json outside the
    requested outdir."""
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(_minimal_raw(name="../escaped"))
    assert str(exc_info.value) == "name must not be a path: '../escaped'"


def test_parse_experiment_rejects_dot_as_name():
    """Parse experiment rejects dot as name."""
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(_minimal_raw(name="."))
    assert str(exc_info.value) == "name must not be a path: '.'"


def test_parse_experiment_rejects_dotdot_as_name():
    """Parse experiment rejects dotdot as name."""
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(_minimal_raw(name=".."))
    assert str(exc_info.value) == "name must not be a path: '..'"


def test_parse_experiment_rejects_a_non_positive_duration():
    """Parse experiment rejects a non positive duration."""
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(_minimal_raw(duration=0))
    assert str(exc_info.value) == "duration must be > 0"


def test_parse_experiment_rejects_a_float_duration():
    """ReCefore's own `duration` option is declared type int; a whole-numbered float still fails it."""
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(_minimal_raw(duration=10.0))
    assert str(exc_info.value) == "duration must be an integer"


# -- links ---------------------------------------------------------------


def test_parse_experiment_rejects_a_self_link():
    """Parse experiment rejects a self link."""
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(_minimal_raw(links=[[0, 0], [0, 1]]))
    assert str(exc_info.value) == "links[0] must not link a host to itself"


def test_parse_experiment_rejects_an_out_of_range_link_host():
    """Parse experiment rejects an out of range link host."""
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(_minimal_raw(links=[[0, 3]]))
    assert str(exc_info.value) == "links[0][1] must be in range [0, 3)"


# -- agents ----------------------------------------------------------------


def test_parse_experiment_rejects_duplicate_agents():
    """Parse experiment rejects duplicate agents."""
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(_minimal_raw(agents=[0, 0, 1]))
    assert str(exc_info.value) == "agents must not contain duplicates"


def test_parse_experiment_sorts_agents():
    """Parse experiment sorts agents."""
    spec = parse_experiment(_minimal_raw(agents=[1, 0]))
    assert spec.agents == (0, 1)


# -- publisher ---------------------------------------------------------------


def test_parse_experiment_rejects_publisher_missing_a_key():
    """Parse experiment rejects publisher missing a key."""
    raw = _minimal_raw()
    del raw["publisher"]["file"]
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(raw)
    assert str(exc_info.value) == "publisher missing required key: 'file'"


def test_parse_experiment_rejects_an_out_of_range_publisher_host():
    """Parse experiment rejects an out of range publisher host."""
    raw = _minimal_raw()
    raw["publisher"]["host"] = 9
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(raw)
    assert str(exc_info.value) == "publisher.host must be in range [0, 3)"


# -- schedule ----------------------------------------------------------------


def test_parse_experiment_rejects_schedule_content_name_as_an_input_key():
    """Schedule entries never carry content_name: gen derives it, so two files can't disagree about it."""
    raw = _minimal_raw(
        schedule={"0": [{"at_s": 3, "content_name": "x", "zone": "zone1", "disaster": "flood", "media": "image"}]}
    )
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(raw)
    assert str(exc_info.value) == "schedule['0'][0] has unknown key: 'content_name'"


def test_parse_experiment_rejects_schedule_for_a_host_not_in_agents():
    """Parse experiment rejects schedule for a host not in agents."""
    raw = _minimal_raw(
        schedule={"2": [{"at_s": 3, "zone": "zone1", "disaster": "flood", "media": "image"}]}
    )
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(raw)
    assert str(exc_info.value) == "schedule['2'] names a host that is not in agents"


def test_parse_experiment_rejects_an_invalid_zone():
    """Parse experiment rejects an invalid zone."""
    raw = _minimal_raw(
        schedule={"0": [{"at_s": 3, "zone": "zoneX", "disaster": "flood", "media": "image"}]}
    )
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(raw)
    assert str(exc_info.value) == f"schedule['0'][0].zone must be one of {sorted(schema.ZONES)}"


def test_parse_experiment_rejects_a_schedule_entry_below_the_startup_margin():
    """Parse experiment rejects a schedule entry below the startup margin."""
    raw = _minimal_raw(
        schedule={"0": [{"at_s": 1, "zone": "zone1", "disaster": "flood", "media": "image"}]}
    )
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(raw)
    assert str(exc_info.value) == "earliest schedule at_s (1.0) is below the 3.0s startup margin (no host_apps readiness hook)"


# -- probes --------------------------------------------------------------


def test_parse_experiment_rejects_probes_for_a_host_not_in_agents():
    """Parse experiment rejects probes for a host not in agents."""
    raw = _minimal_raw(probes={"2": [{"at_s": 5}]})
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(raw)
    assert str(exc_info.value) == "probes['2'] names a host that is not in agents"


def test_parse_experiment_accepts_a_probe_with_no_attrs_filter():
    """Parse experiment accepts a probe with no attrs filter."""
    spec = parse_experiment(_minimal_raw(probes={"1": [{"at_s": 5}]}))
    probe = spec.probes[1][0]
    assert (probe.zone, probe.disaster, probe.media) == (None, None, None)


# -- fault -----------------------------------------------------------------


def test_parse_experiment_rejects_fault_seq_below_one():
    """Parse experiment rejects fault seq below one."""
    raw = _minimal_raw(fault={"0": {"drop_rec": [{"origin": "h0", "seq": 0, "times": 1}]}})
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(raw)
    assert str(exc_info.value) == "fault['0'].drop_rec[0].seq must be >= 1"


def test_parse_experiment_rejects_fault_for_a_host_not_in_agents():
    """Parse experiment rejects fault for a host not in agents."""
    raw = _minimal_raw(fault={"2": {"drop_rec": [{"origin": "h0", "seq": 1, "times": 1}]}})
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(raw)
    assert str(exc_info.value) == "fault['2'] names a host that is not in agents"


# -- events ------------------------------------------------------------------


def test_parse_experiment_rejects_an_unknown_event_type():
    """Parse experiment rejects an unknown event type."""
    raw = _minimal_raw(events=[{"at": 5, "type": "host_down", "host": 0}])
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(raw)
    assert str(exc_info.value) == "events[0].type must be one of ['host_app_restart', 'link_down', 'link_up']"


def test_parse_experiment_rejects_a_link_event_missing_nodes():
    """Parse experiment rejects a link event missing nodes."""
    raw = _minimal_raw(events=[{"at": 5, "type": "link_down"}])
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(raw)
    assert str(exc_info.value) == "events[0] missing required key: 'nodes'"


def test_parse_experiment_rejects_a_restart_event_with_non_positive_downtime():
    """Parse experiment rejects a restart event with non positive downtime."""
    raw = _minimal_raw(events=[{"at": 5, "type": "host_app_restart", "host": 0, "downtime_s": 0}])
    with pytest.raises(ExperimentError) as exc_info:
        parse_experiment(raw)
    assert str(exc_info.value) == "events[0].downtime_s must be > 0"


def test_parse_experiment_accepts_a_restart_event():
    """Parse experiment accepts a restart event."""
    spec = parse_experiment(
        _minimal_raw(events=[{"at": 5, "type": "host_app_restart", "host": 0, "downtime_s": 2}])
    )
    assert len(spec.events) == 1
    event = spec.events[0]
    assert (event.at, event.type, event.fields) == (5.0, "host_app_restart", {"host": 0, "downtime_s": 2})


# -- generate(): the full "must always emit" shape ---------------------------


def _three_host_line_spec():
    """A 3-host line (h0-h1-h2) with one agent host (h1) kept idle-free and one record on h0."""
    raw = _minimal_raw(
        name="g1",
        num=42,
        seed=7,
        group="demo",
        hosts=3,
        links=[[0, 1], [1, 2]],
        agents=[0, 1, 2],
        schedule={"0": [{"at_s": 3, "zone": "zone1", "disaster": "flood", "media": "image"}]},
        probes={"2": [{"at_s": 9, "zone": "zone1"}]},
        events=[{"at": 6, "type": "link_down", "nodes": [0, 1]}],
    )
    return parse_experiment(raw)


def test_generate_recefore_config_has_every_required_shape():
    """Generate recefore config has every required shape."""
    spec = _three_host_line_spec()
    generated = generate(spec, "/abs/out/g1")
    cfg = generated.recefore_config

    assert cfg["k"] == 1
    assert cfg["hosts"] == 3
    assert cfg["num"] == 42
    assert cfg["seed"] == 7
    assert cfg["no_cli"] is True
    assert cfg["results_json"] == "results.json"
    assert cfg["publisher_host"] == 0
    assert cfg["links"] == [[0, 1], [1, 2]]
    assert cfg["debug"] == {"artifacts": ["node_dirs"]}
    assert cfg["warmup_gets"] == [{"host": 0, "uri": "ccnx:/content/h0/body"}]
    assert cfg["prefix_routes"] == {
        "ccnx:/semidx/demo/h0": 0,
        "ccnx:/semidx/demo/h1": 1,
        "ccnx:/semidx/demo/h2": 2,
    }

    # The publisher's put is always first and always at t=0; the passthrough
    # event (link_down) follows it, unmodified except for the envelope keys.
    assert cfg["events"][0] == {
        "at": 0,
        "type": "put",
        "host": 0,
        "uri": "ccnx:/content/h0/body",
        "file": "./sample-putfile",
        "expiry": 3000,
        "cache_time": 3000,
    }
    assert cfg["events"][1] == {"at": 6.0, "type": "link_down", "nodes": [0, 1]}
    assert len(cfg["events"]) == 2

    assert cfg["host_apps"] == [
        {
            "hosts": [0, 1, 2],
            "argv": [
                "/home/lab_shared/cefore-semindex/.venv/bin/python3",
                "-m",
                "semindex.agent",
                "--config",
                "/abs/out/g1/{node}.json",
                "--node",
                "{node}",
                "--t0-ms",
                "{t0_ms}",
                "--cefore-conf-dir",
                "{cefore_conf_dir}",
                "--state-dir",
                "{node_dir}/semindex",
            ],
            "env": {"PYTHONPATH": gen.SEMINDEX_SRC_DIR},
            "log": "{node_dir}/semindex_agent.log",
        }
    ]


def test_host_apps_pythonpath_points_at_the_generating_package_src():
    """The agent is launched against the same src/ tree as the generator that wrote its config."""
    import semindex

    assert gen.SEMINDEX_SRC_DIR == os.path.dirname(os.path.dirname(os.path.abspath(semindex.__file__)))


def test_generate_agent_configs_fill_in_content_name_and_derive_neighbors():
    """Generate agent configs fill in content name and derive neighbors."""
    spec = _three_host_line_spec()
    generated = generate(spec, "/abs/out/g1")

    h0 = generated.agent_configs["h0"]
    assert h0["node"] == "h0"
    assert h0["neighbors"] == ["h1"]
    # h0 is the publisher: its own record names exactly the body it put.
    assert h0["schedule"] == [
        {"at_s": 3.0, "content_name": "ccnx:/content/h0/body", "zone": "zone1", "disaster": "flood", "media": "image"}
    ]
    assert h0["probes"] == []
    assert "fault" not in h0
    assert "stop_at_s" not in h0
    assert "counters_snapshot_at_s" not in h0

    h1 = generated.agent_configs["h1"]
    assert h1["neighbors"] == ["h0", "h2"]
    assert h1["schedule"] == []

    h2 = generated.agent_configs["h2"]
    assert h2["neighbors"] == ["h1"]
    assert h2["probes"] == [{"at_s": 9.0, "zone": "zone1"}]


def test_generate_non_publisher_record_content_name_is_per_node_and_distinct_from_the_publisher():
    """Generate non publisher record content name is per node and distinct from the publisher."""
    raw = _minimal_raw(
        hosts=3,
        links=[[0, 1], [1, 2]],
        agents=[0, 1],
        schedule={"1": [{"at_s": 3, "zone": "zone2", "disaster": "fire", "media": "sensor"}]},
    )
    spec = parse_experiment(raw)
    generated = generate(spec, "/abs/out/mini")
    assert generated.agent_configs["h1"]["schedule"][0]["content_name"] == "ccnx:/content/h1/body"


def test_generate_excludes_a_non_agent_host_from_neighbors():
    """A host on the link graph that is not in `agents` (an idle router) is never a sync neighbor."""
    raw = _minimal_raw(hosts=3, links=[[0, 1], [1, 2]], agents=[0, 1])
    spec = parse_experiment(raw)
    generated = generate(spec, "/abs/out/mini")
    assert generated.agent_configs["h1"]["neighbors"] == ["h0"]


def test_generate_carries_stop_at_s_and_counters_snapshot_at_s_into_every_agent_config():
    """Generate carries stop at s and counters snapshot at s into every agent config."""
    raw = _minimal_raw(hosts=3, links=[[0, 1], [1, 2]], agents=[0, 1], stop_at_s=130, counters_snapshot_at_s=[10])
    spec = parse_experiment(raw)
    generated = generate(spec, "/abs/out/mini")
    for node in ("h0", "h1"):
        assert generated.agent_configs[node]["stop_at_s"] == 130
        assert generated.agent_configs[node]["counters_snapshot_at_s"] == [10]


def test_parse_experiment_accepts_a_float_stop_at_s_and_an_empty_counters_snapshot_at_s():
    """Boundary values for the two new optional keys: a float stop_at_s, an empty snapshot list."""
    spec = parse_experiment(_minimal_raw(stop_at_s=130.0, counters_snapshot_at_s=[]))
    assert spec.stop_at_s == 130.0
    assert spec.counters_snapshot_at_s == ()


def test_parse_experiment_rejects_a_non_numeric_stop_at_s():
    """A non-numeric stop_at_s must raise, not be silently coerced.

    Exact-match, not `match=` substring (CLAUDE.md: mutation testing shows
    substring asserts let mutants through).
    """
    with pytest.raises(ExperimentError) as excinfo:
        parse_experiment(_minimal_raw(stop_at_s="130"))
    assert str(excinfo.value) == "stop_at_s must be a number"


def test_parse_experiment_rejects_a_non_list_counters_snapshot_at_s():
    """A non-list counters_snapshot_at_s must raise, not be silently wrapped.

    Exact-match, not `match=` substring (CLAUDE.md: mutation testing shows
    substring asserts let mutants through).
    """
    with pytest.raises(ExperimentError) as excinfo:
        parse_experiment(_minimal_raw(counters_snapshot_at_s=10))
    assert str(excinfo.value) == "counters_snapshot_at_s must be a list of numbers"


def test_generate_carries_fault_drop_rec_into_the_right_agent_config():
    """Generate carries fault drop rec into the right agent config."""
    raw = _minimal_raw(
        hosts=3,
        links=[[0, 1], [1, 2]],
        agents=[0, 1],
        fault={"0": {"drop_rec": [{"origin": "h0", "seq": 3, "times": 2}]}},
    )
    spec = parse_experiment(raw)
    generated = generate(spec, "/abs/out/mini")
    assert generated.agent_configs["h0"]["fault"] == {"drop_rec": [{"origin": "h0", "seq": 3, "times": 2}]}
    assert "fault" not in generated.agent_configs["h1"]


# -- write_experiment(): files on disk ----------------------------------------


def test_write_experiment_writes_a_yaml_that_is_valid_json_and_matching_node_configs(tmp_path: Path):
    """Write experiment writes a yaml that is valid json and matching node configs."""
    experiment_path = tmp_path / "exp.json"
    experiment_path.write_text(json.dumps(_minimal_raw()), encoding="utf-8")
    outdir = tmp_path / "out"

    written = write_experiment(str(experiment_path), str(outdir))

    yaml_path = outdir / "mini.yaml"
    assert str(yaml_path) in written
    text = yaml_path.read_text(encoding="utf-8")
    assert text.startswith("# Generated by semindex.gen")
    body = text.split("\n", 1)[1]
    cfg = json.loads(body)  # valid YAML-as-JSON: this must parse as plain JSON too
    assert cfg["k"] == 1

    h0_path = outdir / "mini" / "h0.json"
    assert str(h0_path) in written
    h0_config = json.loads(h0_path.read_text(encoding="utf-8"))
    assert h0_config["node"] == "h0"
    assert h0_config["neighbors"] == ["h1"]


def test_write_experiment_bakes_the_absolute_node_config_dir_into_host_apps_argv(tmp_path: Path):
    """Write experiment bakes the absolute node config dir into host apps argv."""
    experiment_path = tmp_path / "exp.json"
    experiment_path.write_text(json.dumps(_minimal_raw()), encoding="utf-8")
    outdir = tmp_path / "out"

    write_experiment(str(experiment_path), str(outdir))

    cfg = json.loads((outdir / "mini.yaml").read_text(encoding="utf-8").split("\n", 1)[1])
    argv = cfg["host_apps"][0]["argv"]
    config_flag_idx = argv.index("--config")
    assert argv[config_flag_idx + 1] == str((outdir / "mini" / "{node}.json").resolve())


# -- CLI -----------------------------------------------------------------


def test_main_exits_1_and_prints_the_error_on_an_invalid_experiment(tmp_path, capsys):
    """Main exits 1 and prints the error on an invalid experiment."""
    raw = _minimal_raw()
    del raw["hosts"]
    experiment_path = tmp_path / "bad.json"
    experiment_path.write_text(json.dumps(raw), encoding="utf-8")

    rc = main([str(experiment_path), str(tmp_path / "out")])

    assert rc == 1
    captured = capsys.readouterr()
    assert captured.err == "semindex.gen: missing required experiment key: 'hosts'\n"


def test_main_exits_0_and_prints_every_written_path(tmp_path, capsys):
    """Main exits 0 and prints every written path."""
    experiment_path = tmp_path / "exp.json"
    experiment_path.write_text(json.dumps(_minimal_raw()), encoding="utf-8")
    outdir = tmp_path / "out"

    rc = main([str(experiment_path), str(outdir)])

    assert rc == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert str(outdir / "mini.yaml") in lines
    assert str(outdir / "mini" / "h0.json") in lines
    assert str(outdir / "mini" / "h1.json") in lines
