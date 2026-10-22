"""Tests for the semindex.evalgen stage-1 evaluation-run generator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from semindex import evalgen, schema
from semindex.gen import ExperimentError, parse_experiment


# -- build_experiment: shape --------------------------------------------------


def test_build_experiment_has_the_fixed_five_host_line_topology():
    """Build experiment has the fixed five host line topology."""
    raw = evalgen.build_experiment(0.2, 7, "s1", "/out")
    assert raw["hosts"] == 5
    assert raw["links"] == [[0, 1], [1, 2], [2, 3], [3, 4]]
    assert raw["agents"] == [0, 1, 2, 3, 4]
    assert raw["publisher"]["host"] == 0
    # 2026-10-23 review r1 (finding 2): the submitted "duration" is the
    # measured window (DURATION_S, 120) plus a pure post-measurement
    # teardown cushion (TEARDOWN_MARGIN_S, 5) -- 125, not 120. See
    # TEARDOWN_MARGIN_S's comment in evalgen.py.
    assert raw["duration"] == 125


def test_build_experiment_puts_30_records_on_h0_and_h4_only():
    """Build experiment puts 30 records on h0 and h4 only."""
    raw = evalgen.build_experiment(0.2, 7, "s1", "/out")
    assert set(raw["schedule"].keys()) == {"0", "4"}
    assert len(raw["schedule"]["0"]) == 30
    assert len(raw["schedule"]["4"]) == 30


def test_build_experiment_probes_from_h2_only():
    """Build experiment probes from h2 only."""
    raw = evalgen.build_experiment(0.2, 7, "s1", "/out")
    assert set(raw["probes"].keys()) == {"2"}


def test_build_experiment_output_is_a_valid_gen_experiment_description():
    """gen.parse_experiment must accept evalgen's own output -- the whole point of reusing gen."""
    raw = evalgen.build_experiment(0.2, 7, "s1", "/out")
    spec = parse_experiment(raw)  # raises ExperimentError if the shape is wrong
    assert spec.hosts == 5


# -- record schedule -----------------------------------------------------


def test_record_schedule_is_2s_apart_starting_at_the_startup_margin():
    """Record schedule is 2s apart starting at the startup margin."""
    raw = evalgen.build_experiment(0.2, 7, "s1", "/out")
    at_s_values = [entry["at_s"] for entry in raw["schedule"]["0"]]
    assert at_s_values[0] == evalgen.RECORD_START_S
    assert at_s_values == [evalgen.RECORD_START_S + i * 2.0 for i in range(30)]


def test_record_schedule_attrs_are_within_the_fixed_vocabulary():
    """Record schedule attrs are within the fixed vocabulary."""
    raw = evalgen.build_experiment(0.2, 7, "s1", "/out")
    for entry in raw["schedule"]["0"] + raw["schedule"]["4"]:
        assert entry["zone"] in schema.ZONES
        assert entry["disaster"] in schema.DISASTERS
        assert entry["media"] in schema.MEDIA


# -- probe schedule --------------------------------------------------------


def test_probe_schedule_is_10s_apart_and_inside_the_run():
    """Probe schedule is 10s apart and inside the run."""
    raw = evalgen.build_experiment(0.2, 7, "s1", "/out")
    at_s_values = [entry["at_s"] for entry in raw["probes"]["2"]]
    assert at_s_values[0] == evalgen.PROBE_START_S
    for a, b in zip(at_s_values, at_s_values[1:]):
        assert b - a == evalgen.PROBE_INTERVAL_S
    assert all(0 <= at_s < evalgen.DURATION_S for at_s in at_s_values)


# -- determinism -----------------------------------------------------------


def test_build_experiment_is_identical_for_the_same_p_seed_stage_outdir():
    """Build experiment is identical for the same p seed stage outdir."""
    raw1 = evalgen.build_experiment(0.2, 7, "s1", "/out")
    raw2 = evalgen.build_experiment(0.2, 7, "s1", "/out")
    assert json.dumps(raw1, sort_keys=True) == json.dumps(raw2, sort_keys=True)


def test_build_experiment_differs_for_a_different_seed():
    """Build experiment differs for a different seed."""
    raw1 = evalgen.build_experiment(0.2, 7, "s1", "/out")
    raw2 = evalgen.build_experiment(0.2, 8, "s1", "/out")
    assert json.dumps(raw1["schedule"], sort_keys=True) != json.dumps(raw2["schedule"], sort_keys=True)
    assert json.dumps(raw1["events"], sort_keys=True) != json.dumps(raw2["events"], sort_keys=True)


def test_build_experiment_differs_for_a_different_p():
    """Build experiment differs for a different p."""
    raw1 = evalgen.build_experiment(0.2, 7, "s1", "/out")
    raw2 = evalgen.build_experiment(0.3, 7, "s1", "/out")
    assert json.dumps(raw1["events"], sort_keys=True) != json.dumps(raw2["events"], sort_keys=True)


def test_record_and_probe_schedules_are_the_same_regardless_of_p_at_the_same_seed():
    """2026-10-23 review r1 (finding 3, minor): only link_down/link_up events may change
    with p at a fixed seed -- record attrs and probe zones must not. Before the fix, p was
    part of every derived random stream's hash, so seed=7's h0 first record was zone1 at
    p=0.2 but zone5 at p=0.3: a comparison meant to isolate the disconnect rate alone also
    silently changed the simulated load."""
    raw1 = evalgen.build_experiment(0.2, 7, "s1", "/out")
    raw2 = evalgen.build_experiment(0.3, 7, "s1", "/out")
    assert json.dumps(raw1["schedule"], sort_keys=True) == json.dumps(raw2["schedule"], sort_keys=True)
    assert json.dumps(raw1["probes"], sort_keys=True) == json.dumps(raw2["probes"], sort_keys=True)


def test_generate_run_writes_byte_identical_files_on_a_second_call(tmp_path: Path):
    """2026-10-23: proves "same (p, seed) => byte-identical output" against the actual files
    on disk, not just the in-memory dict -- generating into the same outdir twice must leave
    every written file's bytes unchanged."""
    outdir = tmp_path / "run"
    written = evalgen.generate_run(0.2, 7, "s1", str(outdir))
    before = {path: Path(path).read_bytes() for path in written}

    written_again = evalgen.generate_run(0.2, 7, "s1", str(outdir))
    assert written_again == written
    after = {path: Path(path).read_bytes() for path in written_again}
    assert before == after


def test_cli_output_is_byte_identical_across_separate_processes(tmp_path: Path):
    """The in-process test above can't catch a determinism bug tied to Python's own
    per-process string-hash randomization (PYTHONHASHSEED) -- exactly the hazard
    `_record_schedule`'s own docstring guards against by drawing from `sorted(schema.ZONES)`
    rather than the frozenset or a plain `list()` of it. Running the CLI as two separate
    interpreter processes (each with its own, independently-randomized hash seed) is what
    actually exercises that guard."""
    outdir = tmp_path / "run"
    cmd = [
        sys.executable,
        "-m",
        "semindex.evalgen",
        "--p",
        "0.2",
        "--seed",
        "7",
        "--stage",
        "s1",
        "--out",
        str(outdir),
    ]
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")}

    result1 = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)
    assert result1.returncode == 0, result1.stderr
    before = {path: Path(path).read_bytes() for path in result1.stdout.splitlines()}

    result2 = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)
    assert result2.returncode == 0, result2.stderr
    after = {path: Path(path).read_bytes() for path in result2.stdout.splitlines()}

    assert before == after


# -- disconnect probability: must be a finite value in [0, 1] ----------------
#
# 2026-10-23 review r1 (finding 4, minor): build_experiment used to accept
# any float for p. p=-0.1/NaN silently drew "never down" (`rng.random() < p`
# is False for every draw) and p=1.1/+inf silently drew "always down" (True
# for every draw) -- a typo'd p produced a normal-looking, fully-valid
# experiment whose condition label did not match its actual behavior.


@pytest.mark.parametrize("bad_p", [-0.1, 1.1, float("nan"), float("inf"), float("-inf")])
def test_build_experiment_rejects_p_outside_zero_one(bad_p: float):
    """An out-of-range or non-finite p must raise, not silently clamp."""
    with pytest.raises(ExperimentError, match=r"p must be"):
        evalgen.build_experiment(bad_p, 7, "s1", "/out")


@pytest.mark.parametrize("good_p", [0.0, 0.2, 1.0])
def test_build_experiment_accepts_p_inside_zero_one(good_p: float):
    """The boundary values 0.0 and 1.0, and an ordinary mid-range p, must still be accepted."""
    evalgen.build_experiment(good_p, 7, "s1", "/out")  # must not raise


# -- disconnect schedule: p = 0 ----------------------------------------------


def test_p_zero_produces_zero_link_events():
    """Build experiment at p=0 produces zero link events."""
    raw = evalgen.build_experiment(0.0, 7, "s1", "/out")
    assert raw["events"] == []


# -- disconnect schedule: shape and ordering ----------------------------------


def _down_intervals(events: list[dict], nodes: list[int]) -> list[tuple[float, float]]:
    """Pair up one link's link_down/link_up events, in the order they appear, as (down_at, up_at)."""
    link_events = [e for e in events if e["nodes"] == nodes]
    assert [e["type"] for e in link_events] == ["link_down", "link_up"] * (len(link_events) // 2)
    return [(link_events[i]["at"], link_events[i + 1]["at"]) for i in range(0, len(link_events), 2)]


def test_link_events_are_sorted_by_at():
    """Link events are sorted by at."""
    raw = evalgen.build_experiment(0.3, 7, "s1", "/out")
    at_values = [e["at"] for e in raw["events"]]
    assert at_values == sorted(at_values)


def test_every_link_up_follows_its_link_down_and_every_down_is_restored_by_the_end():
    """Every link up follows its link down and every down is restored by the end."""
    raw = evalgen.build_experiment(0.3, 7, "s1", "/out")
    for a, b in evalgen.LINKS:
        for down_at, up_at in _down_intervals(raw["events"], [a, b]):
            assert down_at < up_at
            assert up_at <= evalgen.DISCONNECT_WINDOW_S


def test_link_events_stay_inside_the_measured_window_with_teardown_margin_to_spare():
    """Link events stay inside the measured window, with a teardown margin to spare.

    2026-10-23 review r1 (finding 2): disconnect slots are now drawn over
    the FULL measured window (DISCONNECT_WINDOW_S == DURATION_S, all
    NUM_SLOTS=24 of them) -- no event, even at p=0.4's worst case, ever
    lands past DISCONNECT_WINDOW_S. The teardown cushion this run still
    needs (so a restoring link_up at the very end can't race disaster.py's
    teardown sleep) now comes entirely from the submitted "duration" being
    TEARDOWN_MARGIN_S longer than DISCONNECT_WINDOW_S, not from shrinking
    the window p is drawn over (see TEARDOWN_MARGIN_S's comment in
    evalgen.py for why the earlier version's shrunk window was wrong).
    """
    raw = evalgen.build_experiment(0.4, 11, "s1", "/out")
    for e in raw["events"]:
        assert 0 <= e["at"] <= evalgen.DISCONNECT_WINDOW_S
    assert raw["duration"] == evalgen.DISCONNECT_WINDOW_S + evalgen.TEARDOWN_MARGIN_S


# -- disconnect schedule: statistical bounds ----------------------------------
#
# Each link/slot draw is an independent Bernoulli(p) (evalgen._link_events'
# own docstring). Over `NUM_SEEDS` independent seeds, `len(LINKS)` links and
# `evalgen.NUM_SLOTS` slots per link, the observed down-fraction is the mean
# of NUM_SEEDS * len(LINKS) * NUM_SLOTS i.i.d. Bernoulli(p) trials. Its
# standard error is sqrt(p*(1-p)/n); the worst case over the tested p values
# (p=0.4, variance 0.24) with n = 150 * 4 * 24 = 14400 gives SE ~= 0.0041. The
# tolerance below (0.03) is about 7 SE, so a correct implementation fails
# this bound with negligible probability while a broken one (e.g. a biased
# or constant-probability draw) is still caught.
#
# 2026-10-23 review r1 (finding 2): the denominator is DISCONNECT_WINDOW_S,
# which now equals the full DURATION_S (24 slots, not a shrunk 23) -- p is
# drawn over the whole measured run, with no excluded tail. The earlier
# version's DISCONNECT_WINDOW_S was DURATION_S minus a safety margin, which
# would have biased every observed fraction low by a fixed factor
# regardless of how correct the draw itself was; the margin now lives
# entirely in the gap between DISCONNECT_WINDOW_S and the submitted
# "duration" (TEARDOWN_MARGIN_S), outside what this statistic measures.
NUM_SEEDS = 150


@pytest.mark.parametrize("p", [0.1, 0.2, 0.3, 0.4])
def test_mean_down_fraction_over_many_seeds_approaches_p(p: float):
    """Mean down fraction over many seeds approaches p."""
    total_down_s = 0.0
    total_possible_s = 0.0
    for seed in range(NUM_SEEDS):
        raw = evalgen.build_experiment(p, seed, "stat", "/out")
        for a, b in evalgen.LINKS:
            for down_at, up_at in _down_intervals(raw["events"], [a, b]):
                total_down_s += up_at - down_at
            total_possible_s += evalgen.DISCONNECT_WINDOW_S
    observed_fraction = total_down_s / total_possible_s
    assert abs(observed_fraction - p) < 0.03


# -- attrs: uniform-ish over many draws ---------------------------------------
#
# Same statistical reasoning as above: aggregating RECORD_COUNT=30 draws over
# NUM_SEEDS=150 seeds gives n=4500 i.i.d. draws per vocabulary; for zone
# (5 values, p=0.2 each) the worst-case SE is sqrt(0.2*0.8/4500) ~= 0.006, so
# a 0.05 tolerance is about 8 SE.


@pytest.mark.parametrize(
    ("field", "vocabulary"),
    [("zone", schema.ZONES), ("disaster", schema.DISASTERS), ("media", schema.MEDIA)],
)
def test_attr_draws_are_uniform_ish_across_many_seeds(field: str, vocabulary: frozenset[str]):
    """Attr draws are uniform ish across many seeds."""
    counts = {value: 0 for value in vocabulary}
    total = 0
    for seed in range(NUM_SEEDS):
        raw = evalgen.build_experiment(0.2, seed, "stat", "/out")
        for entry in raw["schedule"]["0"]:
            counts[entry[field]] += 1
            total += 1
    expected_fraction = 1.0 / len(vocabulary)
    for value, count in counts.items():
        assert abs(count / total - expected_fraction) < 0.05


# -- generate_run: files on disk ----------------------------------------------


def test_generate_run_writes_the_experiment_description_and_gens_own_files(tmp_path: Path):
    """Generate run writes the experiment description and gens own files."""
    outdir = tmp_path / "run"
    written = evalgen.generate_run(0.2, 7, "s1", str(outdir))

    name = "s1_p0.2_seed7"
    experiment_path = outdir / f"{name}.experiment.json"
    yaml_path = outdir / f"{name}.yaml"
    h0_path = outdir / name / "h0.json"
    assert str(experiment_path) in written
    assert str(yaml_path) in written
    assert str(h0_path) in written

    raw = json.loads(experiment_path.read_text(encoding="utf-8"))
    assert raw["name"] == name

    h0_config = json.loads(h0_path.read_text(encoding="utf-8"))
    assert h0_config["node"] == "h0"


# -- generate_run: rejected input must leave no trace on disk ----------------
#
# 2026-10-23 review r1 (finding 1, major): generate_run used to write
# `<outdir>/<name>.experiment.json` BEFORE gen.write_experiment's internal
# `gen.parse_experiment(raw)` call validated `raw["name"]`. A `stage` that
# makes `name` escape `outdir` (a relative "../" or an absolute path) was
# still rejected in the end, but only after that escaping write had already
# landed on disk. These two tests reproduce each shape and prove nothing is
# written once generate_run validates first.


def test_generate_run_writes_nothing_for_a_stage_that_escapes_outdir(tmp_path: Path):
    """A stage containing '..' must not write even the escaping file before raising."""
    outdir = tmp_path / "run"
    escaped_path = tmp_path / "escaped_p0.2_seed7.experiment.json"

    with pytest.raises(ExperimentError, match=r"name must not be a path"):
        evalgen.generate_run(0.2, 7, "../escaped", str(outdir))

    assert not escaped_path.exists()
    assert not outdir.exists()


def test_generate_run_writes_nothing_for_an_absolute_stage(tmp_path: Path):
    """An absolute-path stage must be rejected before any write is attempted.

    Before the fix this reached `open("/abs_p0.2_seed7.experiment.json", "w")`
    (os.path.join discards its first argument once the second is absolute) --
    a PermissionError in this sandbox, but a write to an attacker/typo
    -controlled absolute path anywhere writable. The fix must raise
    ExperimentError instead, before that open() is ever reached.
    """
    outdir = tmp_path / "run"

    with pytest.raises(ExperimentError, match=r"name must not be a path"):
        evalgen.generate_run(0.2, 7, "/abs", str(outdir))

    assert not outdir.exists()


# -- CLI -----------------------------------------------------------------


def test_main_exits_0_and_prints_every_written_path(tmp_path: Path, capsys):
    """Main exits 0 and prints every written path."""
    outdir = tmp_path / "run"
    rc = evalgen.main(["--p", "0.2", "--seed", "7", "--stage", "s1", "--out", str(outdir)])
    assert rc == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert str(outdir / "s1_p0.2_seed7.yaml") in lines


def test_main_exits_1_and_prints_the_error_on_a_stage_that_escapes_the_outdir(tmp_path: Path, capsys):
    """main()'s error path: a stage label becomes part of `name`, so one containing a path
    separator reaches gen.parse_experiment's existing path-escape check (test_gen.py's own
    ".. /escaped" case) -- evalgen must surface that ExperimentError the same way gen.main does,
    not let it propagate as an uncaught exception."""
    rc = evalgen.main(["--p", "0.2", "--seed", "7", "--stage", "../escaped", "--out", str(tmp_path / "run")])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.err == "semindex.evalgen: name must not be a path: '../escaped_p0.2_seed7'\n"


# -- ReCefore validator: the generated YAML must pass ReCefore's own config validator ---

_RECEFORE_VENV_PYTHON = "/home/lab_shared/cefore-emu-sslab/.venv/bin/python3"
_RECEFORE_SRC = "/home/lab_shared/cefore-emu-sslab/src"


def test_generated_yaml_passes_recefore_config_validator(tmp_path: Path):
    """Generated yaml passes ReCefore config validator.

    Runs ReCefore's own loader+validator, in ReCefore's own venv, against a
    freshly generated run's YAML -- read-only (no sudo, no mininet):
    automates the check that would otherwise be a one-off manual step.
    """
    if not os.path.exists(_RECEFORE_VENV_PYTHON):
        pytest.skip(f"ReCefore venv not found at {_RECEFORE_VENV_PYTHON}")
    outdir = tmp_path / "run"
    written = evalgen.generate_run(0.2, 7, "s1", str(outdir))
    yaml_path = [p for p in written if p.endswith(".yaml")][0]

    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from core.config.loader import load_config\n"
        "from core.config.validator import validate_config\n"
        "cfg = load_config(%r)\n"
        "errors = validate_config(cfg)\n"
        "print(errors)\n"
    ) % (_RECEFORE_SRC, yaml_path)

    result = subprocess.run(
        [_RECEFORE_VENV_PYTHON, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
