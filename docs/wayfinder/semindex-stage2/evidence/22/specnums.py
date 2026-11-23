#!/usr/bin/env python3
"""Every number `spec.md` states that is read off the label files and nothing else prints.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B specnums.py

Writes its whole printed output to specnums.txt.  The two label files are its only source
of data; it runs no simulation, so it takes seconds and may be rerun after any edit to
this directory's prose.  2026-11-23: it imports `label` and `stability` for two
definitions and no data -- the paired t and the bin edge the selection in
`retention_facts` is defined by -- because a copy of either here would be a second
definition of the thing being selected on.

WHY THIS FILE EXISTS
  2026-11-23.  `labels.txt` carries the counts the labelling itself printed, `observe.txt`
  the record's sizes and the polled-peer facts, `stability.txt` the agreement tables.  A
  class of numbers sat outside all three: the median margin over the LABELLED rows, the
  share of labels decided by one poll of the weight or less and the floating-point band
  that share lands in, how many labels the weight correction moved, the counts of the
  render branches a reader is told are live or dead, and the state situations the labelling
  reaches.  Those were measured once by hand, quoted in `spec.md`, and re-derived by
  nothing.  Everything here is recomputed from the files on disk at every run, so a number
  in the prose that no longer matches the data is a diff in this file's output.
  2026-11-23 added two more of the same kind: the decisions the replay makes at or after
  the instant a live agent stops (`post_stop_facts`), and how much of a cell the selection
  `stability.py` hands a learner keeps over all 100 seeds (`retention_facts`, a retention
  and not an agreement -- an agreement would need a second labelling, which no file on
  disk carries for seeds beyond the ten `stability.py` ran).
  2026-11-23, the same day and the same kind again: the rows whose paired t lands ON the
  bar that selection cuts at, counted in `retention_facts` beside the retention itself.
  The bar sits on a mass point of the statistic, so the retention above is a count that
  floating-point rounding settles for a few dozen rows of each cell, and a t recomputed
  elsewhere gives a different one. The count of those rows, which side the
  float put each on, and how far the nearest row off the mass point lies are printed, so
  the size of that slack is a number in this file and not a worry nobody measured.

  It asserts nothing about the values.  Whether a cell reaches a situation or a branch is a
  property of the workload, and a script that failed when a count moved would be refusing a
  different workload rather than finding a fault.  The one thing it does refuse is a label
  file whose bytes are not the ones `labels.sha256` records: then the numbers below would
  describe a labelling nobody can name, and the script stops instead of printing them.

THE SUPERSEDED LABELS
  The weight correction's cost (how many labels moved) needs the labels as they were under
  the superseded weight, and those files are not kept anywhere a reader can reach.  They do
  not have to be: a label row carries `pass_scores` (the per-pass score, delay + weight x
  polls) and `polls` (that pass's poll count) per arm, so the per-pass delay comes back as
  `score - weight x polls` and the superseded score as `delay + superseded weight x polls`.
  That arithmetic is not bit-exact -- the subtraction loses up to about 3e-11 ms of the
  score -- and the loss differs between two arms because their poll counts differ, so an
  EXACT tie under the superseded weight comes back as a margin of a few 1e-12 instead of
  0.0.  `TIE_TOL` is the tolerance that reads those back as the ties they were.
  2026-11-23: the reconstruction was checked row by row against the real superseded files
  while they still existed, over all 16071 + 16148 rows of the two cells, and gave the same
  label on every row and the same seven-way direction breakdown.  The tolerance is printed
  with the number of margins that sit close enough to it to depend on it, so that a later
  run on other data says whether the tolerance is deciding anything.
"""
import gzip
import hashlib
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import observe as OBS  # noqa: E402
# 2026-11-23, for `retention_facts` and nothing else: the paired t and the bin edge the
# selection uses are `label.paired_t` and `stability.T_BINS`, and a second copy of either
# here would be a second definition of the thing the selection is defined by.  Neither
# import runs a simulation; `label` reads `weight.json` at import, which this file reads
# anyway, and `stability`'s module body builds its seed list and nothing else.
import label as L  # noqa: E402
import stability as ST  # noqa: E402

ARMS = [3.695, 6.0, 10.043]
CELLS = ((0.2, "labels_p0.2.jsonl.gz"), (0.4, "labels_p0.4.jsonl.gz"))
# The seeds `stability.py` labels twice, so that the retention below can be read over the
# same rows its agreement is measured on as well as over the whole file.
STABILITY_SEEDS = frozenset(ST.SEEDS)
LEARN = "/home/lab_shared/semindex-learn/ev22"
SHA_FILE = os.path.join(HERE, "labels.sha256")
OUT_TXT = os.path.join(HERE, "specnums.txt")

# The relative tolerance of the "at or below one poll's weight" share.  The comparison
# lands ON the weight, so an exact `<=` cuts a cluster of margins that are the weight to
# the last bit; the share is reported both ways and the band between them printed.
ATOM_TOL = 1e-9
# 2026-11-23.  How close a paired t has to be to the selection bar to count as sitting ON
# it, in units of t, which is dimensionless.  Absolute and not relative, because what is
# being measured is the distance to one fixed bar and not a spread around a scale.  The
# number itself is loose by seven orders of magnitude: the rows it catches are exactly
# on the bar in real arithmetic and sit within 1e-13 of it in doubles, and the nearest row
# off the mass point is printed beside them, so the tolerance can be read as deciding
# nothing rather than taken on trust.
BAR_TOL = 1e-6
# The tie tolerance of the reconstructed superseded label, in ms of score.  Three orders of
# magnitude above the reconstruction's own error, which is about 3e-11 ms.
#
# 2026-11-23: there is NO gap in the margin distribution to put it in.  `label.py` calls a
# decision a tie only when the margin is exactly 0.0, and the files carry labelled rows whose
# margin is a few 1e-12 -- ulp-scale residue of its own `fmean` over the passes -- so rows do
# exist on both sides of this tolerance.  What licenses it here is not a gap but a check: the
# reconstruction was compared row by row against the real superseded files while they were
# still in reach, and gave the same label on all 16071 + 16148 rows.  `moved_facts` prints how
# many labelled rows stand on a margin below the tolerance, because each of those is a label
# this reconstruction cannot carry from one weight to the other.
TIE_TOL = 1e-9

LINES = []


def out(text=""):
    """Print and keep, so specnums.txt is exactly what the terminal showed."""
    print(text, flush=True)
    LINES.append(text)


def digests():
    """{path: digest} from labels.sha256, in the file's own order."""
    want = {}
    with open(SHA_FILE, encoding="utf-8") as fh:
        for line in fh:
            digest, path = line.split()
            want[path] = digest
    return want


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows_of(path):
    """Every label row of one cell, in the file's order."""
    with gzip.open(path, "rb") as fh:
        return [json.loads(line) for line in fh]


def pick(means):
    """`label.py`'s own choice rule, on an arm -> mean score map.

    Written again here rather than imported, and kept that way after `label` became an
    import of this file: this copy IS a check.  The label this rule picks from the row's
    own `pass_scores` is compared against the row's stored `label` below, on every row of
    both cells, so a drift between the rule that wrote the files and the rule stated here
    is printed as a fault.  `paired_t` gets the opposite treatment -- imported, not
    copied -- because no column of the file can check a t against anything.
    """
    order = sorted(ARMS, key=lambda a: (means[a], ARMS.index(a)))
    margin = means[order[1]] - means[order[0]]
    return (None if margin == 0.0 else order[0]), margin


def superseded_label(row, w_new, w_old):
    """(the label under the superseded weight, its margin) from the row's own passes.

    See the module docstring: the delay of one pass is `score - w_new x polls` and the
    superseded score `delay + w_old x polls`.  A margin at or below `TIE_TOL` is read as
    the tie it was before the arithmetic above lost its last bits.
    """
    means = {}
    for arm in ARMS:
        scores, polls = row["pass_scores"][repr(arm)], row["polls"][repr(arm)]
        means[arm] = statistics.fmean(s - w_new * n + w_old * n
                                      for s, n in zip(scores, polls))
    order = sorted(ARMS, key=lambda a: (means[a], ARMS.index(a)))
    margin = means[order[1]] - means[order[0]]
    return (None if margin <= TIE_TOL else order[0]), margin


def name_of(arm):
    return "tie" if arm is None else f"{arm} s"


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------

def margin_facts(rows, weight):
    """The margin numbers: the two medians, the atom on the weight, the two shares."""
    margins = [r["margin"] for r in rows]
    labelled = [r["margin"] for r in rows if r["label"] is not None]
    ties = len(rows) - len(labelled)
    band = ATOM_TOL * weight
    out(f"  decisions {len(rows)}, labelled {len(labelled)}, ties {ties}")
    out("  median margin over ALL decisions      %.4f ms" % statistics.median(margins))
    out("  median margin over the LABELLED rows  %.4f ms  <- a tie enters the first "
        "median at 0 and not the second" % statistics.median(labelled))
    out("  margins exactly equal to the weight: %d; within %g relative of it: %d of the "
        "%d (the atom the median sits in)"
        % (sum(1 for m in margins if m == weight), ATOM_TOL,
           sum(1 for m in margins if abs(m - weight) <= band), len(rows)))
    exact = sum(1 for m in labelled if m <= weight)
    tol = sum(1 for m in labelled if m <= weight + band)
    below = sum(1 for m in labelled if m < weight and weight - m <= band)
    above = sum(1 for m in labelled if m > weight and m - weight <= band)
    out("  labelled rows at or below one poll's weight: %d exactly (%.4f), %d with the "
        "%g tolerance (%.4f)" % (exact, exact / len(labelled), tol, ATOM_TOL,
                                 tol / len(labelled)))
    out("    of which the band on the weight itself: %d a last bit BELOW it and %d a last "
        "bit ABOVE, so the exact `<=` cuts that cluster and the tolerant one keeps it"
        % (below, above))
    out("  the same over ALL decisions, a tie counting as inside: %.4f exactly, %.4f "
        "tolerant" % ((exact + ties) / len(rows), (tol + ties) / len(rows)))


def moved_facts(rows, w_new, w_old):
    """How many labels the weight correction moved, and in which direction."""
    moved, wrong_new, close, thin = 0, 0, 0, 0
    direction = {}
    for row in rows:
        means = {arm: statistics.fmean(row["pass_scores"][repr(arm)]) for arm in ARMS}
        again, _m = pick(means)
        if again != row["label"]:
            wrong_new += 1
        if 0.0 < row["margin"] <= TIE_TOL:
            thin += 1
        old, old_margin = superseded_label(row, w_new, w_old)
        if TIE_TOL < old_margin <= TIE_TOL * 1000.0:
            close += 1
        if old != row["label"]:
            moved += 1
            key = (name_of(old), name_of(row["label"]))
            direction[key] = direction.get(key, 0) + 1
    out("  labels that changed against the superseded weight %.17g: %d of %d (%.4f)"
        % (w_old, moved, len(rows), moved / len(rows)))
    for (src, dst), n in sorted(direction.items(), key=lambda kv: (-kv[1], kv[0])):
        out(f"    {src} -> {dst}: {n}")
    out("  the current weight's label recomputed from the row's own `pass_scores` fails "
        "to reproduce the stored label on %d of the %d rows; any count above 0 means the "
        "file's labels are not the ones its own passes give, which would make the "
        "reconstruction above meaningless" % (wrong_new, len(rows)))
    out("  reconstructed superseded margins above the %g ms tie tolerance but within a "
        "thousand times it: %d" % (TIE_TOL, close))
    out("  labels this file's own rows stand on with a margin at or below that tolerance: "
        "%d. The labelling calls a decision a tie only at a margin of exactly 0.0, so "
        "these rows carry a label decided by the last bits of a mean over the passes, and "
        "the reconstruction above cannot carry such a label from one weight to the other"
        % thin)


def drop_facts(rows):
    """The other scoring choice, for the cross-check against labels.txt's own line."""
    moved = sum(1 for r in rows if r["label_drop_censored"] != r["label"])
    out("  labels that move when the censored records are DROPPED instead of entered at "
        "their lower bound: %d of %d (%.4f) -- labels.txt prints this line too, and the "
        "two have to agree" % (moved, len(rows), moved / len(rows)))


def post_stop_facts(rows):
    """The decisions the replay makes at or after the instant a live agent stops.

    2026-11-23.  A live agent stops at `t0 + replay.STOP_AT_MS`; the replay goes on
    resolving polls that were already in flight at that instant, so the labelling probes
    decisions no testbed makes.  The rows stay in the file and no label moves: a poll
    scheduled after that instant falls outside the scoring window, so the three arms score
    identically and the row carries no label at all.  That is what the counts below say,
    and it is why a relabelling would move no labelled row.  Counted per cell so that a
    reader filtering on `ts_rel_ms` knows how many rows the filter takes.
    """
    late = [r for r in rows if r["ts_rel_ms"] >= OBS.R.STOP_AT_MS]
    labelled = [r for r in late if r["label"] is not None]
    nonzero = [r for r in late if r["margin"] != 0.0]
    out("  decisions at or after ts_rel_ms %.0f, the instant a live agent stops and the "
        "replay keeps resolving past: %d of %d%s. %d of them carry a label and %d have a "
        "margin other than exactly 0.0, so %s"
        % (OBS.R.STOP_AT_MS, len(late), len(rows),
           ", the latest at %.0f" % max(r["ts_rel_ms"] for r in late) if late else "",
           len(labelled), len(nonzero),
           "every one of them is an exact tie and no labelled row sits there"
           if not labelled and not nonzero else
           "a labelled row sits past the stop and a filter on ts_rel_ms would drop it"))


def retention_facts(rows, seeds_of_stability):
    """What `stability.py`'s t selection keeps over this cell's own rows, retention only.

    `stability.py` measures that selection's AGREEMENT on seeds 1 to 10, where every
    decision is labelled twice from two disjoint sets of 8 passes.  Nothing on disk
    carries a second labelling of seeds 1 to 100, so no agreement can be computed here.
    What can be computed is how much of the file the selection keeps, and that is the
    number a reader of the selection needs beside an agreement measured on ten seeds.

    The labelling on disk IS set A of that measurement -- the same 8 passes, keyed the
    same way -- so `label.paired_t` over a row's own `pass_scores` is the same statistic
    the breakdown bins on.  The selection is `t >= 3 or (sd = 0 and labelled)`: a row whose
    passes all scored the two best arms equally has `sd = 0` and no label, and keeping it
    would keep a row no learner can train on.

    2026-11-23.  The same loop counts what that retention is worth to the row, because the
    bar cuts a mass point of the t.  A pattern of per-pass differences as plain as seven
    passes at +a and one at -a gives mean 6a/8 against a sample sd of a/sqrt(2), so t is
    exactly 3 in real arithmetic, and the whole-multiples-of-one-poll shape these cells are
    full of lands on it too.  In doubles those rows sit within 1e-13 either side of 3 and
    `>=` splits them by the order the sums happened to run in.  What is printed is how many
    there are, how the float split them, the range the retention could take and the two
    numbers that say the clump is isolated.  The runner-up tie is counted beside it: when
    the second and third arm have the same mean, the arm the t is computed against is
    picked by the order of `ARMS`, and the selection moves on some of those rows if the
    other arm is taken instead.
    """
    bar = ST.T_BINS[-1][0]
    kept = {True: [0, 0], False: [0, 0]}      # [kept rows, kept rows carrying a label]
    labelled = {True: 0, False: 0}
    total = {True: 0, False: 0}
    on_bar = {True: [], False: []}            # (|t - bar|, kept) per row sitting ON the bar
    outside = {True: None, False: None}       # nearest |t - bar| of a row NOT on the bar
    tied = {True: [0, 0], False: [0, 0]}      # [runner-up ties, of them with no t]
    flips = {True: 0, False: 0}               # ties the other runner-up moves across
    for row in rows:
        # Every row counts into the whole-cell tally (key False) and a row of the ten
        # measured seeds into that tally as well (key True), so the two populations come
        # out of one pass over the file.
        ten = row["seed"] in seeds_of_stability
        has = row["label"] is not None
        idx = range(len(row["pass_scores"][repr(ARMS[0])]))
        t, sd, _n = L.paired_t(row["pass_scores"], idx)
        take = (t is not None and t >= bar) or (t is None and sd == 0.0 and has)
        means = {arm: statistics.fmean(row["pass_scores"][repr(arm)]) for arm in ARMS}
        order = sorted(ARMS, key=lambda a: (means[a], ARMS.index(a)))
        runner_up_tie = means[order[1]] == means[order[2]]
        flipped = False
        if runner_up_tie:
            # Sending the runner-up's own scores to infinity leaves the winner where it is
            # and makes the third arm the runner-up, so the imported `paired_t` forms the
            # pairing the other `ARMS` order would have formed.  Computing that t here
            # instead would put a second definition of the statistic in this file.
            other = dict(row["pass_scores"])
            other[repr(order[1])] = [float("inf")] * len(other[repr(order[1])])
            t2, sd2, _n2 = L.paired_t(other, idx)
            take2 = (t2 is not None and t2 >= bar) or (t2 is None and sd2 == 0.0 and has)
            flipped = take2 != take
        dist = None if t is None else abs(t - bar)
        for key in {ten, False}:
            total[key] += 1
            labelled[key] += has
            if take:
                kept[key][0] += 1
                kept[key][1] += has
            if runner_up_tie:
                tied[key][0] += 1
                tied[key][1] += t is None
                flips[key] += flipped
            if dist is None:
                continue
            if dist <= BAR_TOL:
                on_bar[key].append((dist, take))
            elif outside[key] is None or dist < outside[key]:
                outside[key] = dist
    for key, what in ((False, "this cell's own %d rows" % total[False]),
                      (True, "the seeds %d to %d of it, which stability.py measured"
                       % (min(seeds_of_stability), max(seeds_of_stability)))):
        if not total[key]:
            continue
        n, n_lab = kept[key]
        out("  t >= %g or (sd = 0 and labelled), over %s: keeps %d rows, %d of them "
            "carrying a label (%.4f of all the rows and %.4f of the %d labelled ones). "
            "RETENTION ONLY -- the agreement of this selection needs a second labelling "
            "from disjoint passes and the files on disk are one labelling."
            % (bar, what, n, n_lab, n_lab / total[key], n_lab / labelled[key],
               labelled[key]))
        atoms = on_bar[key]
        on_kept = sum(1 for _d, take in atoms if take)
        if atoms:
            # The clause about the tolerance follows the two distances instead of being
            # asserted over them.  What makes the tolerance decide nothing is room on BOTH
            # sides: the clump a hundredfold inside it and the nearest t off the clump a
            # hundredfold outside, so a BAR_TOL a hundred times larger or smaller would
            # catch the same rows.  Both factors are printed, and on label files where the
            # room is not there the line says the tolerance is deciding the count.
            inner = max(d for d, _k in atoms)
            room = (inner * 100.0 <= BAR_TOL
                    and (outside[key] is None or outside[key] >= 100.0 * BAR_TOL))
            out("    of those rows %d have a t within %g of the bar %g, and the float "
                "comparison keeps %d of them and drops %d. A t of exactly %g is reachable "
                "in real arithmetic: seven passes at +a and one at -a give mean 6a/8 "
                "against a sample sd of a/sqrt(2), and per-pass differences in whole "
                "multiples of one poll's weight can land on it too, so the statistic has "
                "a mass point where the bar is. The largest |t - %g| among these rows is "
                "%g, which is %.3g of the tolerance, and the nearest |t - %g| outside the "
                "tolerance is %s, so %s."
                % (len(atoms), BAR_TOL, bar, on_kept, len(atoms) - on_kept, bar,
                   bar, inner, inner / BAR_TOL, bar,
                   "%g, which is %.3g times it" % (outside[key], outside[key] / BAR_TOL)
                   if outside[key] is not None
                   else "none, every t of this population sitting on the bar",
                   "the clump is isolated and a tolerance a hundred times wider or "
                   "narrower would catch the same rows" if room else
                   "the tolerance has no room on one of its sides and is itself deciding "
                   "which rows these counts cover"))
            out("    SO THE COUNT ABOVE IS NOT REPRODUCIBLE ROW BY ROW, which is a "
                "weakness of the bar as this step defined it: `t >= %g` cuts a mass point "
                "of the statistic instead of a gap in it. The selection would hold %d "
                "rows if none of those %d were kept and %d if all of them were, and which "
                "side a row falls on is settled by the order its sums ran in. A reader who "
                "recomputes the t anywhere else lands somewhere inside that range and not "
                "on this number."
                % (bar, n - on_kept, len(atoms), n - on_kept + len(atoms)))
        else:
            out("    no row of this population has a t within %g of the bar %g, so the "
                "count above does not turn on a floating-point comparison" % (BAR_TOL, bar))
        out("    rows whose runner-up ties the THIRD arm in mean, where the order of "
            "`ARMS` and nothing in the data picks which arm the t is taken against: %d, "
            "%d of them with no t at all. Taking the other arm of the tie instead moves "
            "%d of them across the selection, so that many rows of the count above rest "
            "on this directory's arm order." % (tied[key][0], tied[key][1], flips[key]))


def situation_facts(rows, records):
    """The state situations the labelling reaches, and the three no run of it reaches."""
    tmo = [r for r in rows if r["state"]["last_timeout_s"] is not None]
    per_seed = {}
    for r in tmo:
        per_seed[r["seed"]] = per_seed.get(r["seed"], 0) + 1
    out("  last_timeout_s set: %d of %d, on seeds %s"
        % (len(tmo), len(rows),
           ", ".join(f"{s} ({n})" for s, n in sorted(per_seed.items())) or "none"))
    ages = [rec["fetch_timeout_age_s"] for rec in records
            if rec["fetch_timeout_age_s"] is not None]
    out("  the fetch_timeout_age_s those stamps render as: %d values, %s"
        % (len(ages),
           "%.1f s to %.1f s" % (min(ages), max(ages)) if ages else "none"))
    cells = above = 0
    above_seeds = {}
    for r in rows:
        for origin in r["state"]["origins"].values():
            for adv in origin["advertised"].values():
                cells += 1
                if adv["notified_max"] > adv["hi"]:
                    above += 1
                    above_seeds[r["seed"]] = above_seeds.get(r["seed"], 0) + 1
    out("  notified_max above the advertiser's own hi: %d of %d advertised cells, on "
        "seeds %s" % (above, cells,
                      ", ".join(f"{s} ({n})" for s, n in sorted(above_seeds.items()))
                      or "none"))
    for field, want, how in (("in_flight", lambda v: v > 0, "above 0"),
                             ("truncated", bool, "true"),
                             ("last_poll_status", lambda v: v == "pending", '"pending"')):
        n = sum(1 for r in rows if want(r["state"][field]))
        # 2026-11-23: the clause follows the count instead of being printed over it. The
        # sentence "gives the field no value a learner could read" is true of 0 and false
        # of anything else, and this script is meant to be rerun on other label files.
        out("  %s %s: reached by %d of the %d decisions%s"
            % (field, how, n, len(rows),
               ", so this cell gives the field no value a learner could read" if not n
               else ", so the field carries a value on those decisions and nothing on the "
               "rest"))


def branch_facts(rows, records):
    """The render branches `spec.md` calls dead or live, counted over the whole cell.

    The two caps are read off the STATE and not off the record, because the record is what
    the cap already cut: a state with nine origins would show eight and say so, and the
    sentence being checked is that no state of these cells comes near either cap.
    """
    exchanges = sum(len(rec["exchanges"]) for rec in records)
    clamped = sum(1 for rec in records for ex in rec["exchanges"] if ex["clamped"])
    empty_none = sum(1 for rec in records for ex in rec["exchanges"]
                     if ex["empty"] is None)
    no_adv = sum(1 for rec in records for o in rec["origins"] if not o["advertised"])
    omitted = sum(1 for rec in records for o in rec["origins"]
                  if o["advertisers_omitted"])
    out("  exchanges rendered %d; of them the \"(clamped)\" suffix fires on %d and the "
        "\"-\" of a timed-out exchange's empty field on %d"
        % (exchanges, clamped, empty_none))
    out("  origin rows with no advertiser, which render \"no peer told it\": %d" % no_adv)
    out("  origin rows with an advertiser left out, which render \"(+N more)\": %d; the "
        "most advertisers any one origin of a STATE names is %d against the cap of %d"
        % (omitted,
           max(len(o["advertised"]) for r in rows
               for o in r["state"]["origins"].values()),
           OBS.MAX_PEERS))
    out("  states that omit an origin, which render \" (more left out)\": %d; the most "
        "origins any STATE holds is %d against the cap of %d"
        % (sum(1 for rec in records if rec["origins_omitted"]),
           max(len(r["state"]["origins"]) for r in rows), OBS.MAX_ORIGINS))
    out("  resolved None, which renders a send-time decision: %d of %d records"
        % (sum(1 for rec in records if rec["resolved"] is None), len(records)))
    named = sum(1 for rec in records if rec["polled_peer"] is not None)
    out("  records that name the polled peer: %d of %d (observe.txt prints the null side "
        "of this and its three causes)" % (named, len(records)))


# ---------------------------------------------------------------------------

def main():
    with open(os.path.join(HERE, "weight.json"), encoding="utf-8") as fh:
        wj = json.load(fh)
    w_new = wj["weight_ms_per_poll"]
    w_old = wj["superseded"]["weight_ms_per_poll"]

    want = digests()
    bad = []
    for path, digest in want.items():
        got = sha256_of(path)
        if got != digest:
            bad.append((path, digest, got))
    if bad:
        # Not a line in the output: every number below is a statement about two particular
        # files, and a file that is not the one labels.sha256 names makes all of them
        # describe a labelling nobody can point at.
        for path, digest, got in bad:
            print(f"STOP {path} is {got}, labels.sha256 says {digest}")
        return 1

    out("The numbers spec.md reads off the label files")
    out("=============================================")
    out("Recomputed from the two label files at every run, so that no number in the prose "
        "of this directory is a measurement taken once and never checked again.  It runs "
        "no simulation and decides nothing: the weight, the labels and the J are fixed "
        "elsewhere.")
    out("Weight: %.17g ms per poll, read from weight.json. Superseded: %.17g."
        % (w_new, w_old))
    for path, digest in want.items():
        out(f"  {path}")
        out(f"    sha256 {digest} -- checked before anything below was counted")
    out()

    totals = {"records": 0, "exchanges": 0, "empty_none": 0, "no_adv": 0, "named": 0,
              "resolved_none": 0, "clamped": 0}
    for p, name in CELLS:
        rows = rows_of(os.path.join(LEARN, name))
        records = [OBS.observation(r["state"], r["resolved"], r["history"], r["ts_ms"],
                                   r["neighbor"]) for r in rows]
        out(f"p {p}, from {name}")
        margin_facts(rows, w_new)
        moved_facts(rows, w_new, w_old)
        drop_facts(rows)
        post_stop_facts(rows)
        retention_facts(rows, STABILITY_SEEDS)
        out("  the state situations this cell reaches")
        situation_facts(rows, records)
        out("  the render branches, counted over every decision of the cell")
        branch_facts(rows, records)
        out()
        totals["records"] += len(records)
        totals["exchanges"] += sum(len(rec["exchanges"]) for rec in records)
        totals["clamped"] += sum(1 for rec in records for ex in rec["exchanges"]
                                 if ex["clamped"])
        totals["empty_none"] += sum(1 for rec in records for ex in rec["exchanges"]
                                    if ex["empty"] is None)
        totals["no_adv"] += sum(1 for rec in records for o in rec["origins"]
                                if not o["advertised"])
        totals["named"] += sum(1 for rec in records if rec["polled_peer"] is not None)
        totals["resolved_none"] += sum(1 for rec in records if rec["resolved"] is None)

    out("Both cells together, which is the population spec.md's branch sentences use")
    out("  records %d, exchanges rendered %d" % (totals["records"], totals["exchanges"]))
    out("  \"(clamped)\" %d, the timed-out \"-\" %d, \"no peer told it\" %d"
        % (totals["clamped"], totals["empty_none"], totals["no_adv"]))
    out("  resolved None %d, records that name the polled peer %d"
        % (totals["resolved_none"], totals["named"]))
    out()
    out("What this file cannot say: that any of these counts is right. Each is this "
        "construction counting its own output, and the label files are the only source. "
        "A branch reached 0 times here is reached 0 times by THIS workload, which is not "
        "the same as being unreachable, and a situation reached a dozen times still has "
        "no value compared against a source object.")

    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT_TXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
