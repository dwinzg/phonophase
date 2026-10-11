"""
Tests for the shared cluster draw, src/phonophase/stats.py.

The defect these tests exist to prevent
    Two bootstrap draw designs lived in this repository and disagreed with each
    other. The pre-registered path, `advantage_ci` in
    `scripts/analyse_slices.py`, called `_cluster_draw` once per category inside
    a loop over categories, so the three per category advantages whose mean is
    the registered LEVEL were resampled independently. One source utterance
    feeds all three categories, so drawing them independently discards the
    dependence they share and shrinks the variance of their mean.
    `scripts/sonority_voicing.py` used one shared draw by explicit design and
    said so in its docstring. The registered path was the naive one, and it
    understated every published multi category interval.

Why this file asserts two things that pull in opposite directions
    A bug that merely widened everything, or narrowed everything, would satisfy
    one of these and fail the other. Under a shared draw, with a per utterance
    shock common to every category:

        the LEVEL, a mean over categories, must get WIDER, because independent
        draws average the shared shock away across three categories and a shared
        draw does not;

        the ASYMMETRY, a difference between two group means, must get NARROWER,
        because the shared shock cancels in a difference and only a paired draw
        can exploit that.

    Both references below are the naive estimators, written out here rather than
    kept in the package, so the suite states what was wrong without offering a
    second production path to drift back into.

What is NOT claimed here
    Not that the correction changes any verdict. It is a correction to widths.
    Nor that pairing narrows the ASYMMETRY on every table: on the eval table the
    two designs give nearly the same interval.
    Whether an interval still excludes zero is a question for the real table and
    is recorded with the run that measured it, not asserted in a unit test.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))

import analyse_slices  # noqa: E402
import phonophase.stats as stats  # noqa: E402
import sonority_voicing  # noqa: E402
from analyse_slices import (  # noqa: E402
    MAGNITUDE,
    PHASE,
    advantage_ci,
    asymmetry_ci,
    folded,
    usable,
)
from phonophase.stats import _cluster_draw, _cluster_index, cluster_groups  # noqa: E402

BOOT = 400
SONORANTS = ["NASALS", "APPROXIMANTS", "VOWELS_STEADY"]
OBSTRUENTS = ["FRICATIVES", "PLOSIVES", "AFFRICATES"]


# --------------------------------------------------------------------------
# a table whose categories genuinely share their utterances
# --------------------------------------------------------------------------

def _shared_utterance_table(seed=3, n_utt=60, per_utt=3, utterance_sd=1.2,
                            sonorant_shift=1.6, obstruent_shift=0.8):
    """
    Six categories, every one of them built from the SAME utterances.

    That is the structure the fix exists to respect, and it is the real
    structure of the eval table: a source utterance yields slices in every
    phoneme category it happens to contain, so the per category advantages are
    not independent draws from anything.

    The per utterance shock is applied to the phase statistic and shared across
    all six categories, which is what makes the six AUCs move together. The
    magnitude statistic gets a shock of its own, so it carries utterance
    structure too, as it must when it is measured on the same slices, without
    being collinear with the phase statistic.

    **Both groups separate clearly away from chance here, and that is load
    bearing.** The registered statistic folds, `max(auc, 1-auc)`. Folding
    reflects draws about 0.5, so a group sitting near chance has its draws mapped
    from both sides onto one side, which destroys the very correlation between
    the two groups that a paired draw exists to exploit. A fixture with a near
    chance obstruent group therefore measures the folding pathology instead of
    the pairing, and that case has a test of its own below.
    """
    rng = np.random.default_rng(seed)
    utterances = [(f"B{i:03d}", 0) for i in range(n_utt)]
    utterances += [(f"S{i:03d}", 1) for i in range(n_utt)]
    shock = {utt: rng.normal(0, utterance_sd) for utt, _ in utterances}
    magnitude_shock = {utt: rng.normal(0, utterance_sd) for utt, _ in utterances}

    data = {}
    for category in SONORANTS + OBSTRUENTS:
        shift = sonorant_shift if category in SONORANTS else obstruent_shift
        y, u, attack, phase, magnitude = [], [], [], [], []
        for utt, label in utterances:
            for _ in range(per_utt):
                y.append(label)
                u.append(utt)
                attack.append("-" if label == 0 else f"A{7 + (int(utt[1:]) % 4):02d}")
                bonafide_high = shift if label == 0 else 0.0
                phase.append(bonafide_high + shock[utt] + rng.normal(0, 1))
                magnitude.append(magnitude_shock[utt] + rng.normal(0, 1))
        data[category] = {
            "y": np.asarray(y),
            "u": np.asarray(u),
            "attack": np.asarray(attack),
            PHASE: np.asarray(phase),
            MAGNITUDE: np.asarray(magnitude),
            "bonafide_label": 0,
        }
    return data


# --------------------------------------------------------------------------
# the naive estimators, kept here so production has only one
# --------------------------------------------------------------------------

def _independent_per_category_level(data, categories, n_boot, seed, alpha=0.05,
                                    cluster="utterance"):
    """
    The estimator that produced every published LEVEL interval: one independent
    cluster draw per category, then the mean of the per category advantages.
    """
    rng = np.random.default_rng(seed)
    present = [c for c in categories if c in data and usable(data[c])]
    prepared = {c: _cluster_index(data[c]["y"], cluster_groups(data[c], cluster))
                for c in present}
    draws = []
    for _ in range(n_boot):
        per_category = []
        for cat in present:
            d = data[cat]
            idx = _cluster_draw(rng, prepared[cat])
            if idx.size == 0 or np.unique(d["y"][idx]).size < 2:
                continue
            y = d["y"][idx]
            per_category.append(folded(y, d[PHASE][idx]) - folded(y, d[MAGNITUDE][idx]))
        if per_category:
            draws.append(float(np.mean(per_category)))
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _unpaired_asymmetry_width(data, n_boot, seed, alpha=0.05, cluster="utterance"):
    """
    The asymmetry with its two group means taken from unrelated draws.

    A real bootstrap rather than two intervals combined arithmetically, because
    the question is what the resampling does, and this is what the per category
    design amounts to: every category drawn independently, so nothing the
    sonorant and obstruent means have in common can cancel in their difference.
    """
    rngs = {"son": np.random.default_rng(seed), "obs": np.random.default_rng(seed + 977)}
    groups = {"son": [c for c in SONORANTS if usable(data[c])],
              "obs": [c for c in OBSTRUENTS if usable(data[c])]}
    prepared = {c: _cluster_index(data[c]["y"], cluster_groups(data[c], cluster))
                for c in groups["son"] + groups["obs"]}

    draws = []
    for _ in range(n_boot):
        means = {}
        for side, categories in groups.items():
            per_category = []
            for cat in categories:
                d = data[cat]
                idx = _cluster_draw(rngs[side], prepared[cat])
                if idx.size == 0 or np.unique(d["y"][idx]).size < 2:
                    continue
                y = d["y"][idx]
                per_category.append(folded(y, d[PHASE][idx]) - folded(y, d[MAGNITUDE][idx]))
            means[side] = float(np.mean(per_category)) if per_category else None
        if means["son"] is not None and means["obs"] is not None:
            draws.append(means["son"] - means["obs"])

    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(hi - lo)


# --------------------------------------------------------------------------
# one draw, shared
# --------------------------------------------------------------------------

def test_one_draw_serves_every_category_so_the_utterances_agree():
    """
    The structural statement of the fix. Within one draw, every category must be
    looking at the same resampled utterances, because a drawn utterance brings
    all of its slices with it wherever they fall.

    Here every utterance contributes to all six categories, so the multiset of
    utterances seen by one category must equal the multiset seen by any other.
    Under the per category design these multisets are unrelated.
    """
    data = _shared_utterance_table()
    categories = SONORANTS + OBSTRUENTS
    pooled_y = np.concatenate([data[c]["y"] for c in categories])
    pooled_u = np.concatenate([data[c]["u"] for c in categories])
    owner = np.concatenate([
        np.full(data[c]["y"].size, i) for i, c in enumerate(categories)
    ])

    draws = list(stats.shared_cluster_draws(pooled_y, pooled_u, 5, seed=0))
    assert len(draws) == 5

    for idx in draws:
        assert idx.size > 0
        per_category = [
            sorted(pooled_u[idx[owner[idx] == i]].tolist())
            for i in range(len(categories))
        ]
        first = per_category[0]
        for i, seen in enumerate(per_category[1:], start=1):
            assert seen == first, (
                f"{categories[i]} saw different utterances from {categories[0]} "
                f"inside one draw, so the draw is not shared"
            )


def test_a_shared_draw_is_reproducible_from_its_seed():
    """
    Every published interval has to be re-derivable from the command that
    produced it. Two calls with one seed must give identical draws, and a
    different seed must give different ones.
    """
    data = _shared_utterance_table()
    y = np.concatenate([data[c]["y"] for c in SONORANTS])
    u = np.concatenate([data[c]["u"] for c in SONORANTS])

    a = list(stats.shared_cluster_draws(y, u, 4, seed=0))
    b = list(stats.shared_cluster_draws(y, u, 4, seed=0))
    c = list(stats.shared_cluster_draws(y, u, 4, seed=1))

    assert all(np.array_equal(x, z) for x, z in zip(a, b))
    assert not all(np.array_equal(x, z) for x, z in zip(a, c))


def test_the_level_interval_widens_against_independent_per_category_draws():
    """
    The correction. Averaging three independently resampled categories cancels
    the shared utterance shock and reports a precision the data do not have.

    Measured on the real table the understatement was about 40 percent. Here the
    shock is deliberately strong, so the gap is larger and the assertion can be
    a clear inequality rather than a threshold.
    """
    data = _shared_utterance_table()

    shared = advantage_ci(data, SONORANTS, BOOT, 0)
    naive_lo, naive_hi = _independent_per_category_level(data, SONORANTS, BOOT, 0)

    shared_width = shared["ci_high"] - shared["ci_low"]
    naive_width = naive_hi - naive_lo

    assert shared_width > naive_width * 1.2, (
        f"the shared draw must report a materially wider LEVEL interval: "
        f"shared {shared_width:.4f} against naive {naive_width:.4f}"
    )


def test_the_asymmetry_interval_narrows_against_unpaired_group_draws():
    """
    The other direction, and the reason the asymmetry is the robust quantity.
    The shared utterance shock is common to sonorants and obstruents, so it
    cancels in their difference. A paired draw keeps that cancellation; two
    unrelated draws destroy it and report a difference as less certain than it
    is.
    """
    data = _shared_utterance_table()

    paired = asymmetry_ci(data, BOOT, 0)
    paired_width = paired["ci_high"] - paired["ci_low"]
    unpaired_width = _unpaired_asymmetry_width(data, BOOT, 0)

    # A margin rather than a bare inequality. Two estimators of the same
    # quantity differ by Monte Carlo noise alone, and an assertion that only has
    # to win by a hair would pass about half the time on no fix at all.
    assert paired_width < unpaired_width * 0.8, (
        f"pairing the two group means must materially tighten the ASYMMETRY "
        f"interval: paired {paired_width:.4f} against unpaired "
        f"{unpaired_width:.4f}"
    )


def test_a_near_chance_group_loses_most_of_the_pairing_benefit_to_folding():
    """
    The pairing benefit on the ASYMMETRY is real but conditional, and this is the
    condition.

    The registered statistic folds, `max(auc, 1-auc)`. Folding reflects draws
    about 0.5. A group whose phase statistic sits near chance therefore has its
    draws mapped from both sides onto one, which destroys the correlation with
    the other group's draws that the paired draw exists to exploit.

    This was found by accident, in a fixture built with a near chance obstruent
    group. It is a property of this fixture and is NOT why the eval table's
    ASYMMETRY interval barely moves between draw designs: there, folding
    reflects only a few percent of draws under utterance clustering, and the
    interval is unchanged because the larger variance of each group mean and the
    correlation between them nearly offset.
    """
    clear = _shared_utterance_table(obstruent_shift=0.8)
    near_chance = _shared_utterance_table(obstruent_shift=0.0)

    def ratio(data):
        paired = asymmetry_ci(data, BOOT, 0)
        paired_width = paired["ci_high"] - paired["ci_low"]
        return paired_width / _unpaired_asymmetry_width(data, BOOT, 0)

    assert ratio(clear) < ratio(near_chance), (
        "pairing must buy more when both groups are clear of chance than when "
        "one of them sits on it"
    )
    assert ratio(near_chance) > 0.85, (
        "with a near chance group the paired and unpaired widths should be close, "
        "which is the caveat this test records"
    )


def test_the_point_estimates_do_not_move_when_the_draw_design_changes():
    """
    A resampling design is a statement about uncertainty. If the observed value
    moves, the resampling is being applied to the data rather than to the
    interval, which is a bug and has happened in this project before.

    The arithmetic that keeps the LEVEL and the ASYMMETRY apart is asserted at
    the same time, since both are computed from one draw sequence now.
    """
    data = _shared_utterance_table()

    son = advantage_ci(data, SONORANTS, BOOT, 0)
    obs = advantage_ci(data, OBSTRUENTS, BOOT, 1)
    asym = asymmetry_ci(data, BOOT, 0)

    expected = float(np.mean([
        folded(data[c]["y"], data[c][PHASE]) - folded(data[c]["y"], data[c][MAGNITUDE])
        for c in SONORANTS
    ]))
    assert abs(son["advantage"] - expected) < 1e-12
    assert abs(asym["asymmetry"] - (son["advantage"] - obs["advantage"])) < 1e-12

    # And the observed value is inside its own interval, which a percentile
    # bootstrap does not guarantee but a correct one on data like this gives.
    assert son["ci_low"] < son["advantage"] < son["ci_high"]
    assert asym["ci_low"] < asym["asymmetry"] < asym["ci_high"]


def test_a_category_absent_from_a_draw_is_skipped_rather_than_counted_as_zero():
    """
    Under a shared draw a category can receive no rows at all, or rows from one
    class only, since the draw is made over the pooled table. Counting that as
    an advantage of zero would drag the mean toward zero in proportion to how
    often it happens.

    Asserted on `draw_mean` directly with a hand built draw, because on a
    realistic table a miss is too rare for an interval to show the difference.
    """
    rng = np.random.default_rng(0)
    n = 200
    y = np.tile([0, 1], n)
    owner = np.repeat([0, 1], n)
    phase = y * 2.0 + rng.normal(size=2 * n)
    magnitude = rng.normal(size=2 * n)
    everything = np.arange(2 * n)

    first_only = everything[owner == 0]
    alone = analyse_slices.draw_mean(first_only, owner, y, phase, magnitude, [0])
    assert alone > 0.2, "the fixture must have a real advantage to dilute"

    missed = analyse_slices.draw_mean(first_only, owner, y, phase, magnitude, [0, 1])
    assert missed == alone, "a category the draw missed must not count as zero"

    one_class = np.concatenate([first_only, everything[(owner == 1) & (y == 0)]])
    assert analyse_slices.draw_mean(one_class, owner, y, phase, magnitude,
                                    [0, 1]) == alone, (
        "a category the draw reached with one class only must be skipped too")

    assert analyse_slices.draw_mean(first_only, owner, y, phase, magnitude,
                                    [1]) is None


def test_the_interval_is_the_one_asked_for():
    """A 90 percent interval reported as a 95 percent one passes most tests."""
    data = _shared_utterance_table()
    wide = advantage_ci(data, SONORANTS, BOOT, 0)
    narrow = advantage_ci(data, SONORANTS, BOOT, 0, alpha=0.5)
    assert wide["ci_low"] < narrow["ci_low"] < narrow["ci_high"] < wide["ci_high"]
    assert (narrow["ci_high"] - narrow["ci_low"]) < 0.5 * (wide["ci_high"] - wide["ci_low"])

    draws = np.random.default_rng(0).normal(size=1000)
    r = stats.percentile_interval(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    assert r["ci_low"] == lo and r["ci_high"] == hi


# --------------------------------------------------------------------------
# one definition, not two
# --------------------------------------------------------------------------

def test_both_scripts_draw_through_the_same_object():
    """
    Two copies of the draw is the defect itself, not a stylistic matter. They
    must be the same object, so they cannot drift apart one commit at a time.
    """
    assert analyse_slices.shared_cluster_draws is stats.shared_cluster_draws
    assert sonority_voicing.shared_cluster_draws is stats.shared_cluster_draws


def test_the_bonafide_convention_has_one_definition_too():
    """
    `bonafide_label` and `BONAFIDE_MARKERS` were duplicated in both scripts with
    different failure behaviour, one returning None and one raising. Label 1 is
    SPOOF here, and getting it backwards inverts every direction while leaving
    every magnitude correct, so there is one reader and both scripts use it.
    """
    assert analyse_slices.BONAFIDE_MARKERS is stats.BONAFIDE_MARKERS
    assert sonority_voicing.BONAFIDE_MARKERS is stats.BONAFIDE_MARKERS
    assert analyse_slices.bonafide_label is stats.bonafide_label

    y = np.array([0, 0, 1, 1, 1])
    attack = np.array(["-", "-", "A07", "A08", "A09"])
    assert stats.bonafide_label(y, attack) == 0
    assert stats.require_bonafide_label(y, attack) == 0
    assert sonority_voicing.bonafide_label({"y": y, "attack": attack}) == 0

    inverted = np.array([1, 1, 0, 0, 0])
    assert stats.bonafide_label(inverted, attack) == 1
    assert sonority_voicing.bonafide_label({"y": inverted, "attack": attack}) == 1


def test_an_unreadable_convention_is_None_for_one_caller_and_raises_for_the_other():
    """
    The two callers need different failure behaviour and that is deliberate.
    The registered analysis defaults to utterance clustering, which never
    consults the answer, so a table without a usable attack column must still
    analyse. `sonority_voicing` cannot run at all without the direction, so it
    refuses loudly.
    """
    y = np.array([0, 0, 1, 1])
    blank = np.array(["", "", "", ""])

    assert stats.bonafide_label(y, blank) is None
    try:
        stats.require_bonafide_label(y, blank)
    except ValueError as exc:
        assert "bonafide" in str(exc).lower()
    else:
        raise AssertionError("require_bonafide_label must refuse an unreadable table")


if __name__ == "__main__":
    from _runner import main
    main(globals(), "test_shared_draw")
