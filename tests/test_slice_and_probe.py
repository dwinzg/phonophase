"""
Tests for the speaker column in scripts/slice_and_probe.py.

Why this exists
    The bootstrap in phonophase.stats can already resample by group, but the
    slice tables carried no speaker id, so it could only ever resample
    utterances. ASVspoof 2019 LA eval packs 71237 utterances into 67 speakers,
    so an utterance level bootstrap understates uncertainty by an unknown
    amount and nobody could even check how much. This is the fix: a speaker
    column, sourced from the ASVspoof protocol file the run already reads for
    attack ids, keyed on utterance id.

    Three things are load bearing and each has a direct test.

    The column actually lands in the CSV under the csv.DictWriter contract
    scripts/slice_and_probe.py uses, not just in an in-memory dict that never
    makes it to disk.

    A real utterance id from the shipped ASVspoof protocol resolves to its
    real, known speaker id, not merely to *some* string.

    An utterance id the protocol does not recognise is handled explicitly. It
    must not become "", which downstream would be silently indistinguishable
    from a genuinely blank cell and would quietly corrupt any clustering built
    on this column later.

    Two more tests turn the coverage check this task asked for into a
    standing regression: every utterance_id in the shipped
    results/eval_slices.csv.gz must resolve to a real speaker, and the eval
    protocol's distinct speaker count is pinned at 67, not the 48 that only
    counts speakers targeted by a spoofing system.
"""

import csv
import gzip
import os
import sys
import tempfile
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))

from slice_and_probe import (  # noqa: E402
    DEFAULT_EVAL_PROTOCOL,
    FIELDS,
    UNKNOWN_SPEAKER,
    check_header_compatible,
    read_manifest,
    read_speaker_map,
    speaker_for,
)

from phonophase.probe import STATISTIC_NAMES  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
SHIPPED_EVAL_TABLE = os.path.join(REPO_ROOT, "results", "eval_slices.csv.gz")

# A few lines lifted verbatim from configs/protocols/ASVspoof2019.LA.cm.eval.trl.txt,
# in the real SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY layout, so the parsing
# test is pinned to the real format rather than a guess at it.
SAMPLE_PROTOCOL_LINES = """\
LA_0039 LA_E_2834763 - A11 spoof
LA_0014 LA_E_8877452 - A14 spoof
LA_0030 LA_E_5849185 - - bonafide
"""


def _write(dir_, name, text):
    path = os.path.join(dir_, name)
    with open(path, "w", newline="") as fh:
        fh.write(text)
    return path


# --------------------------------------------------------------------------
# read_speaker_map / speaker_for
# --------------------------------------------------------------------------

def test_read_speaker_map_parses_the_real_protocol_line_layout():
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "protocol.txt", SAMPLE_PROTOCOL_LINES)
        m = read_speaker_map(path)

    assert m == {
        "LA_E_2834763": "LA_0039",
        "LA_E_8877452": "LA_0014",
        "LA_E_5849185": "LA_0030",
    }


def test_known_utterance_id_maps_to_its_known_speaker():
    # Not a synthetic fixture: this is the actual shipped ASVspoof 2019 LA
    # eval protocol, and LA_E_2834763 is its first data row, whose speaker is
    # LA_0039. If this ever fails, either the shipped protocol file changed
    # underneath the code or the parsing broke, both worth knowing loudly.
    assert os.path.exists(DEFAULT_EVAL_PROTOCOL), \
        f"shipped protocol missing: {DEFAULT_EVAL_PROTOCOL}"

    speaker_map = read_speaker_map(DEFAULT_EVAL_PROTOCOL)

    assert speaker_for("LA_E_2834763", speaker_map) == "LA_0039"
    assert speaker_for("LA_E_5849185", speaker_map) == "LA_0030"


def test_missing_utterance_id_is_handled_explicitly_not_silently_empty():
    speaker_map = {"LA_E_2834763": "LA_0039"}

    result = speaker_for("LA_E_0000000_NOT_IN_ANY_PROTOCOL", speaker_map)

    assert result == UNKNOWN_SPEAKER
    assert result != "", "a miss must not silently become an empty cell"
    assert result is not None
    # UNKNOWN can never collide with a genuine ASVspoof speaker id.
    assert not result.startswith("LA_")


# --------------------------------------------------------------------------
# the column is written
# --------------------------------------------------------------------------

def test_fields_includes_speaker_keyed_on_utterance_id():
    assert "utterance_id" in FIELDS
    assert "speaker" in FIELDS
    # speaker belongs beside the other row-identity columns, ahead of the
    # per-slice measurements, so a reader scanning the header sees it with
    # the rest of what identifies the row.
    assert FIELDS.index("speaker") == FIELDS.index("utterance_id") + 1


def test_speaker_column_is_written_and_round_trips_through_the_real_writer():
    # Builds exactly the row shape run() passes to writer.writerow(), using
    # the same FIELDS constant and the same csv.DictWriter/DictReader
    # machinery slice_and_probe.py uses, so this exercises the real
    # contract rather than a stand-in for it.
    row = {
        "utterance_id": "LA_E_2834763", "speaker": "LA_0039", "label": 1,
        "attack": "A11", "language": "en", "corpus": "asvspoof2019_la",
        "category": "FRICATIVES", "phoneme": "s", "seg": "run",
        "start_frame": 12, "n_frames": 3, "duration_s": 0.06,
        **{name: 0.0 for name in STATISTIC_NAMES},
    }
    assert set(row) == set(FIELDS), "fixture row must match the real schema"

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.csv")
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(row)

        with open(path, newline="") as fh:
            written = list(csv.DictReader(fh))

    assert len(written) == 1
    assert written[0]["utterance_id"] == "LA_E_2834763"
    assert written[0]["speaker"] == "LA_0039"


def test_manifest_rows_get_the_not_applicable_speaker_sentinel():
    # A manifest describes an arbitrary corpus with no ASVspoof protocol, so
    # its rows must still carry a speaker cell (every row does, uniformly),
    # but it has to be the existing not-applicable marker, not a fabricated
    # value.
    with tempfile.TemporaryDirectory() as d:
        wav_path = _write(d, "a.wav", "not really audio, path just has to exist")
        manifest = _write(d, "manifest.csv",
                          f"path,label,utterance_id,language\n{wav_path},bonafide,u1,de\n")
        rows = read_manifest(manifest)

    assert len(rows) == 1
    assert rows[0][1] == "u1"
    assert rows[0][-1] == "-"


# --------------------------------------------------------------------------
# resuming into a file from before this column existed
# --------------------------------------------------------------------------

def test_check_header_compatible_allows_a_file_that_does_not_exist_yet():
    with tempfile.TemporaryDirectory() as d:
        check_header_compatible(os.path.join(d, "not_there.csv"), FIELDS)  # no raise


def test_check_header_compatible_allows_a_header_that_already_matches():
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "out.csv", ",".join(FIELDS) + "\n")
        check_header_compatible(path, FIELDS)  # no raise


def test_check_header_compatible_refuses_a_header_from_before_speaker_existed():
    old_fields = [f for f in FIELDS if f != "speaker"]
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "out.csv", ",".join(old_fields) + "\n")
        try:
            check_header_compatible(path, FIELDS)
        except SystemExit as exc:
            assert "speaker" in str(exc) or "header" in str(exc).lower()
            return
    raise AssertionError("expected SystemExit for a stale, pre-speaker header")


# --------------------------------------------------------------------------
# the coverage check this task asked for, kept as a standing regression
# --------------------------------------------------------------------------

def test_shipped_eval_table_utterance_ids_are_fully_covered_by_the_protocol():
    assert os.path.exists(SHIPPED_EVAL_TABLE), \
        f"shipped table missing: {SHIPPED_EVAL_TABLE}"

    speaker_map = read_speaker_map(DEFAULT_EVAL_PROTOCOL)
    with gzip.open(SHIPPED_EVAL_TABLE, "rt", newline="") as fh:
        utterance_ids = {row["utterance_id"] for row in csv.DictReader(fh)}

    assert len(utterance_ids) == 2995, (
        f"expected the shipped table's known 2995 distinct utterances, got "
        f"{len(utterance_ids)}. If this genuinely changed, the coverage "
        "numbers in the task report need rechecking, not this test loosened."
    )

    resolved = {u: speaker_for(u, speaker_map) for u in utterance_ids}
    unresolved = [u for u, spk in resolved.items() if spk == UNKNOWN_SPEAKER]

    assert unresolved == [], (
        f"{len(unresolved)} of {len(utterance_ids)} utterance ids in "
        f"{SHIPPED_EVAL_TABLE} do not resolve to a speaker in "
        f"{DEFAULT_EVAL_PROTOCOL}; sample: {unresolved[:5]}"
    )


def test_eval_protocol_has_67_distinct_speakers_not_48():
    # 48 is real, but it counts only the speakers targeted by a spoofing
    # system. The protocol carries 19 further bonafide-only speakers, for 67
    # distinct speaker ids overall, and it is the 67 the bootstrap needs to
    # know about since bonafide rows are exactly where it matters most.
    speaker_map = read_speaker_map(DEFAULT_EVAL_PROTOCOL)
    distinct_speakers = set(speaker_map.values())

    assert len(distinct_speakers) == 67
    assert sum(1 for s in distinct_speakers if s.startswith("LA_")) == 67


def test_every_speaker_contributes_at_least_one_utterance():
    speaker_map = read_speaker_map(DEFAULT_EVAL_PROTOCOL)
    counts = Counter(speaker_map.values())

    assert min(counts.values()) >= 1
    assert sum(counts.values()) == len(speaker_map) == 71237


if __name__ == "__main__":
    from _runner import main
    main(globals(), "test_slice_and_probe")
