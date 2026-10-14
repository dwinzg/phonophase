"""
Regressions for defects found by running the command line tools on input they
were not written against: a killed run, a renamed table, an empty file, a
corpus with one file name in two directories.
"""
import csv
import gzip
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyse_slices  # noqa: E402
import attack_analysis  # noqa: E402
import slice_and_probe  # noqa: E402
import sonority_voicing  # noqa: E402

from phonophase.corpus import select_balanced  # noqa: E402

EVAL = os.path.join(ROOT, "results", "eval_slices.csv.gz")


def _script(name, *args):
    return subprocess.run([sys.executable, os.path.join(SCRIPTS, name), *args],
                          capture_output=True, text=True)


def _table(path, categories, n_utt=60):
    rng = np.random.default_rng(0)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["utterance_id", "label", "attack", "language", "category",
                    "seg", analyse_slices.PHASE, analyse_slices.MAGNITUDE])
        for u in range(n_utt):
            for c in categories:
                for _ in range(2):
                    w.writerow([f"u{u}", u % 2, "A07" if u % 2 else "-", "en", c,
                                "run", rng.normal(u % 2), rng.normal()])


def test_manifest_ids_are_distinct_when_file_names_repeat():
    with tempfile.TemporaryDirectory() as tmp:
        for lang in ("en", "de"):
            for label in ("real", "fake"):
                os.makedirs(os.path.join(tmp, "c", lang, label))
                for name in ("0001.wav", "0002.wav"):
                    open(os.path.join(tmp, "c", lang, label, name), "w").close()
        out = os.path.join(tmp, "m.csv")
        r = _script("make_manifest.py", "--root", os.path.join(tmp, "c"),
                    "--layout", "lang_label", "--corpus", "c", "--out", out)
        assert r.returncode == 0, r.stderr
        with open(out, newline="") as fh:
            ids = [row["utterance_id"] for row in csv.DictReader(fh)]
        assert len(ids) == 8 and len(set(ids)) == 8, ids


def test_resume_drops_the_utterance_a_kill_left_half_written():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.csv")
        with open(path, "w", newline="") as fh:
            fh.write("utterance_id,x\nuA,1\nuA,2\nuB,1\nuB,")
        assert slice_and_probe.already_done(path) == {"uA"}
        with open(path, newline="") as fh:
            assert fh.read() == "utterance_id,x\nuA,1\nuA,2\n"
        # Idempotent on what is now a clean file, at the cost of one utterance.
        assert slice_and_probe.already_done(path) == set()


def test_resume_reads_a_gzip_table_without_touching_it():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.csv.gz")
        with gzip.open(path, "wt", newline="") as fh:
            fh.write("utterance_id,x\nuA,1\nuB,1\n")
        before = open(path, "rb").read()
        assert slice_and_probe.already_done(path) == {"uA", "uB"}
        assert open(path, "rb").read() == before


def test_the_statistics_commands_import_without_torch():
    code = (
        "import builtins, sys\n"
        "real = builtins.__import__\n"
        "def blocked(name, *a, **k):\n"
        "    if name.split('.')[0] == 'torch':\n"
        "        raise ImportError('torch is blocked for this test')\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = blocked\n"
        f"sys.path.insert(0, {SCRIPTS!r})\n"
        "import band_edge\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_by_language_does_not_say_holds_without_obstruents():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.csv")
        _table(path, analyse_slices.SONORANTS)
        r = _script("analyse_slices.py", "--table", path, "--by-language",
                    "--n-boot", "400")
        assert r.returncode == 0, r.stderr
        line = [x for x in r.stdout.splitlines() if x.startswith("en ")][0]
        assert "too few" in line and "holds" not in line, r.stdout


def test_tables_are_recognised_by_content_not_by_name():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "renamed.csv")
        with open(EVAL, "rb") as src, open(path, "wb") as dst:
            dst.write(src.read())
        assert attack_analysis.load(path, "run")["y"].size == 68996
        assert sonority_voicing.load(path, "run")


def test_bad_input_gets_a_message_not_a_traceback():
    with tempfile.TemporaryDirectory() as tmp:
        empty = os.path.join(tmp, "empty.csv")
        open(empty, "w").close()
        r = _script("analyse_slices.py", "--table", empty)
        assert r.returncode != 0 and "Traceback" not in r.stderr, r.stderr

        r = _script("analyse_slices.py", "--table", EVAL, "--n-boot", "0")
        assert r.returncode != 0 and "Traceback" not in r.stderr, r.stderr

        bad = os.path.join(tmp, "nan.csv")
        _table(bad, analyse_slices.SONORANTS + analyse_slices.OBSTRUENTS, n_utt=4)
        with open(bad, "a", newline="") as fh:
            csv.writer(fh).writerow(["u0", 0, "-", "en", "NASALS", "run", "nan", 0.1])
        r = _script("analyse_slices.py", "--table", bad)
        assert "Traceback" not in r.stderr and "non finite" in r.stderr, r.stderr

        son = os.path.join(tmp, "son.csv")
        _table(son, analyse_slices.SONORANTS)
        r = _script("band_edge.py", "r2", "--table", son, "--n-boot", "400",
                    "--seeds", "0")
        assert r.returncode != 0 and "Traceback" not in r.stderr, r.stderr


def test_every_table_reader_reports_a_missing_file_or_column_without_a_traceback():
    with tempfile.TemporaryDirectory() as tmp:
        missing = os.path.join(tmp, "nope.csv")
        headless = os.path.join(tmp, "headless.csv")
        with open(headless, "w") as fh:
            fh.write("utterance_id,label\nu0,0\n")
        for script, flag in (("attack_analysis.py", "--table"),
                             ("feature_analysis.py", "--table"),
                             ("sonority_voicing.py", "--table"),
                             ("falsification_4.py", "--eval-table")):
            extra = (["--train-table", EVAL] if script == "falsification_4.py" else [])
            for path, says in ((missing, "no such file"), (headless, "has no column")):
                r = _script(script, flag, path, *extra)
                assert r.returncode != 0, (script, path)
                assert "Traceback" not in r.stderr and says in r.stderr, (script, r.stderr)


def test_a_manifest_shares_the_label_vocabulary_and_reports_what_it_drops():
    import io
    from contextlib import redirect_stdout

    import slice_and_probe

    with tempfile.TemporaryDirectory() as tmp:
        for name in ("a.wav", "b.wav", "c.wav"):
            open(os.path.join(tmp, name), "w").close()
        manifest = os.path.join(tmp, "m.csv")
        with open(manifest, "w") as fh:
            fh.write("path,label\na.wav,human\nb.wav,tts\nc.wav,maybe\n")
        out = io.StringIO()
        with redirect_stdout(out):
            rows = slice_and_probe.read_manifest(manifest)
    assert [r[5] for r in rows] == ["bonafide", "spoof"]
    assert "skipped 1 rows" in out.getvalue()


def test_a_spoof_limit_of_zero_selects_no_spoof():
    rows = ([("s", f"A{k:02d}", "spoof") for k in range(13) for _ in range(5)]
            + [("s", "-", "bonafide")] * 10)
    chosen = select_balanced(rows, 10, 0, 1234)
    assert sum(r[2] == "spoof" for r in chosen) == 0
    assert sum(r[2] == "bonafide" for r in chosen) == 10


if __name__ == "__main__":
    from _runner import main
    main(globals(), "test_cli_robustness")
