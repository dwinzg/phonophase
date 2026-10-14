"""The shipped tables are the bytes results/SHA256SUMS records."""

import hashlib
import os

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def test_every_shipped_table_matches_its_checksum():
    with open(os.path.join(RESULTS, "SHA256SUMS")) as fh:
        expected = dict(reversed(line.split()) for line in fh if line.strip())
    shipped = {f for f in os.listdir(RESULTS) if f.endswith((".csv", ".csv.gz"))}
    assert shipped == set(expected), shipped ^ set(expected)
    for name, digest in expected.items():
        with open(os.path.join(RESULTS, name), "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == digest, name


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from _runner import main
    main(globals(), "test_results_integrity")
