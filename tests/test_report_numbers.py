"""
Tests for results/report_numbers.csv, the figures' data.

The figures in report/ are drawn from this file, so a row that points at a
document which is not there, or an estimate outside its own interval, would be
printed in a figure without anything complaining.
"""

import csv
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _rows():
    with open(os.path.join(ROOT, "results", "report_numbers.csv"), newline="") as fh:
        return list(csv.DictReader(fh))


def test_every_row_names_a_document_that_exists():
    for r in _rows():
        assert os.path.exists(os.path.join(ROOT, r["source"])), r


def test_every_estimate_lies_inside_its_interval():
    for r in _rows():
        low, value, high = float(r["low"]), float(r["value"]), float(r["high"])
        assert low <= value <= high, r


def test_the_headline_row_is_the_registered_result():
    registered = [r for r in _rows() if r["figure"] == "clustering"
                  and r["row"].startswith("utterance") and r["series"] == "asymmetry"]
    assert [(r["value"], r["low"], r["high"]) for r in registered] == [
        ("0.063", "0.045", "0.080")]


if __name__ == "__main__":
    from _runner import main
    main(globals(), "test_report_numbers")
