"""An archive with no published checksum is downloaded but never unpacked by default."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import fetch_data  # noqa: E402


def _run(argv, entry):
    calls = {"extract": 0}
    saved = (fetch_data.load_registry, fetch_data.download, fetch_data.extract, sys.argv)
    fetch_data.load_registry = lambda: {"x": entry}
    fetch_data.download = lambda url, dest: open(dest, "w").close()
    fetch_data.extract = lambda dest, out: calls.__setitem__("extract", calls["extract"] + 1)
    sys.argv = ["fetch_data.py", *argv]
    try:
        fetch_data.main()
        return None, calls
    except SystemExit as exc:
        return str(exc), calls
    finally:
        fetch_data.load_registry, fetch_data.download, fetch_data.extract, sys.argv = saved


def test_an_archive_without_a_checksum_is_not_unpacked_unless_asked():
    entry = {"name": "X", "licence_verified": True, "direct_url": "https://example.org/x.zip",
             "filename": "x.zip"}
    with tempfile.TemporaryDirectory() as tmp:
        message, calls = _run(["--dataset", "x", "--out", tmp], entry)
        assert message and "no published checksum" in message and calls["extract"] == 0
        message, calls = _run(["--dataset", "x", "--out", tmp, "--allow-unverified"], entry)
        assert message is None and calls["extract"] == 1


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from _runner import main
    main(globals(), "test_fetch_data")
