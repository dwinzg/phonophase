# Contributing

Issues and pull requests are welcome. Trying to break a claim is the most
useful contribution of all, and there are issue templates for a number that
does not reproduce and for a challenged claim.

## Set up

```bash
git clone https://github.com/dwinzg/phonophase.git && cd phonophase
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" -c constraints.txt
pytest -q
ruff check .
```

`constraints.txt` pins the versions that reproduce every published number.
Without it the package still installs on Python 3.10 or later with current
versions, which CI also tests.

## Send a change

Branch from `main`, add a test built on data whose answer is known by
construction, and open a pull request. CI runs the tests and the linter on
every one.

## The standard a claim has to meet

- Every claim ships with the command that produced it.
- Pre-registration is binding. A new idea found in data already seen is a new
  registration, tested on data that had no part in forming it. Never edit a
  registration above its outcome; add a dated note.
- Undecided is not rejected, and it is not confirmed.
- Every bootstrap is clustered. Report the size of an effect, not only whether
  its interval clears zero.
- No best of many comparisons: compare the registered pair of statistics.

The full standard, including what a re-derivation has to report, is in
[`docs/WORKING-PROTOCOL.md`](docs/WORKING-PROTOCOL.md).

Security reports go through [`SECURITY.md`](SECURITY.md), not a public issue.
