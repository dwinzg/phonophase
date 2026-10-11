# Contributing

Issues and pull requests are welcome, and trying to break a claim is a welcome
contribution.

```bash
git clone https://github.com/dwinzg/phonophase.git && cd phonophase
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Branch from `main`, add a test built on data whose answer is known by
construction, and open a pull request. CI runs the suite on every one.

What a change to a number or a claim needs is in the
[README](README.md#contributing) and, in full, in
[`docs/WORKING-PROTOCOL.md`](docs/WORKING-PROTOCOL.md). The short version:
ship the command with the claim, cluster every bootstrap, and never edit a
pre-registration above its outcome.

Security reports go through [`SECURITY.md`](SECURITY.md), not a public issue.
