"""
Minimal test runner.

pytest may not be installed, so every test file here also runs standalone with
plain python. The files are ordinary pytest test functions, so once pytest is
present the same files work unchanged under it. The files are ordinary pytest test functions, so once pytest
is installed from requirements-dev.txt the same files work unchanged under it.

    python tests/test_metrics.py       run one file with no dependencies
    pytest tests/                      run everything, once pytest is present
"""

import sys
import traceback


def run(namespace, name):
    tests = sorted(
        (k, v) for k, v in namespace.items()
        if k.startswith("test_") and callable(v)
    )
    failures = []
    for test_name, fn in tests:
        try:
            fn()
            print(f"  pass  {test_name}")
        except BaseException as exc:
            # pytest.skip raises outside the Exception tree.
            if type(exc).__name__ == "Skipped":
                print(f"  skip  {test_name}: {exc}")
                continue
            if not isinstance(exc, Exception):
                raise
            failures.append(test_name)
            print(f"  FAIL  {test_name}")
            traceback.print_exc()

    print(f"\n{name}: {len(tests) - len(failures)} passed, {len(failures)} failed")
    return 1 if failures else 0


def main(namespace, name):
    sys.exit(run(namespace, name))
