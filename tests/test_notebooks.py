"""
Structural tests for the generated notebooks.

These exist because a shipped notebook failed on its first cell in Colab with a
SyntaxError. nbformat stores source as a list of strings and Jupyter joins them
with "".join(), so every element except the last must END WITH a newline. The
generator stripped them and the lines welded together.

The original check missed it because it repaired the newlines before parsing,
so it validated a different artifact from the one on disk. Nothing here repairs
anything, and cells are transformed with IPython's own transformer rather than
a hand rolled approximation, because approximations are what let the defect
through the first time.
"""

import ast
import glob
import json
import os
import re

NOTEBOOK_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
NOTEBOOKS = sorted(glob.glob(os.path.join(NOTEBOOK_DIR, "*.ipynb")))

try:
    from IPython.core.inputtransformer2 import TransformerManager
    _TRANSFORM = TransformerManager().transform_cell
    HAVE_IPYTHON = True
except ImportError:  # pragma: no cover
    HAVE_IPYTHON = False

    def _TRANSFORM(source):
        """
        Fallback when IPython is absent. Replaces shell escapes and magics with
        pass at the same indentation, consuming backslash continuations, so a
        magic that is the only statement in a block does not leave an empty
        body. Weaker than the real transformer, so install the dev extras.
        """
        out, lines, i = [], source.split("\n"), 0
        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()
            if stripped.startswith(("!", "%")):
                indent = line[:len(line) - len(stripped)]
                while line.rstrip().endswith("\\") and i + 1 < len(lines):
                    i += 1
                    line = lines[i]
                out.append(f"{indent}pass")
            else:
                out.append(line)
            i += 1
        return "\n".join(out)


def _notebooks():
    assert NOTEBOOKS, f"no notebooks found under {NOTEBOOK_DIR}"
    for path in NOTEBOOKS:
        with open(path) as fh:
            yield os.path.basename(path), json.load(fh)


def test_notebooks_are_valid_nbformat():
    for name, nb in _notebooks():
        assert nb.get("nbformat") == 4, f"{name}: nbformat {nb.get('nbformat')}"
        assert nb.get("cells"), f"{name}: no cells"
        for i, cell in enumerate(nb["cells"]):
            assert cell["cell_type"] in ("code", "markdown"), f"{name} cell {i}"
            assert isinstance(cell["source"], list), f"{name} cell {i}: source not a list"


def test_every_source_line_ends_with_a_newline_except_the_last():
    # The exact defect that broke the first run in Colab.
    for name, nb in _notebooks():
        for i, cell in enumerate(nb["cells"]):
            for j, line in enumerate(cell["source"][:-1]):
                assert line.endswith("\n"), (
                    f"{name} cell {i} line {j} has no trailing newline. Jupyter "
                    f"joins with ''.join(), so this welds onto the next line: "
                    f"{line[:60]!r}"
                )


def test_code_cells_parse_after_ipython_transformation():
    # Joined exactly as Jupyter joins them, then transformed exactly as IPython
    # transforms them. No cell is skipped.
    for name, nb in _notebooks():
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            try:
                ast.parse(_TRANSFORM(source))
            except SyntaxError as exc:
                raise AssertionError(
                    f"{name} cell {i} does not parse as written: {exc}\n"
                    f"first 140 chars: {source[:140]!r}"
                ) from exc




def test_invoked_scripts_exist():
    # A notebook that shells out to a script that is not in the repository
    # fails only once someone is sitting in Colab waiting for it.
    pattern = re.compile(r"!\s*python\s+(?:-m\s+)?([\w./]+)")
    for name, nb in _notebooks():
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            for target in pattern.findall("".join(cell["source"])):
                if target.endswith(".py"):
                    path = os.path.join(REPO_ROOT, target)
                    assert os.path.exists(path), \
                        f"{name} cell {i} runs {target}, which does not exist"
                else:
                    module = os.path.join(REPO_ROOT, "src", *target.split(".")) + ".py"
                    assert os.path.exists(module), \
                        f"{name} cell {i} runs -m {target}, which does not exist"


def test_referenced_repo_files_exist():
    # Paths the notebooks point readers at, in prose or in code.
    pattern = re.compile(r"(?:configs|docs|src|scripts|tests)/[\w./-]+\.(?:py|yaml|md|sh)")
    for name, nb in _notebooks():
        for i, cell in enumerate(nb["cells"]):
            for target in set(pattern.findall("".join(cell["source"]))):
                assert os.path.exists(os.path.join(REPO_ROOT, target)), \
                    f"{name} cell {i} references {target}, which does not exist"


def test_variables_are_defined_before_shell_interpolation():
    # IPython interpolates {NAME} in a shell escape from the namespace. If the
    # cell that defines NAME comes later, the run dies partway through, which
    # is how the earlier cell ordering bug would have shown up.
    brace = re.compile(r"!\s*[^\n]*?\{(\w+)\}")
    for name, nb in _notebooks():
        defined = set()
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            # Walk line by line so a variable defined earlier in the SAME cell
            # counts, which is the common and correct pattern.
            for line in "".join(cell["source"]).split("\n"):
                for used in brace.findall(line):
                    assert used in defined, (
                        f"{name} cell {i} interpolates {{{used}}} into a shell "
                        f"command before anything defines it"
                    )
                for target in re.findall(r"^\s*(\w+)\s*=", line):
                    defined.add(target)
                for target in re.findall(r"^\s*(?:for|with)\s+(\w+)", line):
                    defined.add(target)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from _runner import main
    main(globals(), "test_notebooks")
