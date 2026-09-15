"""Static drift guard for the notebooks: they must stay thin drivers over src/.

No kernel is needed. Every code cell must compile, define no functions or classes,
never touch torch.save / torch.load / load_state_dict directly, and every
``from src.<module> import name`` must resolve against the current src/ package.
"""
import glob
import importlib
import json
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = sorted(glob.glob(os.path.join(REPO, "*.ipynb")))
FORBIDDEN = ("torch.save(", "torch.load(", "load_state_dict(")
DEF_RE = re.compile(r"^\s*(class|def)\s", re.MULTILINE)
IMPORT_RE = re.compile(r"^from (src(?:\.\w+)*) import \(?([^)]*?)\)?\s*$", re.MULTILINE | re.DOTALL)


def _code_cells(path):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            yield i, src


def _strip_magics(src):
    return "\n".join("pass" if line.lstrip().startswith(("!", "%")) else line for line in src.splitlines())


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[os.path.basename(p) for p in NOTEBOOKS])
def test_notebook_cells_compile_and_stay_thin(path):
    assert NOTEBOOKS, "no notebooks found"
    for i, src in _code_cells(path):
        compile(_strip_magics(src), f"{os.path.basename(path)}:cell{i}", "exec")
        assert not DEF_RE.search(src), f"{os.path.basename(path)} cell {i} defines a function/class; move it to src/"
        for token in FORBIDDEN:
            assert token not in src, f"{os.path.basename(path)} cell {i} uses {token}; go through src.utils"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[os.path.basename(p) for p in NOTEBOOKS])
def test_notebook_src_imports_resolve(path):
    seen = 0
    for i, src in _code_cells(path):
        for m in IMPORT_RE.finditer(src):
            module = importlib.import_module(m.group(1))
            names = [n.strip().split(" as ")[0] for n in m.group(2).replace("\n", ",").split(",")]
            for name in filter(None, names):
                assert hasattr(module, name), f"{os.path.basename(path)} cell {i}: {m.group(1)} has no '{name}'"
                seen += 1
    assert seen > 0, "notebook imports nothing from src"


def test_notebooks_have_no_outputs():
    for path in NOTEBOOKS:
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                assert not cell.get("outputs"), f"{os.path.basename(path)} cell {i} has stored outputs"
