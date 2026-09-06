"""Every script in examples/ runs. That is the reason they are scripts and
not notebooks: a broken example fails here instead of rotting."""
import runpy
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

SCRIPTS = sorted((Path(__file__).parent.parent / "examples").glob("*.py"))
CACHE = Path.home() / ".cache" / "dasio"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_example_runs(script, monkeypatch):
    # A script that downloads runs only where its data is already cached;
    # the suite never pulls gigabytes.
    if "urlretrieve" in script.read_text() and not any(CACHE.glob("*.h5")):
        pytest.skip("example data not cached")
    matplotlib.use("Agg")
    monkeypatch.setattr(plt, "show", lambda: None)
    runpy.run_path(str(script), run_name="__main__")
