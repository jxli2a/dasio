import pathlib


def test_no_realtimemonitor_strings_in_package():
    pkg = pathlib.Path(__file__).resolve().parent.parent / "dasio"
    offenders = []
    for py in pkg.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if "realTimeMonitor" in text:
            offenders.append(str(py.relative_to(pkg)))
    assert not offenders, f"stale realTimeMonitor refs in: {offenders}"
