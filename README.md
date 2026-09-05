# dasio

Lightweight, standalone IO + basic processing for DAS (Distributed Acoustic
Sensing) data: vendor HDF5 readers (ASN/OptoDAS, OptaSense/QuantX, AP Sensing, Silixa iDAS),
a `Proc` concatenate/downsample format, a `Basic` format for converted analysis
products, an `Event` format, a numpy `DASdata` container, a file catalog
(`DASdb`), and signal processing including a C++/OpenMP bandpass.

## Install (development)

This package contains a compiled C++ extension. **Always install editable** so
the extension builds in place:

```bash
pip install -e .
```

Build requirements: a C++14 compiler, CMake ≥3.15, and OpenMP (e.g.
`apt-get install build-essential libomp-dev`). pybind11 and scikit-build-core
are pulled automatically by the build.

### Optional extras

Ambient-noise cross-correlation and phase picking have heavier, optional
dependencies, exposed as install extras (still editable):

```bash
pip install -e '.[viewer]'      # interactive viewer (dasio.viewer.view) -> fastplotlib, ipywidgets
#                                 run it from a Jupyter kernel you already have
pip install -e '.[noise]'       # ambient-noise cross-correlation (dasio.noise) -> PyTorch
pip install -e '.[pick]'        # PhaseNet-DAS P/S picking (dasio.pick_phases) -> phasenet (+ PyTorch)
pip install -e '.[noise,pick]'  # both
```

Both are imported lazily, so the rest of dasio works without them installed.

## Quickstart

One file:

```python
from dasio import DASFile

d = DASFile('file.h5').read().to_physical()          # format auto-detected -> microstrain (or /s)
d.bandpass(1.0, 10.0).subtract_common_mode().plot()  # filter -> denoise -> waterfall
```

A directory, queried by time — `DASdb` catalogues the files once, then `read`
stitches any window across file boundaries:

```python
from datetime import datetime, timedelta, timezone
from dasio import DASdb

db = DASdb.from_dir('/data/das/100Hz')       # scan once; format auto-detected
db.to_file('dasdb.csv')                      # DASdb.from_file() next time

t0 = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
d = db.read(t0, t0 + timedelta(seconds=120)) # concatenated, gaps zero-filled
d = d.to_physical()                          # vendor units -> microstrain (or /s)
```

Readers return the instrument's own units (OptaSense counts, AP Sensing
radian/s, Silixa iDAS counts, ASN strain/s); `to_physical()` is the one conversion step, and a no-op
if already converted. Raw OptaSense can roll over at 2**32 — call `d.unwrap()`
first, on the concatenated window rather than per file.

## Command line

Three tools, all run with `python -m`, all with `--help`.

Catalog a directory of DAS files. Run the same command again later and it adds
only what is new:

```bash
python -m dasio.dasdb --from /data/das/100Hz --dasdb dasdb.parquet
```

Downsample to 25 Hz. `--fmax` is the low-pass corner in Hz and sets the output
rate with it, at about 2.5x fmax:

```bash
python -m dasio.desample --from /data/das/raw --to /data/das/25Hz --fmax 10 \
    --dasdb raw.parquet --all --nworkers 4 --nthreads 6
```

`--all` processes every window missing from `--to`. A window counts as done when
its output file exists, so a stopped job restarts by re-running the same
command. After an unclean kill, delete the partial file first: it sits at its
final path and counts as done. To pick one window instead, give `--since` and
`--until` together, with an explicit UTC offset.

Cut one file per event from an event catalog CSV:

```bash
python -m dasio.extract_events --catalog events.csv --dasdb dasdb.parquet \
    --to /data/das/events
```
