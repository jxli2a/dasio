# dasio

Lightweight, standalone IO + basic processing for DAS (Distributed Acoustic
Sensing) data: vendor HDF5 readers (ASN/OptoDAS, OptaSense/QuantX, AP Sensing),
a `Proc` processed format, an `Event` format, a numpy `DASdata` container, a
file catalog (`DASdb`), and signal processing including a C++/OpenMP bandpass.

## Install (development)

This package contains a compiled C++ extension. **Always install editable** so
the extension builds in place:

```bash
pip install -e .
```

Build requirements: a C++14 compiler, CMake ≥3.15, and OpenMP (e.g.
`apt-get install build-essential libomp-dev`). pybind11 and scikit-build-core
are pulled automatically by the build.

## Quickstart

```python
from dasio import DASFile

d = DASFile('file.h5').read()                        # auto-detects the vendor format -> DASdata
d.bandpass(1.0, 10.0).subtract_common_mode().plot()  # filter -> denoise -> waterfall
```
