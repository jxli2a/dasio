import numpy as np
import h5py
import pytest

NX, NT, FS = 4, 256, 100.0
T0_US = 1_700_000_000_000_000  # microseconds since epoch


@pytest.fixture
def optasense_file(tmp_path):
    p = tmp_path / "opta.hdf5"
    with h5py.File(p, "w") as f:
        acq = f.create_group("Acquisition")
        acq.attrs["GaugeLength"] = 10.0
        acq.attrs["SpatialSamplingInterval"] = 1.0
        raw = acq.create_group("Raw[0]")
        raw.attrs["OutputDataRate"] = FS
        raw.create_dataset("RawData", data=np.arange(NX * NT, dtype=np.int32).reshape(NX, NT))
        raw.create_dataset("RawDataTime", data=T0_US + (np.arange(NT) * (1e6 / FS)).astype(np.int64))
        custom = acq.create_group("Custom")
        custom.attrs["Fibre Refractive Index"] = 1.468
        custom.attrs["Laser Wavelength (nm)"] = 1550.0
        custom.attrs["FPGA Drawing Number"] = 7804701
        custom.attrs["FPGA Version"] = "2.0"
        custom.attrs["Num Output Channels"] = NX
    return p


@pytest.fixture
def asn_file(tmp_path):
    p = tmp_path / "asn.hdf5"
    with h5py.File(p, "w") as f:
        f.create_group("acqSpec")  # detection marker
        f.create_dataset("data", data=np.ones((NT, NX), dtype=np.float32))
        h = f.create_group("header")
        h.create_dataset("sensitivities", data=np.float32(1.0e8))
        h.create_dataset("dataScale", data=np.float32(1.0))
        h.create_dataset("dt", data=np.float64(1.0 / FS))
        h.create_dataset("dx", data=np.float64(2.0))
        h.create_dataset("channels", data=np.arange(NX, dtype=np.int64))
        h.create_dataset("gaugeLength", data=np.float64(10.0))
        h.create_dataset("time", data=np.float64(T0_US / 1e6))
        dr = h.create_group("dimensionRanges").create_group("dimension0")
        dr.create_dataset("unitScale", data=np.float64(1.0 / FS))
        f.create_group("timing").create_dataset("sampleSkew", data=np.float64(0.0))
    return p


@pytest.fixture
def apsensing_file(tmp_path):
    p = tmp_path / "aps.hdf5"
    with h5py.File(p, "w") as f:
        ts = f.create_group("Timestamps")
        ts.create_dataset("DataTimestamps",
                          data=(T0_US + (np.arange(NT) * (1e6 / FS))).astype(np.int64)[:, None])
        f.create_dataset("DAS", data=np.ones((NT, NX), dtype=np.float32))  # time-first
        ps = f.create_group("ProcessingServer")
        ps.create_dataset("DataRate", data=np.array([FS]))
        ps.create_dataset("SpatialSampling", data=np.array([2.0]))
        ps.create_dataset("GaugeLength", data=np.array([10.0]))
        ps.create_dataset("RadiansToNanoStrain", data=np.array([100.0]))
    return p


@pytest.fixture
def proc_file(tmp_path):
    p = tmp_path / "proc.hdf5"
    with h5py.File(p, "w") as f:
        d = f.create_dataset("Data", data=np.ones((NX, NT), dtype=np.float32))
        d.attrs["nCh"] = NX
        d.attrs["nt"] = NT
        d.attrs["dt"] = 1.0 / FS
        d.attrs["fs"] = FS
        d.attrs["dCh"] = 2.0
        d.attrs["GaugeLength"] = 10.0
        d.attrs["startTime"] = "2023-11-14T22:13:20.000000+00:00"
        d.attrs["endTime"] = "2023-11-14T22:13:22.550000+00:00"
        f.create_dataset("Acquisition_origin", data=np.float32(0.0))  # no marker -> Unknown origin
    return p


@pytest.fixture
def event_file(tmp_path):
    p = tmp_path / "event.hdf5"
    with h5py.File(p, "w") as f:
        d = f.create_dataset("data", data=np.ones((NX, NT), dtype=np.float32))
        d.attrs["begin_time"] = "2023-11-14T22:13:20.000+00:00"
        d.attrs["end_time"] = "2023-11-14T22:13:22.550+00:00"
        d.attrs["dt_s"] = 1.0 / FS
        d.attrs["dx_m"] = 2.0
        d.attrs["event_id"] = "test01"
        d.attrs["event_time"] = "2023-11-14T22:13:21.000+00:00"
        d.attrs["event_time_index"] = 100
        d.attrs["time_before"] = 1.0
        d.attrs["time_after"] = 1.55
        d.attrs["latitude"] = 64.0
        d.attrs["longitude"] = -17.0
        d.attrs["depth_km"] = 5.0
        d.attrs["magnitude"] = 3.0
        d.attrs["unit"] = "microstrain/s"
    return p
