from dasio.dasfile import DASFile
from dasio.readers.detector import detect_format
import h5py


def test_detect_kinds(optasense_file, asn_file, apsensing_file, silixa_file, proc_file,
                      event_file):
    expected = {
        optasense_file: "OptaSense", asn_file: "ASN", silixa_file: "Silixa",
        apsensing_file: "APSensing", proc_file: "Proc", event_file: "Event",
    }
    for path, kind in expected.items():
        with h5py.File(path, "r") as f:
            assert detect_format(f) == kind


def test_each_reader_returns_correct_shape(optasense_file, asn_file, apsensing_file,
                                           silixa_file, proc_file, event_file):
    for path in (optasense_file, asn_file, apsensing_file, silixa_file, proc_file, event_file):
        d = DASFile(path).read()
        assert d.data.shape == (d.nx, d.nt)
        assert d.nx == 4 and d.nt == 256


def test_every_reader_returns_c_contiguous_data(
        optasense_file, asn_file, apsensing_file, silixa_file, proc_file, event_file):
    """The C++ bandpass reads the raw buffer, so an F-contiguous payload is
    walked in the wrong order and returns a periodic pattern rather than an
    error. Four readers reach (nx, nt) by transposing, and `astype`'s default
    order='K' preserves that layout."""
    from dasio.dasfile import DASFile

    for f in (optasense_file, asn_file, apsensing_file, silixa_file, proc_file, event_file):
        d = DASFile(f).read()
        assert d.data.flags["C_CONTIGUOUS"], f"{DASFile(f).format} payload is not contiguous"
