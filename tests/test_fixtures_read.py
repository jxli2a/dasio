from dasio.dasfile import DASFile
from dasio.readers.detector import detect_data_kind
import h5py


def test_detect_kinds(optasense_file, asn_file, apsensing_file, proc_file, event_file):
    expected = {
        optasense_file: "OptaSense", asn_file: "ASN",
        apsensing_file: "APSensing", proc_file: "Proc", event_file: "Event",
    }
    for path, kind in expected.items():
        with h5py.File(path, "r") as f:
            assert detect_data_kind(f) == kind


def test_each_reader_returns_correct_shape(optasense_file, asn_file, apsensing_file,
                                           proc_file, event_file):
    for path in (optasense_file, asn_file, apsensing_file, proc_file, event_file):
        d = DASFile(path).read()
        assert d.data.shape == (d.nx, d.nt)
        assert d.nx == 4 and d.nt == 256
