def test_package_imports_and_has_version():
    import dasio
    assert dasio.__version__ == "0.1.0"
    # DASdata is part of the eager public surface.
    assert "DASdata" in dasio.__all__


def test_compiled_bandpass_extension_imports():
    # The C++ extension must build + import from a clean checkout.
    from dasio.cpp import lfilter, lfilter_double, highcut, lowcut  # noqa: F401
