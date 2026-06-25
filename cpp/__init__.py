"""C++ extension subpackage. Wraps compiled _bandpass.so."""
from ._bandpass import lfilter, lfilter_double, highcut, lowcut  # noqa: F401
