"""A synthetic record, so a snippet or a notebook has something to run on.

Built in the order the physics happens: a wavefield along the fibre,
differentiated into strain, then what the interrogator adds on top. Strain,
not strain rate: that is what the OptaSense archive stores, and
`differentiate()` is one call away.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.signal import fftconvolve, gausspulse

from .dasdata import DASdata

# Cars on the road beside the fibre: (speed m/s, metres along at t=0,
# weight N). One drives right and two drive left, so the streaks cross.
TRAFFIC = ((25.0, 200.0, 15e3), (-22.0, 2600.0, 18e3), (-18.0, 2200.0, 12e3))
DEPTH, OFFSET = 1.0, 5.0    # fibre burial depth and distance from the wheel path, m
SHEAR, NU = 80e6, 0.3       # road bed: shear modulus Pa, Poisson's ratio
QUAKE = 2.0                 # peak S strain in microstrain, an M~3 at this range
AMBIENT = 0.02              # ambient strain noise, microstrain rms


def strain(w):
    """Displacement along the fibre to the strain the fibre feels."""
    return np.gradient(w, axis=0)


def boussinesq_strain(x, y, z, load, shear=SHEAR, nu=NU):
    """Strain along x at (x, y, z) below a vertical point `load` on the
    surface of a half-space: d/dx of Boussinesq's horizontal displacement.
    Checked against the numerical derivative to 1e-6."""
    r = np.sqrt(x * x + y * y + z * z)
    return load / (4 * np.pi * shear) * (
        z / r ** 3 - 3 * x * x * z / r ** 5
        - (1 - 2 * nu) * (1 / (r * (r + z)) - x * x * (2 * r + z) / (r ** 3 * (r + z) ** 2)))


def scatter_coda(
        x, src, t0, fs, nt, vp, vs, fc, rng, a=40.0, kappa=0.3, cell=25.0,
        v_slow=500.0, q=150.0):
    """Coda by single (Born) scattering off a random near-surface medium.

    The medium is a velocity perturbation on a grid with a von Karman
    autocorrelation: length `a`, Hurst exponent `kappa`. Its power spectrum,
    `(1 + k^2 a^2)^-(kappa+1)`, is what sets how strongly each wavelength
    scatters, and the power-law tail is why the crust scatters at every
    scale where a Gaussian medium would go quiet. Every cell scatters the
    incident wave once -- no multiple scattering, which is what Born means.

    Every arrival is the same wavelet at a different delay, so the delays go
    into a spike train and one convolution serves every channel: cells x nx
    x nt wavelet evaluations become cells x nx additions and one FFT.
    """
    # Wide enough that the slowest paths still have cells to come from at
    # the end of the record -- coda lasts as long as the medium is big.
    gx = np.arange(x[0] - 1000.0, x[-1] + 1000.0, cell)
    gy = np.arange(-1000.0, 1000.0, cell)
    kx = 2 * np.pi * np.fft.fftfreq(gx.size, cell)
    ky = 2 * np.pi * np.fft.fftfreq(gy.size, cell)
    spectrum = (1 + (kx[:, None] ** 2 + ky ** 2) * a ** 2) ** (-(kappa + 1) / 2)
    white = np.fft.fft2(rng.standard_normal((gx.size, gy.size)))
    dv = np.fft.ifft2(white * spectrum).real
    dv /= dv.std()
    sx, sy = (g.ravel() for g in np.meshgrid(gx, gy, indexing='ij'))
    dv = dv.ravel()

    d_si = np.hypot(sx - src[0], sy - src[1])           # source -> cell
    d_ir = np.hypot(x[:, None] - sx, sy)                # cell -> channel
    rows = np.broadcast_to(np.arange(x.size)[:, None], d_ir.shape)
    spikes = np.zeros(x.size * nt)
    # Three ways round, as (v_in, v_out, outgoing spreading, weight): P->P,
    # which puts a fast coda right behind P; S->S, the body scattering that
    # carries most of a real coda; and S converting to slow surface waves,
    # which is what draws the Vs off strong cells. A body wave spreads as
    # 1/r on the way out, a surface wave as 1/sqrt(r).
    modes = ((vp, vp, 1.0, 0.12), (vs, vs, 1.0, 1.0), (vs, v_slow, 0.5, 0.5))
    for v_in, v_out, spread, weight in modes:
        tau = t0 + d_si / v_in + d_ir / v_out
        amp = weight * dv / (d_si * (d_ir + 1.0) ** spread)
        amp = amp * np.exp(-np.pi * fc * (tau - t0) / q)        # anelastic loss
        j = np.rint(tau * fs).astype(int)
        ok = (j >= 0) & (j < nt)
        spikes += np.bincount(rows[ok] * nt + j[ok], amp[ok], minlength=spikes.size)
    wavelet = gausspulse(np.arange(-25, 26) / fs, fc)    # +-0.25 s is all it has
    return fftconvolve(
        spikes.reshape(x.size, nt), wavelet[None, :], mode='same', axes=1)


def demo_record(nx=300, nt=6000, fs=100.0, dx=10.0, seed=0) -> DASdata:
    """A roadside fibre: a local earthquake with its coda, three vehicles, and
    the noise a real interrogator writes underneath.

    The defaults are a compromise the signals force. P and S only separate if
    the source is far enough away (2.4 km: four wavelet periods apart), the
    hyperbolas only curve if the array is long enough to see it, and a car at
    25 m/s needs a minute to cross any of that -- so 3 km of fibre and 60 s.

    Amplitudes are physical: peak S strain for the quake, axle weights on a
    road bed for the cars, an rms for the noise, all in microstrain.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(nt) / fs
    x = np.arange(nx) * dx
    vp, vs, fc, t0 = 5000.0, 5000.0 / 1.73, 12.0, 5.0
    src = (x.mean(), 2400.0)                    # source 2.4 km off the fibre

    # --- the wavefield, as ground motion --------------------------------
    d = np.hypot(src[1], x - src[0])[:, None]
    quake = 0.5 * gausspulse(t - d / vp - t0, fc)          # P, the weaker
    quake += gausspulse(t - d / vs - t0, fc)               # S
    coda = scatter_coda(x, src, t0, fs, nt, vp, vs, fc, rng)

    # A car is a weight rolling along the surface, and what the fibre feels is
    # the quasi-static strain of the ground under it, carried along at the
    # car's speed. It comes out as strain directly, in microstrain, so it
    # joins after the derivative.
    cars = np.zeros((nx, nt))
    for speed, start, load in TRAFFIC:
        away = x[:, None] - start - speed * t               # metres from the car
        cars += 1e6 * boussinesq_strain(away, OFFSET, DEPTH, load)

    # Ambient noise is red in what the interrogator records.
    ambient = np.cumsum(rng.standard_normal((nx, nt)), axis=1)

    # Everything in microstrain. The quake is pinned to its peak S strain and
    # the coda to a quarter of that; the cars already carry their own units.
    quake, coda = strain(quake), strain(coda)
    data = (QUAKE * quake / np.abs(quake).max()
            + 0.25 * QUAKE * coda / np.abs(coda).max()
            + cars
            + AMBIENT * ambient / ambient.std())

    # --- what the interrogator adds -------------------------------------
    data *= rng.lognormal(0.0, 0.35, (nx, 1))       # per-channel optical fading
    data += 0.3 * np.cumsum(rng.standard_normal(nt)) / np.sqrt(nt)    # laser drift
    data[rng.choice(nx, nx // 40, replace=False)] = 0.0               # dead channels

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return DASdata(
        data=data.astype(np.float32),
        fs=fs, dt=1.0 / fs, nt=nt, nx=nx, dx=dx,
        begin_time=start, end_time=start + timedelta(seconds=(nt - 1) / fs),
        gauge_length_m=10.0, format='Basic', origin='synthetic',
        units='microstrain',
    )
