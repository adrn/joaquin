import numpy as np
import finufft


def nufft_lowpass(x, y, fcut, bad_mask=None, **kwargs):
    assert len(x) == len(y)

    if bad_mask is None:
        bad_mask = np.zeros(len(x), dtype=bool)

    Nx = len(x)
    minx, maxx = np.min(x), np.max(x)

    fac = (maxx - minx) * (Nx + 1) / Nx / np.pi  # TODO: needs justification

    scaled_x = (x - minx) / fac

    shift_y = np.mean(y[~bad_mask])
    shifted_y = y - shift_y

    Nhalf = np.round(fcut * fac * 2*np.pi)  # TODO: needs justification
    Nf = int(2 * Nhalf + 1)  # TODO: needs justification

    f = finufft.nufft1d1(scaled_x[~bad_mask],
                         shifted_y[~bad_mask].astype(np.complex128),
                         n_modes=Nf,
                         **kwargs)

    # TODO: needs justification
    inv_fft_y = finufft.nufft1d2(scaled_x, f, **kwargs) / len(scaled_x) / 2
    assert np.all(np.abs(inv_fft_y.imag) < 1e-10)

    return inv_fft_y.real + shift_y


def renormalize(x, vmin=None, vmax=None):
    x = np.array(x)

    if vmin is None:
        vmin = np.min(x)

    if vmax is None:
        vmax = np.max(x)

    return (x - vmin) / (vmax - vmin)
