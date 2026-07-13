from __future__ import annotations

import numpy as np
import pytest

from pygama.math.distributions import gauss_on_step
from pygama.pargen.survival_fractions import compton_sf, get_survival_fraction


def test_compton_sf_data_mask_restricts_population():
    """``data_mask`` must restrict both numerator and denominator.

    Regression test for the bug where ``data_mask`` was applied to the
    passing count but the total was left as the full, unmasked length,
    diluting the survival fraction.
    """
    # First four events are inside the mask; three of them pass (>0).
    cut_param = np.array(
        [5.0, 5.0, 5.0, -5.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    )
    data_mask = np.array(
        [True, True, True, True, False, False, False, False, False, False]
    )

    result = compton_sf(cut_param, low_cut_val=0.0, mode="greater", data_mask=data_mask)

    # 3 of 4 masked-in events survive -> 75 %, independent of the six
    # masked-out events (which would drag it to 90 % under the old bug).
    assert result["sf"] == pytest.approx(75.0)
    assert result["sf_err"] == pytest.approx(100 * np.sqrt((0.75 * 0.25) / 4))


def test_compton_sf_masking_equivalent_to_pre_slicing():
    """Masking is equivalent to counting only the masked sub-population."""
    rng = np.random.default_rng(1234)
    cut_param = rng.normal(size=500)
    data_mask = rng.random(size=500) < 0.6

    masked = compton_sf(cut_param, low_cut_val=0.0, mode="greater", data_mask=data_mask)
    direct = compton_sf(cut_param[data_mask], low_cut_val=0.0, mode="greater")

    assert masked["sf"] == pytest.approx(direct["sf"])
    assert masked["sf_err"] == pytest.approx(direct["sf_err"])


def _toy_peak(rng, n_sig=800, n_bkg=200, mu=1000.0, sigma=2.0, lo=980.0, hi=1020.0):
    """Gaussian signal on a flat background, with a correlated cut parameter."""
    sig_e = rng.normal(mu, sigma, size=n_sig)
    sig_e = sig_e[(sig_e >= lo) & (sig_e <= hi)]
    bkg_e = rng.uniform(lo, hi, size=n_bkg)
    energy = np.concatenate([sig_e, bkg_e])

    # 80 % of signal passes, 50 % of background passes (cut_val = 0, "greater").
    sig_cut = np.where(rng.random(len(sig_e)) < 0.8, 1.0, -1.0)
    bkg_cut = np.where(rng.random(len(bkg_e)) < 0.5, 1.0, -1.0)
    cut_param = np.concatenate([sig_cut, bkg_cut])
    return energy, cut_param, (lo, hi), mu


def _manual_pars(energy, fit_range, peak, sigma=2.0):
    """A fixed-shape parameter set so the efficiency fit skips the staged fit."""
    return {
        "x_lo": fit_range[0],
        "x_hi": fit_range[1],
        "mu": peak,
        "sigma": sigma,
        "hstep": 0.0,
        "n_sig": float(np.sum(np.abs(energy - peak) < 3 * sigma)),
        "n_bkg": float(np.sum(np.abs(energy - peak) >= 3 * sigma)),
    }


def test_get_survival_fraction_excludes_out_of_range_and_masked():
    """Events outside ``fit_range`` or ``data_mask`` must not affect the fit.

    Regression test for two bugs fixed together: the failing selection used
    ``~(passing & data_mask)`` (so masked-out events leaked into the failing
    side), and ``fit_range`` was never enforced on the data entering the
    likelihood.  With both fixed, appending events that are either outside the
    fit window or masked out leaves the survival fraction unchanged.
    """
    rng = np.random.default_rng(42)
    energy, cut_param, fit_range, peak = _toy_peak(rng)
    pars = _manual_pars(energy, fit_range, peak)

    sf_clean, err_clean, _, _ = get_survival_fraction(
        energy,
        cut_param,
        cut_val=0.0,
        peak=peak,
        eres_pars=2.0 * 2.355,
        fit_range=fit_range,
        pars=pars,
        func=gauss_on_step,
    )

    # Pollute the sample with at least as many garbage events as clean ones.
    # Every garbage event MUST be ignored, via one of two independent routes:
    #  - out of fit_range (energies far below/above the window), mask True
    #  - inside the window but masked out
    n_clean = len(energy)
    n_out = n_clean // 2 + 1
    n_masked = n_clean - n_out + 1  # total garbage >= n_clean
    out_e = rng.uniform(900.0, 960.0, size=n_out)  # below the fit window
    masked_e = rng.uniform(fit_range[0], fit_range[1], size=n_masked)  # in-window
    garbage_e = np.concatenate([out_e, masked_e])
    garbage_cut = np.where(rng.random(len(garbage_e)) < 0.5, 1.0, -1.0)
    garbage_mask = np.concatenate(
        [
            np.ones(n_out, dtype=bool),  # kept in mask, excluded by fit_range
            np.zeros(n_masked, dtype=bool),  # masked out
        ]
    )
    assert len(garbage_e) >= n_clean

    poll_energy = np.concatenate([energy, garbage_e])
    poll_cut = np.concatenate([cut_param, garbage_cut])
    poll_mask = np.concatenate([np.ones(n_clean, dtype=bool), garbage_mask])

    sf_poll, err_poll, _, _ = get_survival_fraction(
        poll_energy,
        poll_cut,
        cut_val=0.0,
        peak=peak,
        eres_pars=2.0 * 2.355,
        fit_range=fit_range,
        data_mask=poll_mask,
        pars=pars,
        func=gauss_on_step,
    )

    assert sf_poll == pytest.approx(sf_clean, rel=1e-9)
    assert err_poll == pytest.approx(err_clean, rel=1e-9)
