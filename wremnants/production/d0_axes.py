"""Template-axis definitions shared by the D0 histmaker and its tests."""

import hist

# Upper bound of the mRpi (K<->pi swapped reduced mass) axis. The swap scales the
# reduced mass by ~(M_pi/M_K)^2 ~ 0.08 relative to mRK, with an energy-ratio tail
# extending a bit past that, so the populated range is ~[0, 0.20] (see the
# --kinDiagnostics study). The mRpi template axis is uniform over [0, MRPI_MAX].
MRPI_MAX = 0.20

DEFAULT_MRK_EDGES = (
    0.00,
    0.06,
    0.10,
    0.14,
    0.18,
    0.22,
    0.26,
    0.30,
    0.34,
    0.38,
    0.42,
    0.46,
    0.50,
    0.60,
    0.70,
    0.85,
    1.05,
    1.30,
    1.55,
    1.80,
)


def _make_eta_axis(eta_bins, name):
    return hist.axis.Regular(eta_bins, -2.4, 2.4, name=name)


def _validate_edges(edges, name):
    """Return the edges as a list after checking they are usable."""
    e = [float(x) for x in edges]
    if len(e) < 2:
        raise ValueError(f"{name} requires at least 2 edges, got {len(e)}")
    if any(b <= a for a, b in zip(e, e[1:])):
        raise ValueError(f"{name} edges must be strictly increasing, got {e}")
    return e


def _make_mrk_axis(mrk_bins, mrk_edges):
    """The mRK reduced-mass axis. Priority: explicit edges > regular bins over the
    mRK range > legacy variable-width binning (the 3D default)."""
    if mrk_edges is not None:
        return hist.axis.Variable(_validate_edges(mrk_edges, "mrk_edges"), name="mRK")
    if mrk_bins is not None:
        return hist.axis.Regular(
            mrk_bins, DEFAULT_MRK_EDGES[0], DEFAULT_MRK_EDGES[-1], name="mRK"
        )
    return hist.axis.Variable(DEFAULT_MRK_EDGES, name="mRK")


def _make_mrpi_axis(mrpi_bins, mrpi_edges):
    """The mRpi reduced-mass axis (5D only). Priority: explicit edges > uniform
    bins over [0, MRPI_MAX]. mRpi has no legacy default; one of the two is required."""
    if mrpi_edges is not None:
        return hist.axis.Variable(_validate_edges(mrpi_edges, "mrpi_edges"), name="mRpi")
    return hist.axis.Regular(mrpi_bins, 0.0, MRPI_MAX, name="mRpi")


def make_d0_template_axes(
    eta_bins=24,
    mrk_bins=None,
    pion_axes=False,
    eta_pi_bins=None,
    mrpi_bins=None,
    mrk_edges=None,
    mrpi_edges=None,
):
    """Return the calibration-template axes.

    Default (3D): (etaK, mRK, D0mass). With ``pion_axes=True`` the pion
    analogues are inserted before the shape axis, giving the 5D layout
    (etaK, mRK, etaPi, mRpi, D0mass). ``eta_pi_bins`` defaults to ``eta_bins``.
    D0mass is always the last (shape) axis.

    Each reduced-mass axis can be given either as a number of uniform bins
    (``mrk_bins`` over the mRK range, ``mrpi_bins`` over [0, MRPI_MAX]) or as
    explicit ``mrk_edges`` / ``mrpi_edges``; the two are mutually exclusive per
    axis. In 5D mode each of mRK and mRpi must be specified one way or the other
    (bins or edges); the legacy variable-width mRK binning is only the 3D default.
    """
    if eta_bins <= 0:
        raise ValueError("eta_bins must be positive")
    if mrk_bins is not None and mrk_bins <= 0:
        raise ValueError("mrk_bins must be positive when specified")
    if eta_pi_bins is not None and eta_pi_bins <= 0:
        raise ValueError("eta_pi_bins must be positive when specified")
    if mrpi_bins is not None and mrpi_bins <= 0:
        raise ValueError("mrpi_bins must be positive when specified")
    if mrk_bins is not None and mrk_edges is not None:
        raise ValueError("Specify only one of mrk_bins or mrk_edges")
    if mrpi_bins is not None and mrpi_edges is not None:
        raise ValueError("Specify only one of mrpi_bins or mrpi_edges")
    if pion_axes:
        if mrk_bins is None and mrk_edges is None:
            raise ValueError(
                "5D templates (pion_axes=True) require mrk_bins or mrk_edges"
            )
        if mrpi_bins is None and mrpi_edges is None:
            raise ValueError(
                "5D templates (pion_axes=True) require mrpi_bins or mrpi_edges"
            )

    eta_axis = _make_eta_axis(eta_bins, "etaK")
    mrk_axis = _make_mrk_axis(mrk_bins, mrk_edges)
    mass_axis = hist.axis.Regular(25, 1.8, 1.93, name="D0mass")

    if not pion_axes:
        return eta_axis, mrk_axis, mass_axis

    eta_pi_axis = _make_eta_axis(eta_pi_bins if eta_pi_bins is not None else eta_bins, "etaPi")
    mrpi_axis = _make_mrpi_axis(mrpi_bins, mrpi_edges)
    return eta_axis, mrk_axis, eta_pi_axis, mrpi_axis, mass_axis
