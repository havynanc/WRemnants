"""Direct category-specific kinematic reweighting for dimuon resonances.

The weights are built in one histmaker pass from a fine-binned
``(category, ptll, yll, cosThetaStarll)`` histogram and applied as an event
weight in a second pass. Everything here is resonance agnostic apart from the
binning registry below; the category definitions live in the per-resonance
selection modules and are reached through :func:`selection_module`.
"""

import h5py
import hist
import numpy as np

from wremnants.production import jpsi_trigger_selection, upsilon_trigger_selection
from wums import ioutils

SELECTION_MODULES = {
    "jpsi": jpsi_trigger_selection,
    "upsilon": upsilon_trigger_selection,
}

# The rapidity and cos(theta*) binnings are common to all resonances; only the
# pT spectrum differs enough to need its own edges.
RAPIDITY_EDGES = np.asarray(
    [
        -2.4,
        -2.0,
        -1.75,
        -1.5,
        -1.25,
        -1.0,
        -0.75,
        -0.5,
        -0.25,
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.4,
    ]
)
COSTHETA_EDGES = np.round(np.arange(-1.0, 1.0001, 0.1), 8)

PT_EDGES = {
    "jpsi": np.round(
        np.concatenate(
            (
                np.arange(0.0, 10.0, 0.25),
                np.arange(10.0, 20.0, 0.5),
                np.arange(20.0, 30.0, 1.0),
                np.arange(30.0, 60.0001, 2.5),
            )
        ),
        8,
    ),
    "upsilon": np.round(
        np.concatenate(
            (
                np.arange(8.5, 20.0, 0.5),
                np.arange(20.0, 40.0, 1.0),
                np.arange(40.0, 60.0, 2.0),
                np.arange(60.0, 100.1, 5.0),
            )
        ),
        8,
    ),
}

# Step of the fine input axis each coarse pT binning is derived from. The coarse
# edges must be a subset of the fine ones for the rebinning in _rebin to work;
# this is checked once at import time rather than left implicit.
FINE_PT_STEP = {"jpsi": 0.25, "upsilon": 0.5}
FINE_RAPIDITY_STEP = 0.05
FINE_COSTHETA_STEP = 0.05

# Range of the single-muon pT diagnostics. No map is derived on these, so they
# carry no coarse binning to be a subset of; they only have to span the per-muon
# pT the selections can reach, binned like the dimuon pT of the same resonance.
MUON_PT_RANGE = {"jpsi": (0.0, 30.0), "upsilon": (0.0, 60.0)}

AXIS_NAMES = ("triggerCategory", "ptll", "yll", "cosThetaStarll")
INPUT_HIST_NAME = "vReweightInput"
DIAGNOSTIC_HIST_PREFIX = "kinDiagnostics"
FALLBACK_CATEGORY_LABELS = ("all",)
COVERAGE_HIST_NAME = "vReweightCoverage"
PAYLOAD_KEY = "v_reweighting"
PAYLOAD_VERSION = 1

_CS_VARIABLES_DECLARED = False

CATEGORY_COLUMN = "vrw_category"
KINEMATIC_COLUMNS = ("vrw_ptll", "vrw_yll", "vrw_costheta")
MUON_PT_COLUMNS = ("vrw_ptlead", "vrw_ptsublead", "vrw_ptplus", "vrw_ptminus")


def resonances():
    """Return the resonances this module knows how to reweight."""

    return tuple(sorted(PT_EDGES))


def selection_module(resonance):
    """Return the per-resonance module defining the trigger categories."""

    if resonance not in SELECTION_MODULES:
        raise ValueError(
            f"No V-reweighting selections defined for resonance '{resonance}'; "
            f"choose from {tuple(sorted(SELECTION_MODULES))}"
        )
    return SELECTION_MODULES[resonance]


def category_labels(resonance, selection):
    """Return the ordered category labels for *resonance* and *selection*."""

    return selection_module(resonance).category_labels(selection)


def fine_kinematic_axes(resonance):
    """Return the fine-binned kinematic axes for *resonance*.

    These are the axes the maps are derived on, shared with the standalone
    diagnostic histograms so the two are directly comparable.
    """

    pt_edges = _coarse_edges(resonance)["ptll"]
    return [
        hist.axis.Variable(
            _fine_edges(pt_edges[0], pt_edges[-1], FINE_PT_STEP[resonance]),
            name="ptll",
        ),
        hist.axis.Variable(
            _fine_edges(RAPIDITY_EDGES[0], RAPIDITY_EDGES[-1], FINE_RAPIDITY_STEP),
            name="yll",
        ),
        hist.axis.Variable(
            _fine_edges(COSTHETA_EDGES[0], COSTHETA_EDGES[-1], FINE_COSTHETA_STEP),
            name="cosThetaStarll",
        ),
    ]


def fine_muon_pt_axes(resonance):
    """Return the fine-binned single-muon pT axes.

    Both orderings are booked: pT-ordered (leading, subleading), which is what
    the selections cut on, and charge-ordered (mu+, mu-), which is how the
    calibration histogram bins its pt1/pt2 -- the two are not the same muon, so
    neither substitutes for the other.

    Diagnostics only: these are deliberately kept out of
    :func:`fine_kinematic_axes` so the axes the maps are derived on stay
    unchanged.
    """

    if resonance not in MUON_PT_RANGE:
        raise ValueError(
            f"No muon pT diagnostic binning defined for resonance "
            f"'{resonance}'; choose from {resonances()}"
        )
    low, high = MUON_PT_RANGE[resonance]
    edges = _fine_edges(low, high, FINE_PT_STEP[resonance])
    return [
        hist.axis.Variable(edges, name="ptlead"),
        hist.axis.Variable(edges, name="ptsublead"),
        hist.axis.Variable(edges, name="ptplus"),
        hist.axis.Variable(edges, name="ptminus"),
    ]


def category_axis(resonance, selection):
    """Return the category axis for *selection*.

    A *selection* of ``None`` means no categories were defined, which happens
    when the diagnostics are booked on their own; the axis is then a single
    catch-all bin so that every consumer sees the same axis structure.
    """

    labels = (
        FALLBACK_CATEGORY_LABELS
        if selection is None
        else category_labels(resonance, selection)
    )
    return hist.axis.Integer(
        0,
        len(labels),
        name="triggerCategory",
        underflow=False,
        overflow=False,
        metadata={
            "resonance": resonance,
            "selection": selection,
            "labels": labels,
        },
    )


def fine_axes(resonance, selection):
    """Return the fine-binned derivation axes for *resonance*."""

    return [category_axis(resonance, selection), *fine_kinematic_axes(resonance)]


def _declare_cs_variables():
    """Make wrem::csSineCosThetaPhi visible to the interpreter.

    Nothing else the calibration-input histmaker imports pulls in this header,
    and the declaration is deferred so the module stays importable without ROOT.
    """

    global _CS_VARIABLES_DECLARED
    if _CS_VARIABLES_DECLARED:
        return
    import narf

    narf.clingutils.Declare('#include "csVariables.hpp"')
    _CS_VARIABLES_DECLARED = True


def define_kinematics(df, columns):
    """Define the reconstructed dimuon kinematics used by the maps on *df*."""

    _declare_cs_variables()
    muon_mass = 0.1056583755
    return (
        df.Define(
            "vrw_muplus_mom4",
            "ROOT::Math::PtEtaPhiMVector("
            f"{columns['plus_pt']}, {columns['plus_eta']}, "
            f"{columns['plus_phi']}, {muon_mass})",
        )
        .Define(
            "vrw_muminus_mom4",
            "ROOT::Math::PtEtaPhiMVector("
            f"{columns['minus_pt']}, {columns['minus_eta']}, "
            f"{columns['minus_phi']}, {muon_mass})",
        )
        .Define("vrw_dimuon_mom4", "vrw_muplus_mom4 + vrw_muminus_mom4")
        .Define(
            "vrw_ptlead",
            f"std::max(double({columns['plus_pt']}), "
            f"double({columns['minus_pt']}))",
        )
        .Define(
            "vrw_ptsublead",
            f"std::min(double({columns['plus_pt']}), "
            f"double({columns['minus_pt']}))",
        )
        .Define("vrw_ptplus", f"double({columns['plus_pt']})")
        .Define("vrw_ptminus", f"double({columns['minus_pt']})")
        .Define("vrw_ptll", "vrw_dimuon_mom4.pt()")
        .Define("vrw_yll", "vrw_dimuon_mom4.Rapidity()")
        .Define(
            "vrw_cs",
            "wrem::csSineCosThetaPhi(vrw_muplus_mom4, vrw_muminus_mom4)",
        )
        .Define("vrw_costheta", "vrw_cs.costheta")
    )


def book_input(df, resonance, selection, weight_column="weight"):
    """Book the fine-binned input from which the maps are derived."""

    return df.HistoBoost(
        INPUT_HIST_NAME,
        fine_axes(resonance, selection),
        [CATEGORY_COLUMN, *KINEMATIC_COLUMNS, weight_column],
    )


def book_kinematic_diagnostics(
    df, resonance, selection, weight_column="analysis_weight"
):
    """Book one finely binned histogram per reweighting variable.

    The single-muon pT are booked alongside them in both orderings: the maps are
    not derived on those, but the pT cuts the selections apply act on the
    pT-ordered pair, so they are what shows whether a map moved events across a
    threshold, while the charge-ordered pair is what the calibration histogram
    bins in pt1/pt2.

    Each carries the same leading category axis as :func:`fine_axes`, so the
    booking is identical for a single trivial J/psi category and for the Upsilon
    barrel/high split. Unlike :func:`book_input` they are booked with the applied
    weight and are not mutually exclusive with applying a map, so they can be
    produced in either pass and compared directly.
    """

    category = category_axis(resonance, selection)
    axes = [*fine_kinematic_axes(resonance), *fine_muon_pt_axes(resonance)]
    columns = [*KINEMATIC_COLUMNS, *MUON_PT_COLUMNS]
    return [
        df.HistoBoost(
            f"{DIAGNOSTIC_HIST_PREFIX}_{axis.name}",
            [category, axis],
            [CATEGORY_COLUMN, column, weight_column],
        )
        for axis, column in zip(axes, columns)
    ]


def build_payload(data_hist, mc_hist, resonance, channel):
    """Build direct data/MC shape weights independently in every category."""

    _validate_inputs(data_hist, mc_hist, resonance)
    data_hist = _rebin(data_hist, resonance)
    mc_hist = _rebin(mc_hist, resonance)
    data = data_hist.values()
    mc = mc_hist.values()

    usable_values = (data > 0.0) & (mc > 0.0)
    weight_values = np.ones_like(mc)
    summary = []
    labels = tuple(data_hist.axes[0].metadata["labels"])

    for category, label in enumerate(labels):
        usable = usable_values[category]
        data_supported = float(data[category][usable].sum())
        mc_supported = float(mc[category][usable].sum())
        if data_supported <= 0.0 or mc_supported <= 0.0:
            raise ValueError(f"Category '{label}' has no common data/MC support")

        weights = weight_values[category]
        weights[usable] = (data[category][usable] / data_supported) / (
            mc[category][usable] / mc_supported
        )
        weighted_supported = float((mc[category][usable] * weights[usable]).sum())
        if not np.isclose(weighted_supported, mc_supported, rtol=1e-12):
            raise RuntimeError(f"Weight normalization failed for category '{label}'")

        mc_total = float(mc_hist[{"triggerCategory": category}].values(flow=True).sum())
        data_total = float(
            data_hist[{"triggerCategory": category}].values(flow=True).sum()
        )
        mc_fallback = mc_total - mc_supported
        data_uncovered = data_total - data_supported
        quantiles = np.quantile(weights[usable], [0.0, 0.5, 0.9, 0.99, 1.0])
        summary.append(
            {
                "category": label,
                "data_total": data_total,
                "mc_total": mc_total,
                "data_supported": data_supported,
                "mc_supported": mc_supported,
                "data_uncovered_fraction": data_uncovered / data_total,
                "mc_fallback_fraction": mc_fallback / mc_total,
                "usable_cells": int(usable.sum()),
                "total_cells": int(usable.size),
                "weight_min": float(quantiles[0]),
                "weight_median": float(quantiles[1]),
                "weight_q90": float(quantiles[2]),
                "weight_q99": float(quantiles[3]),
                "weight_max": float(quantiles[4]),
            }
        )

    correction, usable = _output_histograms(data_hist, resonance)
    correction.values()[..., 0] = weight_values
    usable.values()[..., 0] = usable_values
    return {
        "version": PAYLOAD_VERSION,
        "resonance": resonance,
        "selection": data_hist.axes[0].metadata["selection"],
        "channel": channel,
        "category_labels": labels,
        "prescription": "(data_bin/data_supported)/(mc_bin/mc_supported)",
        "background_subtracted": False,
        "fallback": "unit weight outside common nonzero data/MC support",
        "correction": correction,
        "usable": usable,
        "summary": summary,
    }


def load_payload(path, resonance, selection, channel):
    """Load and validate a payload for *resonance*, *selection* and *channel*."""

    with h5py.File(path, "r") as h5file:
        if PAYLOAD_KEY not in h5file:
            raise ValueError(f"Correction file '{path}' has no '{PAYLOAD_KEY}' payload")
        payload = ioutils.pickle_load_h5py(h5file[PAYLOAD_KEY])
        for name in ("correction", "usable"):
            value = payload.get(name)
            payload[name] = value.get() if hasattr(value, "get") else value

    _validate_payload(payload, resonance, selection, channel)
    return payload


def make_helpers(path, resonance, selection, channel):
    """Return validated correction and common-support lookup helpers."""

    from wremnants.production import correctionsTensor_helper

    payload = load_payload(path, resonance, selection, channel)
    return (
        correctionsTensor_helper.makeCorrectionsTensor(payload["correction"]),
        correctionsTensor_helper.makeCorrectionsTensor(payload["usable"]),
        payload,
    )


def define_weights(df, correction_helper, usable_helper):
    """Define the applied event weight and common-support flag on *df*.

    The category column is cast to ``double`` because makeCorrectionsTensor only
    emits an ``int`` column type for axes whose name contains "charge".
    """

    helper_inputs = ["vrw_category_value", *KINEMATIC_COLUMNS]
    return (
        df.Define("vrw_category_value", f"double({CATEGORY_COLUMN})")
        .Define("vrw_unit_weight", "1.0")
        .Define(
            "vrw_weight_tensor",
            correction_helper,
            [*helper_inputs, "weight"],
        )
        .Define(
            "vrw_usable_tensor",
            usable_helper,
            [*helper_inputs, "vrw_unit_weight"],
        )
        .Define("analysis_weight", "vrw_weight_tensor(0)")
        .Define("vrw_usable", "int(vrw_usable_tensor(0) > 0.5)")
    )


def book_coverage(df, resonance, selection, weight_column="weight"):
    """Book category-resolved counts inside and outside common map support."""

    axes = [
        hist.axis.Integer(
            0,
            len(category_labels(resonance, selection)),
            name="triggerCategory",
            underflow=False,
            overflow=False,
        ),
        hist.axis.Integer(
            0,
            2,
            name="usable",
            underflow=False,
            overflow=False,
        ),
    ]
    return df.HistoBoost(
        COVERAGE_HIST_NAME,
        axes,
        [CATEGORY_COLUMN, "vrw_usable", weight_column],
    )


def _fine_edges(low, high, step):
    return np.round(np.arange(low, high + 1e-4, step), 8)


def _coarse_edges(resonance):
    if resonance not in PT_EDGES:
        raise ValueError(
            f"No V-reweighting binning defined for resonance '{resonance}'; "
            f"choose from {resonances()}"
        )
    return {
        "ptll": PT_EDGES[resonance],
        "yll": RAPIDITY_EDGES,
        "cosThetaStarll": COSTHETA_EDGES,
    }


def _rebin(source, resonance):
    result = source
    for axis_name, axis_edges in _coarse_edges(resonance).items():
        result = result[{axis_name: hist.rebin(edges=axis_edges)}]
    return result


def _output_histograms(source, resonance):
    category = source.axes["triggerCategory"]
    coarse = _coarse_edges(resonance)
    axes = [
        hist.axis.Integer(
            int(category.edges[0]),
            int(category.edges[-1]),
            name="triggerCategory",
            underflow=False,
            overflow=False,
            metadata=category.metadata,
        ),
        hist.axis.Variable(coarse["ptll"], name="ptll"),
        hist.axis.Variable(coarse["yll"], name="yll"),
        hist.axis.Variable(coarse["cosThetaStarll"], name="cosThetaStarll"),
        hist.axis.Integer(0, 1, name="vars", underflow=False, overflow=False),
    ]
    correction = hist.Hist(*axes, storage=hist.storage.Double())
    usable = correction.copy()
    # Flow bins carry the fallback: unit weight, flagged as outside support.
    correction.values(flow=True)[...] = 1.0
    usable.values(flow=True)[...] = 0.0
    return correction, usable


def _validate_inputs(data_hist, mc_hist, resonance):
    for label, source in (("data", data_hist), ("simulation", mc_hist)):
        if tuple(source.axes.name) != AXIS_NAMES:
            raise ValueError(
                f"Unexpected {label} axes {tuple(source.axes.name)}; "
                f"expected {AXIS_NAMES}"
            )
        if not np.all(np.isfinite(source.values(flow=True))):
            raise ValueError(f"The {label} input contains non-finite values")
        if np.any(source.values(flow=True) < 0.0):
            raise ValueError(f"The {label} input contains negative values")
    if data_hist.axes != mc_hist.axes:
        raise ValueError("Data and simulation input axes differ")
    metadata = data_hist.axes["triggerCategory"].metadata
    expected_keys = {"resonance", "selection", "labels"}
    if not isinstance(metadata, dict) or not expected_keys <= metadata.keys():
        raise ValueError("The triggerCategory axis lacks selection metadata")
    if metadata["resonance"] != resonance:
        raise ValueError(
            f"The inputs were produced for resonance '{metadata['resonance']}', "
            f"not '{resonance}'"
        )


def _validate_payload(payload, resonance, selection, channel):
    if not isinstance(payload, dict):
        raise ValueError("The V-reweighting payload is not a dictionary")
    if payload.get("version") != PAYLOAD_VERSION:
        raise ValueError(
            f"Unsupported V-reweighting payload version "
            f"{payload.get('version')}; expected {PAYLOAD_VERSION}"
        )
    for key, expected in (
        ("resonance", resonance),
        ("selection", selection),
        ("channel", channel),
    ):
        if payload.get(key) != expected:
            raise ValueError(
                f"Correction {key} '{payload.get(key)}' does not match "
                f"requested {key} '{expected}'"
            )

    expected_labels = category_labels(resonance, selection)
    if tuple(payload.get("category_labels", ())) != expected_labels:
        raise ValueError(
            f"Correction categories {payload.get('category_labels')} do not match "
            f"expected categories {expected_labels}"
        )

    expected_axes = (*AXIS_NAMES, "vars")
    for name in ("correction", "usable"):
        source = payload.get(name)
        if not isinstance(source, hist.Hist):
            raise ValueError(f"Payload entry '{name}' is not a histogram")
        if tuple(source.axes.name) != expected_axes:
            raise ValueError(
                f"Unexpected {name} axes {tuple(source.axes.name)}; "
                f"expected {expected_axes}"
            )
        if not np.all(np.isfinite(source.values(flow=True))):
            raise ValueError(f"Payload entry '{name}' contains non-finite values")

    correction = payload["correction"]
    usable = payload["usable"]
    if correction.axes != usable.axes:
        raise ValueError("Correction and usable-mask axes differ")
    category = correction.axes["triggerCategory"]
    metadata = category.metadata
    if (
        not isinstance(metadata, dict)
        or metadata.get("resonance") != resonance
        or metadata.get("selection") != selection
        or tuple(metadata.get("labels", ())) != expected_labels
        or not np.array_equal(category.edges, np.arange(len(expected_labels) + 1))
    ):
        raise ValueError("The triggerCategory axis is inconsistent with the selection")
    expected_edges = {**_coarse_edges(resonance), "vars": np.asarray([0.0, 1.0])}
    for axis_name, edges in expected_edges.items():
        if not np.array_equal(correction.axes[axis_name].edges, edges):
            raise ValueError(f"Unexpected binning for payload axis '{axis_name}'")
    if np.any(correction.values(flow=True) < 0.0):
        raise ValueError("The correction contains negative weights")
    usable_values = usable.values(flow=True)
    if not np.all((usable_values == 0.0) | (usable_values == 1.0)):
        raise ValueError("The usable mask contains values other than zero or one")


def _check_binning_contract():
    """The coarse edges must be a subset of the fine ones for _rebin to work."""

    for resonance, pt_edges in PT_EDGES.items():
        pairs = (
            (pt_edges, FINE_PT_STEP[resonance]),
            (RAPIDITY_EDGES, FINE_RAPIDITY_STEP),
            (COSTHETA_EDGES, FINE_COSTHETA_STEP),
        )
        for coarse, step in pairs:
            fine = _fine_edges(coarse[0], coarse[-1], step)
            if not np.all(np.isin(np.round(coarse, 8), fine)):
                raise AssertionError(
                    f"V-reweighting coarse edges for '{resonance}' are not a "
                    f"subset of the fine axis with step {step}"
                )


_check_binning_contract()
