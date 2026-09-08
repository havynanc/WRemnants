"""Direct category-specific kinematic reweighting for Upsilon simulation."""

import h5py
import hist
import numpy as np
from wums import ioutils

from wremnants.production import upsilon_trigger_selection

PT_EDGES = np.concatenate(
    (
        np.arange(8.5, 20.0, 0.5),
        np.arange(20.0, 40.0, 1.0),
        np.arange(40.0, 60.0, 2.0),
        np.arange(60.0, 100.1, 5.0),
    )
)
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
AXIS_NAMES = ("triggerCategory", "ptll", "yll", "cosThetaStarll")
PAYLOAD_KEY = "upsilon_v_reweighting"
PAYLOAD_VERSION = 1


def build_payload(data_hist, mc_hist):
    """Build direct data/MC shape weights independently in every category."""

    _validate_inputs(data_hist, mc_hist)
    data_hist = _rebin(data_hist)
    mc_hist = _rebin(mc_hist)
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

    correction, usable = _output_histograms(data_hist)
    correction.values()[..., 0] = weight_values
    usable.values()[..., 0] = usable_values
    return {
        "version": PAYLOAD_VERSION,
        "selection": data_hist.axes[0].metadata["selection"],
        "category_labels": labels,
        "prescription": "(data_bin/data_supported)/(mc_bin/mc_supported)",
        "background_subtracted": False,
        "fallback": "unit weight outside common nonzero data/MC support",
        "correction": correction,
        "usable": usable,
        "summary": summary,
    }


def load_payload(path, selection):
    """Load and validate a V-reweighting payload for *selection*."""

    with h5py.File(path, "r") as h5file:
        if PAYLOAD_KEY not in h5file:
            raise ValueError(f"Correction file '{path}' has no '{PAYLOAD_KEY}' payload")
        payload = ioutils.pickle_load_h5py(h5file[PAYLOAD_KEY])
        for name in ("correction", "usable"):
            value = payload.get(name)
            payload[name] = value.get() if hasattr(value, "get") else value

    _validate_payload(payload, selection)
    return payload


def make_helpers(path, selection):
    """Return validated correction and common-support lookup helpers."""

    from wremnants.production import correctionsTensor_helper

    payload = load_payload(path, selection)
    return (
        correctionsTensor_helper.makeCorrectionsTensor(payload["correction"]),
        correctionsTensor_helper.makeCorrectionsTensor(payload["usable"]),
        payload,
    )


def define_weights(df, correction_helper, usable_helper):
    """Define the applied event weight and common-support flag on *df*."""

    helper_inputs = [
        "upsilon_trigger_category_value",
        "upsilon_ptll",
        "upsilon_yll",
        "upsilon_costheta",
    ]
    return (
        df.Define(
            "upsilon_trigger_category_value",
            "double(upsilon_trigger_category)",
        )
        .Define("upsilon_v_unit_weight", "1.0")
        .Define(
            "upsilon_v_weight_tensor",
            correction_helper,
            [*helper_inputs, "weight"],
        )
        .Define(
            "upsilon_v_usable_tensor",
            usable_helper,
            [*helper_inputs, "upsilon_v_unit_weight"],
        )
        .Define("analysis_weight", "upsilon_v_weight_tensor(0)")
        .Define(
            "upsilon_v_reweight_usable",
            "int(upsilon_v_usable_tensor(0) > 0.5)",
        )
    )


def book_coverage(df, selection, weight_column="weight"):
    """Book category-resolved counts inside and outside common map support."""

    axes = [
        hist.axis.Integer(
            0,
            len(upsilon_trigger_selection.category_labels(selection)),
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
        "upsilonVReweightCoverage",
        axes,
        ["upsilon_trigger_category", "upsilon_v_reweight_usable", weight_column],
    )


def _rebin(source):
    edges = {
        "ptll": PT_EDGES,
        "yll": RAPIDITY_EDGES,
        "cosThetaStarll": COSTHETA_EDGES,
    }
    result = source
    for axis_name, axis_edges in edges.items():
        result = result[{axis_name: hist.rebin(edges=axis_edges)}]
    return result


def _output_histograms(source):
    category = source.axes["triggerCategory"]
    axes = [
        hist.axis.Integer(
            int(category.edges[0]),
            int(category.edges[-1]),
            name="triggerCategory",
            underflow=False,
            overflow=False,
            metadata=category.metadata,
        ),
        hist.axis.Variable(PT_EDGES, name="ptll"),
        hist.axis.Variable(RAPIDITY_EDGES, name="yll"),
        hist.axis.Variable(COSTHETA_EDGES, name="cosThetaStarll"),
        hist.axis.Integer(0, 1, name="vars", underflow=False, overflow=False),
    ]
    correction = hist.Hist(*axes, storage=hist.storage.Double())
    usable = correction.copy()
    correction.values(flow=True)[...] = 1.0
    usable.values(flow=True)[...] = 0.0
    return correction, usable


def _validate_inputs(data_hist, mc_hist):
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
    if not isinstance(metadata, dict) or not {"selection", "labels"} <= metadata.keys():
        raise ValueError("The triggerCategory axis lacks selection metadata")


def _validate_payload(payload, selection):
    if not isinstance(payload, dict):
        raise ValueError("The Upsilon V-reweighting payload is not a dictionary")
    if payload.get("version") != PAYLOAD_VERSION:
        raise ValueError(
            f"Unsupported Upsilon V-reweighting payload version "
            f"{payload.get('version')}; expected {PAYLOAD_VERSION}"
        )
    if payload.get("selection") != selection:
        raise ValueError(
            f"Correction selection '{payload.get('selection')}' does not match "
            f"requested selection '{selection}'"
        )

    expected_labels = upsilon_trigger_selection.category_labels(selection)
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
        or metadata.get("selection") != selection
        or tuple(metadata.get("labels", ())) != expected_labels
        or not np.array_equal(category.edges, np.arange(len(expected_labels) + 1))
    ):
        raise ValueError("The triggerCategory axis is inconsistent with the selection")
    expected_edges = {
        "ptll": PT_EDGES,
        "yll": RAPIDITY_EDGES,
        "cosThetaStarll": COSTHETA_EDGES,
        "vars": np.asarray([0.0, 1.0]),
    }
    for axis_name, edges in expected_edges.items():
        if not np.array_equal(correction.axes[axis_name].edges, edges):
            raise ValueError(f"Unexpected binning for payload axis '{axis_name}'")
    if np.any(correction.values(flow=True) < 0.0):
        raise ValueError("The correction contains negative weights")
    usable_values = usable.values(flow=True)
    if not np.all((usable_values == 0.0) | (usable_values == 1.0)):
        raise ValueError("The usable mask contains values other than zero or one")
