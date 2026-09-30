"""Unit tests for the V-reweighting core.

These cover the pure-Python parts only and need no ROOT or event loop:

    python -m pytest scripts/tests/test_v_reweighting.py -v
"""

import re

import hist
import numpy as np
import pytest

from wremnants.production import v_reweighting as vrw
from wums import ioutils

CASES = [
    ("d0", "inclusive"),
    ("jpsi", "inclusive"),
    ("upsilon", "inclusive_or"),
    ("upsilon", "muon_eta"),
]


def make_inputs(resonance, selection, seed=1234, hole_fraction=0.3):
    """Build a data/simulation input pair with a known pattern of holes."""

    rng = np.random.default_rng(seed)
    axes = vrw.fine_axes(resonance, selection)
    data = hist.Hist(*axes, storage=hist.storage.Weight())
    mc = data.copy()

    shape = data.values().shape
    data_values = rng.uniform(1.0, 50.0, size=shape)
    mc_values = rng.uniform(1.0, 50.0, size=shape)
    # Punch independent holes in each so the common support is a strict subset.
    data_values[rng.random(shape) < hole_fraction] = 0.0
    mc_values[rng.random(shape) < hole_fraction] = 0.0

    data.values()[...] = data_values
    mc.values()[...] = mc_values
    return data, mc


@pytest.mark.parametrize("resonance", sorted(vrw.PT_EDGES))
def test_coarse_edges_are_a_subset_of_the_fine_axis(resonance):
    """_rebin only works if every coarse edge exists on the fine axis."""

    for selection in vrw.selection_module(resonance).SELECTIONS:
        fine = {axis.name: axis.edges for axis in vrw.fine_axes(resonance, selection)}
        for name, coarse in vrw._coarse_edges(resonance).items():
            assert np.all(np.isin(np.round(coarse, 8), fine[name])), name


@pytest.mark.parametrize("resonance,selection", CASES)
def test_diagnostic_axes_match_the_derivation_axes(resonance, selection):
    """The diagnostics are only useful if they share the derivation binning."""

    diagnostic = vrw.fine_kinematic_axes(resonance)
    derivation = vrw.fine_axes(resonance, selection)[1:]
    assert [a.name for a in diagnostic] == [a.name for a in derivation]
    for a, b in zip(diagnostic, derivation):
        assert np.array_equal(a.edges, b.edges)
    assert [a.name for a in diagnostic] == [
        c.removeprefix("vrw_").replace("costheta", "cosThetaStarll")
        for c in vrw.KINEMATIC_COLUMNS
    ]


@pytest.mark.parametrize("resonance,selection", CASES)
def test_only_the_map_variables_are_booked_here(resonance, selection):
    """Single-body diagnostics belong to the histmakers, not to this module.

    v_reweighting books exactly the three variables the maps are derived on. Any
    further diagnostic is the histmaker's business, because what counts as a
    useful single-body ordering depends on the final state.
    """

    axes = vrw.fine_kinematic_axes(resonance)
    assert [a.name for a in axes] == ["ptll", "yll", "cosThetaStarll"]
    assert len(vrw.KINEMATIC_COLUMNS) == len(axes)
    assert not hasattr(vrw, "fine_muon_pt_axes")
    assert not hasattr(vrw, "MUON_PT_COLUMNS")
    assert not hasattr(vrw, "MUON_PT_RANGE")


@pytest.mark.parametrize("resonance,selection", CASES)
def test_diagnostics_share_the_derivation_category_axis(resonance, selection):
    """The leading axis must match so the two can be compared per category."""

    category = vrw.category_axis(resonance, selection)
    derivation = vrw.fine_axes(resonance, selection)[0]
    assert category.name == derivation.name == "triggerCategory"
    assert category.size == derivation.size
    assert category.metadata == derivation.metadata
    assert tuple(category.metadata["labels"]) == vrw.category_labels(
        resonance, selection
    )
    assert not category.traits.underflow and not category.traits.overflow


@pytest.mark.parametrize("resonance", sorted(vrw.PT_EDGES))
def test_category_axis_without_a_selection_is_a_single_catch_all(resonance):
    """--kinDiagnostics works on its own, so the axis must still exist."""

    category = vrw.category_axis(resonance, None)
    assert category.size == 1
    assert tuple(category.metadata["labels"]) == vrw.FALLBACK_CATEGORY_LABELS
    assert category.metadata["selection"] is None


@pytest.mark.parametrize("resonance,selection", CASES)
def test_rebin_preserves_the_total_yield(resonance, selection):
    data, _ = make_inputs(resonance, selection)
    rebinned = vrw._rebin(data, resonance)
    assert np.isclose(rebinned.values().sum(), data.values().sum())
    assert tuple(rebinned.axes.name) == vrw.AXIS_NAMES


@pytest.mark.parametrize("resonance,selection", CASES)
def test_build_payload_preserves_supported_yield(resonance, selection):
    data, mc = make_inputs(resonance, selection)
    payload = vrw.build_payload(data, mc, resonance, "some_channel")

    data_coarse = vrw._rebin(data, resonance).values()
    mc_coarse = vrw._rebin(mc, resonance).values()
    weights = payload["correction"].values()[..., 0]
    usable = payload["usable"].values()[..., 0].astype(bool)

    assert np.array_equal(usable, (data_coarse > 0.0) & (mc_coarse > 0.0))

    for category, entry in enumerate(payload["summary"]):
        mask = usable[category]
        mc_supported = mc_coarse[category][mask].sum()
        # The reweighting is shape-only: it must not move the supported yield.
        assert np.isclose(
            (mc_coarse[category][mask] * weights[category][mask]).sum(), mc_supported
        )
        # Everything outside the common support falls back to unit weight.
        assert np.all(weights[category][~mask] == 1.0)
        # And the weights reproduce the normalized data/simulation ratio.
        data_supported = data_coarse[category][mask].sum()
        expected = (data_coarse[category][mask] / data_supported) / (
            mc_coarse[category][mask] / mc_supported
        )
        assert np.allclose(weights[category][mask], expected)

        assert entry["mc_supported"] == pytest.approx(mc_supported)
        assert entry["data_supported"] == pytest.approx(data_supported)
        assert entry["usable_cells"] == int(mask.sum())
        assert entry["total_cells"] == int(mask.size)
        mc_total = mc_coarse[category].sum()
        assert entry["mc_fallback_fraction"] == pytest.approx(
            (mc_total - mc_supported) / mc_total
        )


@pytest.mark.parametrize("resonance,selection", CASES)
def test_flow_bins_carry_the_fallback(resonance, selection):
    """Out-of-range lookups land in flow, which must mean "unit weight"."""

    data, mc = make_inputs(resonance, selection)
    payload = vrw.build_payload(data, mc, resonance, "some_channel")

    for name, fallback in (("correction", 1.0), ("usable", 0.0)):
        values = payload[name].values(flow=True)
        # triggerCategory and vars are flow-less; ptll, yll and cosThetaStarll
        # each gain an under- and an overflow bin.
        interior = np.zeros(values.shape, dtype=bool)
        interior[:, 1:-1, 1:-1, 1:-1, :] = True
        assert np.all(values[~interior] == fallback)


@pytest.mark.parametrize("resonance,selection", CASES)
def test_payload_round_trips_through_hdf5(resonance, selection, tmp_path):
    import h5py

    data, mc = make_inputs(resonance, selection)
    payload = vrw.build_payload(data, mc, resonance, "some_channel")

    path = tmp_path / "weights.hdf5"
    with h5py.File(path, "w") as h5file:
        ioutils.pickle_dump_h5py(vrw.PAYLOAD_KEY, payload, h5file)

    loaded = vrw.load_payload(path, resonance, selection, "some_channel")
    assert loaded["category_labels"] == payload["category_labels"]
    assert np.array_equal(
        loaded["correction"].values(flow=True),
        payload["correction"].values(flow=True),
    )


@pytest.mark.parametrize(
    "resonance,selection,channel",
    [
        ("upsilon", "inclusive", "some_channel"),  # wrong resonance
        ("jpsi", "muon_eta", "some_channel"),  # wrong selection
        ("jpsi", "inclusive", "other_channel"),  # wrong channel
    ],
)
def test_mismatched_payload_is_rejected(resonance, selection, channel):
    data, mc = make_inputs("jpsi", "inclusive")
    payload = vrw.build_payload(data, mc, "jpsi", "some_channel")
    with pytest.raises(ValueError):
        vrw._validate_payload(payload, resonance, selection, channel)


def test_category_without_common_support_is_rejected():
    data, mc = make_inputs("jpsi", "inclusive")
    mc.values()[...] = 0.0
    with pytest.raises(ValueError, match="no common data/MC support"):
        vrw.build_payload(data, mc, "jpsi", "some_channel")


def test_inputs_from_another_resonance_are_rejected():
    data, mc = make_inputs("upsilon", "inclusive_or")
    with pytest.raises(ValueError, match="produced for resonance"):
        vrw.build_payload(data, mc, "jpsi", "some_channel")


def test_every_resonance_is_registered_in_all_four_dicts():
    """A resonance missing from one registry must not reach a histmaker job.

    Three of the four dicts are exercised by the parametrised tests above and
    fail at import, but SELECTION_MODULES is only touched at run time, deep
    inside a histmaker. This is the tripwire for adding a resonance and
    forgetting one of its entries.
    """

    expected = set(vrw.PT_EDGES)
    assert set(vrw.FINE_PT_STEP) == expected
    assert set(vrw.SELECTION_MODULES) == expected
    assert set(vrw.resonances()) == expected


@pytest.mark.parametrize("resonance", sorted(vrw.PT_EDGES))
def test_selection_module_satisfies_the_interface(resonance):
    """Every selection module must expose what the histmakers call on it."""

    module = vrw.selection_module(resonance)
    assert isinstance(module.SELECTIONS, tuple) and module.SELECTIONS
    for selection in module.SELECTIONS:
        labels = module.category_labels(selection)
        assert isinstance(labels, tuple) and labels
        assert isinstance(module.category_expression(selection), str)
        assert module.cfg_overrides(selection) is not None
        # Not asserted to be a pass-through: the Upsilon module genuinely
        # defines trigger and geometry columns here. Only J/psi and D0, which
        # have nothing to add, return the frame untouched.
        assert callable(module.define_trigger_columns)
    if resonance in ("d0", "jpsi"):
        sentinel = object()
        assert (
            module.define_trigger_columns(sentinel, {}, module.SELECTIONS[0])
            is sentinel
        )
    with pytest.raises(ValueError):
        module.category_labels("not_a_selection")


def test_d0_has_exactly_one_never_rejecting_category():
    """D0 has no trigger, so the category axis is a single catch-all bin."""

    assert vrw.selection_module("d0").SELECTIONS == ("inclusive",)
    assert vrw.category_labels("d0", "inclusive") == ("inclusive",)
    # Never returns the -1 reject code the Upsilon selections use, so no event
    # the offline selection kept is silently dropped by the category filter.
    assert vrw.selection_module("d0").category_expression("inclusive") == "0"
    assert vrw.category_axis("d0", "inclusive").size == 1


def test_d0_pt_axis_covers_the_selected_spectrum():
    """The D0 pT axis must span the *selected* spectrum, not the inclusive one.

    The Dst_pt > 5 GeV requirement makes the selected D0 spectrum far harder
    than the raw one: data sits at a median of ~12 GeV with p99 ~67 GeV. An axis
    stopping at 20 GeV, as a raw-spectrum measurement would suggest, leaves ~21%
    of data in overflow at unit weight.
    """

    edges = vrw.PT_EDGES["d0"]
    assert edges[0] == 0.0
    assert edges[-1] >= 100.0


def test_define_kinematics_still_defaults_to_muons():
    """The dimuon call site passes no masses, so the default must be muons."""

    import inspect

    parameters = inspect.signature(vrw.define_kinematics).parameters
    assert parameters["masses"].default == (vrw.MUON_MASS, vrw.MUON_MASS)


# Column names a selection module may rely on without defining them itself: the
# reweighting core defines these before any selection module is called.
CORE_VRW_COLUMNS = frozenset(vrw.KINEMATIC_COLUMNS) | {vrw.CATEGORY_COLUMN}

# Deliberately free of the "vrw_" prefix so a reference to one of these is never
# mistaken for a reweighting column.
FAKE_RECO_COLUMNS = {
    key: f"COL_{key}"
    for key in (
        "plus_pt", "plus_eta", "plus_phi",
        "minus_pt", "minus_eta", "minus_phi",
        "mass",
    )
}

_VRW_REFERENCE = re.compile(r"\bvrw_\w+\b")


class _RecordingFrame:
    """Stand-in for an RDataFrame that records Define calls without ROOT.

    Selection modules only ever chain Define and Filter, so this is enough to
    learn which columns a module creates and what each one reads.
    """

    def __init__(self):
        self.defines = []

    def Define(self, name, expression):
        self.defines.append((name, str(expression)))
        return self

    def Filter(self, expression):
        return self


@pytest.mark.parametrize("resonance", sorted(vrw.PT_EDGES))
def test_selection_only_references_columns_that_exist(resonance):
    """Every vrw_ column a selection reads must be one it or the core defines.

    This is the regression test for a selection cutting on a column that the
    reweighting core used to provide and no longer does. That failure is
    invisible to every other test here, because it only surfaces as a C++
    just-in-time compilation error once a real event loop is built.
    """

    module = vrw.selection_module(resonance)
    for selection in module.SELECTIONS:
        frame = _RecordingFrame()
        module.define_trigger_columns(frame, FAKE_RECO_COLUMNS, selection)

        available = set(CORE_VRW_COLUMNS)
        for name, expression in frame.defines:
            for reference in _VRW_REFERENCE.findall(expression):
                assert reference in available, (
                    f"{resonance}/{selection}: column '{name}' reads "
                    f"'{reference}', which nothing has defined yet"
                )
            available.add(name)

        for reference in _VRW_REFERENCE.findall(
            module.category_expression(selection)
        ):
            assert reference in available, (
                f"{resonance}/{selection}: the category expression reads "
                f"'{reference}', which neither the reweighting core nor "
                f"define_trigger_columns provides"
            )
