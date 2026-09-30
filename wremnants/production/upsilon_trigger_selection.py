"""Trigger-motivated offline selections for Upsilon events.

The selections in this module define mutually exclusive event categories that
mirror the kinematic and geometric requirements of the low- and high-pT
Upsilon triggers. They are deliberately independent of histogram booking and
weight construction so the same definitions can be reused in both steps.

Which category an event lands in is decided by offline geometry alone; the HLT
path is then a requirement attached to the branch the geometry picked. The two
paths overlap heavily, so a split on path membership would not partition the
sample, and the trigger emulation in simulation is not reliable enough to drive
the assignment.
"""

import hist

SELECTIONS = (
    "inclusive_or",
    "muon_eta",
    "dimuon_rapidity",
    "combined_geometry",
)

MUON_ETA_MAX = 2.4
MASS_MIN = 8.8
MASS_MAX = 10.8

DIMUON_PT_MIN = 8.5
HIGH_DIMUON_PT_MIN = 13.5
HIGH_LEADING_PT_MIN = 13.5
HIGH_SUBLEADING_PT_MIN = 6.5

BARREL_MUON_ETA_MAX = 1.4
BARREL_DIMUON_RAPIDITY_MAX = 1.25
BARREL_DELTA_ETA_MAX = 1.8


def category_labels(selection):
    """Return the ordered category labels produced by *selection*."""

    _validate_selection(selection)
    return ("inclusive",) if selection == "inclusive_or" else ("barrel", "high")


def category_expression(selection):
    """Return a C++ expression assigning the selected category index.

    The expression returns ``-1`` for rejected events. Category indices follow
    :func:`category_labels` and are therefore stable in histogram axes and
    correction payloads.
    """

    _validate_selection(selection)
    trigger_or = "(vrw_trigger_dimuon8 || vrw_trigger_dimuon13)"
    if selection == "inclusive_or":
        return f"({trigger_or} && vrw_ptll > {DIMUON_PT_MIN}) ? 0 : -1"

    route = {
        "muon_eta": "vrw_barrel_muon_eta",
        "dimuon_rapidity": "vrw_barrel_dimuon_rapidity",
        "combined_geometry": "vrw_barrel_geometry",
    }[selection]
    barrel = (
        f"({route} && vrw_trigger_dimuon8 && vrw_barrel_geometry "
        f"&& vrw_ptll > {DIMUON_PT_MIN})"
    )
    high = (
        f"(!{route} && vrw_trigger_dimuon13 "
        f"&& vrw_ptll > {HIGH_DIMUON_PT_MIN} "
        f"&& vrw_ptlead > {HIGH_LEADING_PT_MIN} "
        f"&& vrw_ptsublead > {HIGH_SUBLEADING_PT_MIN})"
    )
    return f"{barrel} ? 0 : ({high} ? 1 : -1)"


def define_trigger_columns(df, columns, selection):
    """Define the trigger and geometry columns used by *selection*.

    The dimuon kinematics are defined separately by the reweighting core; the
    HLT decisions, the barrel geometry and the per-muon pT thresholds are
    resonance specific.

    The leading/subleading muon pT are defined here because the ``high``
    category cuts on them. They are deliberately not part of the reweighting
    core, which defines only the three variables the maps are derived on.
    """

    _validate_selection(selection)
    return (
        df.Define(
            "vrw_ptlead",
            f"std::max(double({columns['plus_pt']}), "
            f"double({columns['minus_pt']}))",
        )
        .Define(
            "vrw_ptsublead",
            f"std::min(double({columns['plus_pt']}), "
            f"double({columns['minus_pt']}))",
        )
        .Define(
            "vrw_trigger_dimuon8",
            "bool(HLT_Dimuon8_Upsilon_Barrel != 0)",
        )
        .Define("vrw_trigger_dimuon13", "bool(HLT_Dimuon13_Upsilon != 0)")
        .Define(
            "vrw_barrel_muon_eta",
            f"bool(std::fabs({columns['plus_eta']}) < {BARREL_MUON_ETA_MAX} && "
            f"std::fabs({columns['minus_eta']}) < {BARREL_MUON_ETA_MAX})",
        )
        .Define(
            "vrw_barrel_dimuon_rapidity",
            f"bool(std::fabs(vrw_yll) < {BARREL_DIMUON_RAPIDITY_MAX})",
        )
        .Define(
            "vrw_barrel_delta_eta",
            f"bool(std::fabs(double({columns['plus_eta']}) - "
            f"double({columns['minus_eta']})) < {BARREL_DELTA_ETA_MAX})",
        )
        .Define(
            "vrw_barrel_geometry",
            "bool(vrw_barrel_muon_eta && "
            "vrw_barrel_dimuon_rapidity && vrw_barrel_delta_eta)",
        )
    )


def cfg_overrides(selection):
    """Return resonance-option overrides required by *selection*.

    The trigger categories reach beyond the default Y(1S) barrel configuration,
    so the eta acceptance and mass window are widened to cover the full
    Y(1S)/Y(2S)/Y(3S) system.
    """

    _validate_selection(selection)
    return {
        "default_eta_bins": 24,
        "eta_range": (-MUON_ETA_MAX, MUON_ETA_MAX),
        "mass_axis": hist.axis.Regular(80, MASS_MIN, MASS_MAX, name="mass"),
        "mass_range": (MASS_MIN, MASS_MAX),
    }


def _validate_selection(selection):
    if selection not in SELECTIONS:
        raise ValueError(
            f"Unknown Upsilon trigger selection '{selection}'; choose from {SELECTIONS}"
        )
