"""Trigger-motivated offline selections for Upsilon events.

The selections in this module define mutually exclusive event categories that
mirror the kinematic and geometric requirements of the low- and high-pT
Upsilon triggers. They are deliberately independent of histogram booking and
weight construction so the same definitions can be reused in both steps.
"""

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
    trigger_or = "(upsilon_trigger_dimuon8 || upsilon_trigger_dimuon13)"
    if selection == "inclusive_or":
        return f"({trigger_or} && upsilon_ptll > {DIMUON_PT_MIN}) ? 0 : -1"

    route = {
        "muon_eta": "upsilon_barrel_muon_eta",
        "dimuon_rapidity": "upsilon_barrel_dimuon_rapidity",
        "combined_geometry": "upsilon_barrel_geometry",
    }[selection]
    barrel = (
        f"({route} && upsilon_trigger_dimuon8 && upsilon_barrel_geometry "
        f"&& upsilon_ptll > {DIMUON_PT_MIN})"
    )
    high = (
        f"(!{route} && upsilon_trigger_dimuon13 "
        f"&& upsilon_ptll > {HIGH_DIMUON_PT_MIN} "
        f"&& upsilon_ptlead > {HIGH_LEADING_PT_MIN} "
        f"&& upsilon_ptsublead > {HIGH_SUBLEADING_PT_MIN})"
    )
    return f"{barrel} ? 0 : ({high} ? 1 : -1)"


def define_columns(df, columns, selection):
    """Define the reconstructed kinematics and selected category on *df*."""

    _validate_selection(selection)
    muon_mass = 0.1056583755
    return (
        df.Define(
            "upsilon_muplus_mom4",
            "ROOT::Math::PtEtaPhiMVector("
            f"{columns['plus_pt']}, {columns['plus_eta']}, "
            f"{columns['plus_phi']}, {muon_mass})",
        )
        .Define(
            "upsilon_muminus_mom4",
            "ROOT::Math::PtEtaPhiMVector("
            f"{columns['minus_pt']}, {columns['minus_eta']}, "
            f"{columns['minus_phi']}, {muon_mass})",
        )
        .Define("upsilon_dimuon_mom4", "upsilon_muplus_mom4 + upsilon_muminus_mom4")
        .Define(
            "upsilon_ptlead",
            f"std::max(double({columns['plus_pt']}), "
            f"double({columns['minus_pt']}))",
        )
        .Define(
            "upsilon_ptsublead",
            f"std::min(double({columns['plus_pt']}), "
            f"double({columns['minus_pt']}))",
        )
        .Define("upsilon_ptll", "upsilon_dimuon_mom4.pt()")
        .Define("upsilon_yll", "upsilon_dimuon_mom4.Rapidity()")
        .Define(
            "upsilon_cs",
            "wrem::csSineCosThetaPhi(upsilon_muplus_mom4, upsilon_muminus_mom4)",
        )
        .Define("upsilon_costheta", "upsilon_cs.costheta")
        .Define(
            "upsilon_trigger_dimuon8",
            "bool(HLT_Dimuon8_Upsilon_Barrel != 0)",
        )
        .Define("upsilon_trigger_dimuon13", "bool(HLT_Dimuon13_Upsilon != 0)")
        .Define(
            "upsilon_barrel_muon_eta",
            f"bool(std::fabs({columns['plus_eta']}) < {BARREL_MUON_ETA_MAX} && "
            f"std::fabs({columns['minus_eta']}) < {BARREL_MUON_ETA_MAX})",
        )
        .Define(
            "upsilon_barrel_dimuon_rapidity",
            f"bool(std::fabs(upsilon_yll) < {BARREL_DIMUON_RAPIDITY_MAX})",
        )
        .Define(
            "upsilon_barrel_delta_eta",
            f"bool(std::fabs(double({columns['plus_eta']}) - "
            f"double({columns['minus_eta']})) < {BARREL_DELTA_ETA_MAX})",
        )
        .Define(
            "upsilon_barrel_geometry",
            "bool(upsilon_barrel_muon_eta && "
            "upsilon_barrel_dimuon_rapidity && upsilon_barrel_delta_eta)",
        )
        .Define("upsilon_trigger_category", category_expression(selection))
    )


def _validate_selection(selection):
    if selection not in SELECTIONS:
        raise ValueError(
            f"Unknown Upsilon trigger selection '{selection}'; choose from {SELECTIONS}"
        )
