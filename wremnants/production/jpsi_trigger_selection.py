"""Trigger-motivated offline selections for J/psi events.

The J/psi channels in the calibration-input histmaker already apply a hard HLT
cut of their own, so the category axis is deliberately trivial here: one
category holding everything the channel selected. It exists so that the
correction payload has the same shape for every resonance.

A finer split was considered and deferred. The J/psi trigger emulation in the
available simulation does not reproduce the data menu (HLT_Dimuon10_Jpsi_Barrel
fires in simulation but never in data, and the Mu7p5_Track* paths are far more
common in simulation than in data), so path-based categories would need their
own study first.
"""

SELECTIONS = ("inclusive",)


def category_labels(selection):
    """Return the ordered category labels produced by *selection*."""

    _validate_selection(selection)
    return ("inclusive",)


def category_expression(selection):
    """Return a C++ expression assigning the selected category index.

    Every event the channel cut kept is accepted, so this never yields the
    ``-1`` reject code that the other resonances use.
    """

    _validate_selection(selection)
    return "0"


def define_trigger_columns(df, columns, selection):
    """Define the trigger and geometry columns used by *selection*.

    The channel's own HLT cut is the trigger requirement for J/psi, so there is
    nothing to add here.
    """

    _validate_selection(selection)
    return df


def cfg_overrides(selection):
    """Return resonance-option overrides required by *selection*.

    J/psi already uses 24 eta bins over +/-2.4, so nothing needs changing.
    """

    _validate_selection(selection)
    return {}


def _validate_selection(selection):
    if selection not in SELECTIONS:
        raise ValueError(
            f"Unknown J/psi trigger selection '{selection}'; choose from {SELECTIONS}"
        )
