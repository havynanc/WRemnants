"""Offline selection categories for D0 -> K pi events.

Unlike the dimuon resonances there is nothing to partition on here: the D*
ALCARECO trees the D0 histmaker reads carry no HLT information, and the sample
has no trigger channels at all. The category axis is therefore deliberately
trivial -- one category holding everything ``data_selection()`` kept -- and
exists only so that the correction payload has the same shape for every
resonance.

The module is named for what it is rather than following the
``*_trigger_selection`` convention of the other two: there is no D0 trigger, and
a file named for one would promise a dependence that cannot exist.
"""

SELECTIONS = ("inclusive",)


def category_labels(selection):
    """Return the ordered category labels produced by *selection*."""

    _validate_selection(selection)
    return ("inclusive",)


def category_expression(selection):
    """Return a C++ expression assigning the selected category index.

    Every event the offline selection kept is accepted, so this never yields the
    ``-1`` reject code that the Upsilon selections use.
    """

    _validate_selection(selection)
    return "0"


def define_trigger_columns(df, columns, selection):
    """Define the trigger and geometry columns used by *selection*.

    There is no trigger requirement to express for D0, so nothing is added.
    """

    _validate_selection(selection)
    return df


def cfg_overrides(selection):
    """Return resonance-option overrides required by *selection*.

    The D0 histmaker has no resonance-option dict to override; the function is
    kept so the module satisfies the same interface as the other selections.
    """

    _validate_selection(selection)
    return {}


def _validate_selection(selection):
    if selection not in SELECTIONS:
        raise ValueError(
            f"Unknown D0 selection '{selection}'; choose from {SELECTIONS}"
        )
