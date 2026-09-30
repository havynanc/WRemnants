#!/usr/bin/env python3

"""Build category-specific kinematic V weights from a histmaker output.

The inputs are produced by ``dimuon_resonances_calinput.py --makeVReweightInputs``
for one resonance, selection and trigger channel. One correction file covers
exactly that combination; the loader in the histmaker refuses to apply it to
anything else.
"""

import argparse
import json
from pathlib import Path

import h5py

from wremnants.production import v_reweighting
from wremnants.utilities import parsing
from wremnants.utilities.io_tools import base_io
from wums import ioutils, logging

logger = logging.child_logger(__name__)


# Not every histmaker names its datasets after the resonance and channel. The D0
# histmaker's names are consumed verbatim by scripts/rabbit/d0_tensor.py, so they
# are recorded here rather than renamed at the source.
DATASET_NAME_OVERRIDES = {
    "d0": {"data": "D0_data", "simulation": "D0_mc"},
}


def dataset_names(resonance, channel):
    """Return the dataset names the histmaker writes for *channel*."""

    if resonance in DATASET_NAME_OVERRIDES:
        return dict(DATASET_NAME_OVERRIDES[resonance])
    return {
        "data": f"{resonance}_data_{channel}",
        "simulation": f"{resonance}_mc_{channel}",
    }


def load_inputs(path, resonance, channel):
    """Return the data and simulation input histograms for one channel."""

    expected = dataset_names(resonance, channel)
    selected = {}
    available = []
    with h5py.File(path, "r") as h5file:
        results = base_io.load_results_h5py(h5file)
        for key, result in results.items():
            if not isinstance(result, dict) or "dataset" not in result:
                continue
            output = result.get("output", {})
            if v_reweighting.INPUT_HIST_NAME not in output:
                continue
            name = result["dataset"].get("name", key)
            available.append(name)
            kind = "data" if result["dataset"].get("is_data", False) else "simulation"
            if name != expected[kind]:
                continue
            if kind in selected:
                raise ValueError(
                    f"Found more than one {kind} input histogram for dataset "
                    f"'{name}'"
                )
            value = output[v_reweighting.INPUT_HIST_NAME]
            selected[kind] = value.get() if hasattr(value, "get") else value

    missing = sorted(set(expected) - set(selected))
    if missing:
        raise ValueError(
            f"No {' and '.join(missing)} '{v_reweighting.INPUT_HIST_NAME}' histogram "
            f"for resonance '{resonance}' and channel '{channel}' in '{path}'. "
            f"Expected dataset(s) {[expected[k] for k in missing]}; the file "
            f"contains {sorted(available)}"
        )
    return selected["data"], selected["simulation"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parsing.base_parser()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input histmaker HDF5 file")
    parser.add_argument("output", type=Path, help="Output correction HDF5 file")
    parser.add_argument(
        "--resonance",
        required=True,
        choices=v_reweighting.resonances(),
        help="Resonance the inputs were produced for",
    )
    parser.add_argument(
        "--selection",
        required=True,
        help="Trigger selection the inputs were produced with",
    )
    parser.add_argument(
        "--channel",
        required=True,
        help="Trigger channel label the inputs were produced with",
    )
    args = parser.parse_args()

    global logger
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    module = v_reweighting.selection_module(args.resonance)
    if args.selection not in module.SELECTIONS:
        raise ValueError(
            f"Unknown selection '{args.selection}' for resonance "
            f"'{args.resonance}'; choose from {module.SELECTIONS}"
        )
    # The D0 dataset names do not encode the channel, so a typo here would not be
    # caught by the name lookup below and would only surface as a confusing
    # mismatch when the payload is applied.
    if args.resonance == "d0" and args.channel != "inclusive":
        raise ValueError(
            "The D0 sample has a single channel; pass --channel inclusive"
        )

    data_hist, mc_hist = load_inputs(args.input, args.resonance, args.channel)
    payload = v_reweighting.build_payload(
        data_hist, mc_hist, args.resonance, args.channel
    )
    payload["meta_info"] = {
        "input_file": str(args.input),
        "input_meta": base_io.get_metadata(str(args.input)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as h5file:
        ioutils.pickle_dump_h5py(v_reweighting.PAYLOAD_KEY, payload, h5file)
    logger.info(f"Wrote {args.output}")
    logger.info(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
