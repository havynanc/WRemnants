#!/usr/bin/env python3

"""Build category-specific Upsilon V weights from a histmaker output."""

import argparse
import json
from pathlib import Path

import h5py

from wremnants.production import upsilon_v_reweighting
from wremnants.utilities.io_tools import base_io
from wums import ioutils


def load_inputs(path):
    selected = {}
    with h5py.File(path, "r") as h5file:
        results = base_io.load_results_h5py(h5file)
        for result in results.values():
            if not isinstance(result, dict) or "dataset" not in result:
                continue
            output = result.get("output", {})
            if "upsilonTriggerVReweightInput" not in output:
                continue
            kind = "data" if result["dataset"].get("is_data", False) else "simulation"
            if kind in selected:
                raise ValueError(f"Found more than one {kind} input histogram")
            value = output["upsilonTriggerVReweightInput"]
            selected[kind] = value.get() if hasattr(value, "get") else value

    if set(selected) != {"data", "simulation"}:
        raise ValueError(
            "The input must contain exactly one data and one simulation "
            "upsilonTriggerVReweightInput histogram"
        )
    return selected["data"], selected["simulation"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input histmaker HDF5 file")
    parser.add_argument("output", type=Path, help="Output correction HDF5 file")
    args = parser.parse_args()

    data_hist, mc_hist = load_inputs(args.input)
    payload = upsilon_v_reweighting.build_payload(data_hist, mc_hist)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as h5file:
        ioutils.pickle_dump_h5py("upsilon_v_reweighting", payload, h5file)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
