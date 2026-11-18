"""Apply production-level MC sample normalizations before histogramming."""

import json
import math
from pathlib import Path


def load_normalizations(path, sample_names):
    """Return effective cross section (pb) / generated sum of weights per sample."""
    sample_names = tuple(sample_names)
    with Path(path).expanduser().open() as stream:
        records = json.load(stream)
    if not isinstance(records, dict) or set(records) != set(sample_names):
        raise ValueError(f"{path}: expected one normalization for each of {sample_names}")

    factors = {}
    for name in sample_names:
        try:
            xsec = float(records[name]["effective_xsec_pb"])
            sumw = float(records[name]["generated_sumw"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{path}: invalid normalization record for {name!r}") from exc
        if not all(math.isfinite(x) and x > 0 for x in (xsec, sumw)):
            raise ValueError(f"{path}: {name!r} needs positive finite cross section and sum of weights")
        factor = xsec / sumw
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError(f"{path}: invalid normalization factor for {name!r}")
        factors[name] = factor
    return factors


def define_normalized_weight(df, files_by_sample, factors, tree_name, generator_weight="genWeight"):
    """Define the per-event weight for a TChain spanning several MC samples."""
    import narf

    if not df.HasColumn(generator_weight):
        raise ValueError(f"MC sample normalization requires the {generator_weight!r} branch")
    narf.clingutils.Declare("#include <stdexcept>\n#include <unordered_map>")
    entries = ",".join(
        "{" + json.dumps(f"{path}/{tree_name}") + f",{factor:.17g}" + "}"
        for name, factor in factors.items() for path in files_by_sample[name]
    )
    df = df.DefinePerSample(
        "sample_normalization",
        "static const std::unordered_map<std::string, double> scales{" + entries + "}; "
        "const auto &key = rdfsampleinfo_.AsString(); "
        "const auto it = scales.find(key); "
        'if (it == scales.end()) throw std::runtime_error("No MC normalization for " + key); '
        "return it->second;",
    )
    return df.Define("weight", f"double({generator_weight}) * sample_normalization")
