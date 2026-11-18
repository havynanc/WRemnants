"""Input inventory and charge-ordered columns for Upsilon NanoV15."""

import json
from pathlib import Path
from urllib.parse import urlsplit

from wremnants.production import nano_compat


MC_SLICES = ("pt0to8", "pt8to13", "pt13toInf")
INPUT_GROUPS = ("data", *MC_SLICES)


def load_inputs(path):
    """Read each input group from a JSON list of ROOT files or local directories.

    Local paths may be relative to the JSON file. A directory is searched
    recursively; point it at the desired production tag. Different CRAB tasks
    may have the same output basename, so use full paths for identity.
    """
    path = Path(path).expanduser().resolve()
    with path.open() as stream:
        groups = json.load(stream)
    if not isinstance(groups, dict) or set(groups) != set(INPUT_GROUPS):
        raise ValueError(f"{path}: expected exactly the input groups {INPUT_GROUPS}")

    result = {}
    seen = set()
    for group in INPUT_GROUPS:
        entries = groups[group]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{path}: {group} must be a nonempty list of paths")
        result[group] = []
        for entry in entries:
            if not isinstance(entry, str) or not entry:
                raise ValueError(f"{path}: invalid {group} input path {entry!r}")
            uri = urlsplit(entry)
            if uri.scheme:
                if not uri.path.endswith(".root") or not uri.netloc:
                    raise ValueError(f"{path}: remote input must be a ROOT file URL: {entry}")
                files = [entry]
            else:
                local = (path.parent / Path(entry).expanduser()).resolve()
                if local.is_dir():
                    files = [str(f.resolve()) for f in sorted(local.rglob("*.root")) if f.is_file()]
                else:
                    files = [str(local)] if local.is_file() and local.suffix == ".root" else []
                if not files:
                    raise ValueError(f"{path}: no ROOT files found at {local}")
            for filename in files:
                if filename in seen:
                    raise ValueError(f"{path}: duplicate input path {filename}")
                seen.add(filename)
                result[group].append(filename)
    return result


def _dimuon_mass(df, name, plus, minus, valid=None):
    muon_mass = 0.1056583755
    expr = (
        f"(ROOT::Math::PtEtaPhiMVector({plus}_pt, {plus}_eta, {plus}_phi, {muon_mass}) + "
        f"ROOT::Math::PtEtaPhiMVector({minus}_pt, {minus}_eta, {minus}_phi, {muon_mass})).M()"
    )
    return df.Define(name, expr if valid is None else f"{valid} ? {expr} : -1.")


def define_columns(df, is_data):
    """Expose the legacy histmaker's scalar columns from the two stored refits."""
    df = nano_compat.define_nano_compat(df, "v15")
    branch = "cvh" if is_data else "cvhideal"
    charge, pt, eta, phi, pixel = (
        f"Muon_{branch}{field}" for field in ("Charge", "Pt", "Eta", "Phi", "NValidPixelHits")
    )

    df = df.Filter(
        " && ".join(f"{col}.size() == 2" for col in (charge, pt, eta, phi, pixel)),
        "NanoV15 exactly two refitted muons",
    ).Filter(
        f"{pt}[0] > 0.f && {pt}[1] > 0.f && "
        f"(({charge}[0] == 1 && {charge}[1] == -1) || "
        f"({charge}[0] == -1 && {charge}[1] == 1))",
        "NanoV15 valid opposite-charge refits",
    )
    df = df.Define("upsilon_plus_idx", f"int({charge}[0] == 1 ? 0 : 1)")
    df = df.Define("upsilon_minus_idx", "1 - upsilon_plus_idx")
    for sign in ("plus", "minus"):
        idx = f"upsilon_{sign}_idx"
        for field, column in (("pt", pt), ("eta", eta), ("phi", phi)):
            df = df.Define(f"Mu{sign}cor_{field}", f"{column}[{idx}]")
        df = df.Define(f"Mu{sign}_nvalidpixel", f"int({pixel}[{idx}])")
    df = _dimuon_mass(df, "Jpsicor_mass", "Mupluscor", "Muminuscor")
    if is_data:
        return df

    for sign in ("plus", "minus"):
        df = df.Define(
            f"upsilon_{sign}_gen_idx",
            f"Muon_genPartIdx.size() == 2 ? int(Muon_genPartIdx[upsilon_{sign}_idx]) : -1",
        )
    df = df.Define(
        "upsilon_valid_gen_match",
        "GenPart_pt.size() == GenPart_eta.size() && GenPart_pt.size() == GenPart_phi.size() && "
        "upsilon_plus_gen_idx >= 0 && upsilon_minus_gen_idx >= 0 && "
        "upsilon_plus_gen_idx < int(GenPart_pt.size()) && "
        "upsilon_minus_gen_idx < int(GenPart_pt.size())",
    )
    for sign in ("plus", "minus"):
        for field, fallback in (("pt", "-1.f"), ("eta", "0.f"), ("phi", "0.f")):
            df = df.Define(
                f"Mu{sign}gen_{field}",
                f"upsilon_valid_gen_match ? GenPart_{field}[upsilon_{sign}_gen_idx] : {fallback}",
            )
    return _dimuon_mass(df, "Jpsigen_mass", "Muplusgen", "Muminusgen", "upsilon_valid_gen_match")


def define_ancestry(df):
    """Label MC by the nearest Upsilon ancestor shared by both matched muons."""
    import narf

    narf.clingutils.Declare('#include "upsilon_nanov15.hpp"')
    return (
        df.Define(
            "upsilon_pdg_assignment",
            "wrem::commonUpsilonAncestor(upsilon_plus_gen_idx, upsilon_minus_gen_idx, "
            "GenPart_genPartIdxMother, GenPart_pdgId)",
        )
        .Define("upsilon_pdg_state", "upsilon_pdg_assignment[0]")
        .Define("upsilon_pdg_failure", "upsilon_pdg_assignment[1]")
    )
