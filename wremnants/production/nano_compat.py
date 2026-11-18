"""NanoV15 aliases and type conversions from WMass/WRemnants PR #744."""

# NanoV9 name: NanoV15 name. Only alias columns present in the input schema.
renamed_columns_v15 = {
    "MET_pt": "PFMET_pt",
    "MET_phi": "PFMET_phi",
    "MET_sumEt": "PFMET_sumEt",
    "RawMET_pt": "RawPFMET_pt",
    "RawMET_phi": "RawPFMET_phi",
    "RawMET_sumEt": "RawPFMET_sumEt",
    "MET_fiducialGenPt": "FiducialMET_pt",
    "MET_fiducialGenPhi": "FiducialMET_phi",
    "fixedGridRhoFastjetAll": "Rho_fixedGridRhoFastjetAll",
    "fixedGridRhoFastjetCentral": "Rho_fixedGridRhoFastjetCentral",
    "fixedGridRhoFastjetCentralCalo": "Rho_fixedGridRhoFastjetCentralCalo",
    "fixedGridRhoFastjetCentralChargedPileUp": "Rho_fixedGridRhoFastjetCentralChargedPileUp",
    "fixedGridRhoFastjetCentralNeutral": "Rho_fixedGridRhoFastjetCentralNeutral",
}

retyped_columns_v15 = {
    "nMuon": "unsigned int",
    "nElectron": "unsigned int",
    "PV_npvsGood": "int",
    "Muon_genPartIdx": "ROOT::VecOps::RVec<int>",
    "Muon_jetIdx": "ROOT::VecOps::RVec<int>",
    "Muon_svIdx": "ROOT::VecOps::RVec<int>",
    "Muon_nTrackerLayers": "ROOT::VecOps::RVec<int>",
    "Electron_cutBased": "ROOT::VecOps::RVec<int>",
    "GenPart_genPartIdxMother": "ROOT::VecOps::RVec<int>",
    "GenPart_statusFlags": "ROOT::VecOps::RVec<int>",
    "TrigObj_id": "ROOT::VecOps::RVec<int>",
    "TrigObj_filterBits": "ROOT::VecOps::RVec<int>",
}


def define_nano_compat(df, nano_version="v9"):
    """Expose the available NanoV15 columns under their NanoV9 names and types."""
    if nano_version == "v9":
        return df
    if nano_version != "v15":
        raise ValueError(f"Unknown NanoAOD version {nano_version!r}; expected 'v9' or 'v15'")

    columns = set(df.GetColumnNames())
    for name, name_v15 in renamed_columns_v15.items():
        if name not in columns and name_v15 in columns:
            df = df.Alias(name, name_v15)

    for name, cpp_type in retyped_columns_v15.items():
        if name not in columns:
            continue
        expr = (
            f"{cpp_type}({name}.begin(), {name}.end())"
            if cpp_type.startswith("ROOT::VecOps::RVec")
            else f"static_cast<{cpp_type}>({name})"
        )
        df = df.Redefine(name, expr)

    return df
