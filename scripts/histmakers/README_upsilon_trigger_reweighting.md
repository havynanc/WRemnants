# Upsilon trigger selections and kinematic reweighting

The Upsilon calibration-input histmaker can apply trigger-motivated offline
selections and direct, category-specific shape weights in
`(ptll, yll, cosThetaStarll)`. The feature is opt-in; the existing default
selection and weights are unchanged.

## Workflow

First produce the unweighted data and simulation inputs for one selection:

```bash
python3 scripts/histmakers/dimuon_resonances_calinput.py \
  --resonance upsilon --era 2016PostVFP --triggers inclusive \
  --upsilonTriggerSelection muon_eta \
  --makeUpsilonVReweightInputs -j 16 -o OUTPUT_UNWEIGHTED
```

Build one correction file from the resulting HDF5 file:

```bash
python3 scripts/corrections/make_upsilon_v_reweighting.py \
  OUTPUT_UNWEIGHTED/dimuon_resonances_calinput_upsilon_2016PostVFP.hdf5 \
  upsilon_v_weights_muon_eta.hdf5
```

Then rerun the histmaker with the same selection and apply that file:

```bash
python3 scripts/histmakers/dimuon_resonances_calinput.py \
  --resonance upsilon --era 2016PostVFP --triggers inclusive \
  --upsilonTriggerSelection muon_eta \
  --upsilonVReweightFile upsilon_v_weights_muon_eta.hdf5 \
  -j 16 -o OUTPUT_WEIGHTED
```

Use a separate correction file for each selection scheme. The loader rejects
a file if its recorded selection, category labels, axes, or binning do not
match the requested selection. Data are never weighted. In simulation, the
same kinematic weight multiplies the nominal and systematic histograms.

## Selections and categories

`inclusive_or` retains one category requiring the Dimuon8-barrel or Dimuon13
HLT path and reconstructed dimuon pT above 8.5 GeV. The other choices make two
exclusive categories. Their barrel route is selected using both-muon eta
(`muon_eta`), dimuon rapidity (`dimuon_rapidity`), or their combination with
the dimuon delta-eta requirement (`combined_geometry`). Barrel events must
pass the Dimuon8-barrel path, its full offline geometry, and dimuon pT above
8.5 GeV. Remaining events must pass Dimuon13 and have dimuon, leading-muon,
and subleading-muon pT above 13.5, 13.5, and 6.5 GeV, respectively.

## Weight definition and fallback

Weights are derived independently in every retained category as

```text
(data bin / supported data yield) / (MC bin / supported MC yield).
```

Only bins with positive data and simulation content enter this calculation.
No background subtraction, clipping, smoothing, interpolation, or
regularization is applied. Bins without common support, including flow bins,
receive unit weight. The payload records the fallback fraction and weight
summary for every category; weighted histmaker output also contains
`upsilonVReweightCoverage`, split by category and usable/fallback status. A
common pT axis starts at 8.5 GeV, so bins below the 13.5 GeV high-category
threshold are unreachable for that category and remain at unit weight.
