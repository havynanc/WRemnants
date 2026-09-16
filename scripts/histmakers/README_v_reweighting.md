# Trigger selections and kinematic V reweighting

The calibration-input histmaker can apply trigger-motivated offline selections
and direct, category-specific shape weights in `(ptll, yll, cosThetaStarll)`.
The feature is opt-in and works for both J/psi and Upsilon; the existing default
selection and weights are unchanged.

## Workflow

Each correction file covers exactly one resonance, selection and trigger
channel, so both passes take a single `--triggers` label.

First produce the unweighted data and simulation inputs:

```bash
python3 scripts/histmakers/dimuon_resonances_calinput.py \
  --resonance jpsi --era 2016PostVFP --triggers dimuon20_jpsi \
  --vReweightSelection inclusive \
  --makeVReweightInputs -j 16 -o OUTPUT
```

Build the correction file from the resulting HDF5 file:

```bash
python3 scripts/corrections/make_v_reweighting.py \
  OUTPUT/dimuon_resonances_calinput_jpsi_2016PostVFP_vrwSel_inclusive_vrwChan_dimuon20_jpsi.hdf5 \
  jpsi_v_weights_dimuon20_jpsi.hdf5 \
  --resonance jpsi --selection inclusive --channel dimuon20_jpsi
```

Then rerun the histmaker with the same resonance, selection and channel:

```bash
python3 scripts/histmakers/dimuon_resonances_calinput.py \
  --resonance jpsi --era 2016PostVFP --triggers dimuon20_jpsi \
  --vReweightSelection inclusive \
  --vReweightFile jpsi_v_weights_dimuon20_jpsi.hdf5 \
  -j 16 -o OUTPUT
```

The Upsilon workflow is identical with `--resonance upsilon --triggers inclusive`
and one of the Upsilon selections, for example `--vReweightSelection muon_eta`.

The loader rejects a file whose recorded resonance, selection, channel, category
labels, axes or binning do not match the requested run, so the two passes cannot
silently disagree. The selection and channel are part of the output filename, so
the weighted and unweighted passes do not collide. Data are never weighted. In
simulation, the same kinematic weight multiplies the nominal and systematic
histograms.

## Selections and categories

Which category an event lands in is decided by offline geometry, not by which
HLT path fired. The paths overlap heavily, so path membership would not
partition the sample, and the trigger emulation in simulation does not reproduce
the data menu closely enough to drive the assignment.

**J/psi** has a single trivial `inclusive` category. The J/psi channels already
apply a hard HLT cut of their own, so the category axis exists only to give the
payload the same shape for every resonance. A finer split is deferred: in the
available simulation `HLT_Dimuon10_Jpsi_Barrel` fires but never fires in data,
and the `Mu7p5_Track*` paths are far more common in simulation than in data, so
path-based categories need their own study first.

**Upsilon** offers four selections. `inclusive_or` retains one category
requiring the Dimuon8-barrel or Dimuon13 HLT path and reconstructed dimuon pT
above 8.5 GeV. The other choices make two exclusive categories. Their barrel
route is selected using both-muon eta (`muon_eta`), dimuon rapidity
(`dimuon_rapidity`), or their combination with the dimuon delta-eta requirement
(`combined_geometry`). Barrel events must pass the Dimuon8-barrel path, its full
offline geometry, and dimuon pT above 8.5 GeV. Remaining events must pass
Dimuon13 and have dimuon, leading-muon, and subleading-muon pT above 13.5, 13.5,
and 6.5 GeV, respectively. Under `muon_eta` and `dimuon_rapidity` an event that
satisfies the looser route but fails the full barrel geometry is rejected rather
than reassigned; only `combined_geometry` partitions losslessly.

## Binning

The rapidity and cos(theta*) binnings are shared. The pT binning is per
resonance, since the spectra differ: J/psi covers 0-60 GeV in 82 bins, Upsilon
8.5-100 GeV in 61 bins. The coarse map edges must be a subset of the fine
derivation axis; this is checked at import time in `v_reweighting.py`.

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
`vReweightCoverage`, split by category and usable/fallback status.

A category whose offline thresholds sit above the lower edge of the shared pT
axis leaves the bins below that threshold unreachable, so they remain at unit
weight. This applies to the Upsilon `high` category, whose 13.5 GeV threshold is
above the 8.5 GeV axis edge.

## A note on normalization

The weights preserve the supported simulation yield exactly, but they do so in
the map's own `(ptll, yll, cosThetaStarll)` space. The nominal calibration
histogram lives in `(eta1, eta2, pt1, pt2, mass)` and its muon pT axes stop
short of the selection, so roughly 10-17% of the selected simulation falls in
its flow bins. The weights are not uniform across that boundary, so the in-axis
integral of the nominal histogram does move a little: measured at +3.2% for
Upsilon `combined_geometry` and -0.2% for the J/psi `dimuon20_jpsi` channel.
Summed including flow the integral is unchanged to seven digits. This is an
acceptance effect, not a normalization bug.
