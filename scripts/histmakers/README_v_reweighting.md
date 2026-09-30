# Trigger selections and kinematic V reweighting

The calibration-input histmakers can apply trigger-motivated offline selections
and direct, category-specific shape weights in `(ptll, yll, cosThetaStarll)`.
The feature is opt-in and covers three resonances: J/psi and Upsilon in
`dimuon_resonances_calinput.py`, and D0 in `d0_mass.py`. The existing default
selection and weights are unchanged.

For D0 the reweighted system is the `K pi` pair rather than a dimuon, so the
axis names read as dimuon-flavoured but mean the parent pT, the parent rapidity,
and the Collins-Soper polar angle built from the two daughters with the kaon in
the first slot. The axis names are shared deliberately: the payload then has the
same shape for every resonance, and no consumer has to branch.

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

### D0

D0 lives in a different histmaker and takes no `--triggers`; the channel is
pinned to `inclusive` on both sides.

```bash
python3 scripts/histmakers/d0_mass.py --era 2016PostVFP --mcSelection dataLike \
  --vReweightSelection inclusive --makeVReweightInputs --kinDiagnostics \
  -j 16 -o OUTPUT

python3 scripts/corrections/make_v_reweighting.py \
  OUTPUT/d0_mass_2016PostVFP_vrwSel_inclusive_vrwChan_inclusive_vrwInputs_kinDiag.hdf5 \
  d0_v_weights.hdf5 \
  --resonance d0 --selection inclusive --channel inclusive

python3 scripts/histmakers/d0_mass.py --era 2016PostVFP --mcSelection dataLike \
  --vReweightSelection inclusive --vReweightFile d0_v_weights.hdf5 \
  --kinDiagnostics -j 16 -o OUTPUT
```

**Use `--mcSelection dataLike` in both passes.** The map is a data/MC shape
ratio, so the two sides have to occupy the same phase space, and the default
`truthMatched` selection is much looser than the data one: it drops the D0 mass
window, the deltaM window, the `Dst_pt > 5 GeV` cut and the `pis_dR_D0` cut.
Deriving across that gap folds the selection difference into what is nominally a
production weight. The histmaker warns but does not refuse.

Unlike the dimuon histmaker, `d0_mass.py` names its datasets `D0_data` and
`D0_mc` rather than after the resonance and channel, because those names are
consumed verbatim by `scripts/rabbit/d0_tensor.py`. `make_v_reweighting.py`
carries a small override table for this.

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

**D0** has a single trivial `inclusive` category and, unlike the other two, no
trigger at all: the D* ALCARECO trees carry no HLT information and the histmaker
has no `--triggers` argument. The category axis exists purely so the payload
shape is resonance independent.

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
8.5-100 GeV in 61 bins, D0 0-100 GeV in 22 bins.

The D0 binning is deliberately coarse, for two reasons. The shared rapidity and
cos(theta*) edges already cost 18 x 20 cells per pT bin, and the D0 data sample
is far smaller than either dimuon one, so a fine pT axis buys empty cells rather
than resolution. Its range is set by the *selected* spectrum, not the inclusive
one: the `Dst_pt > 5 GeV` requirement makes the selected D0 spectrum much harder
than D0 production, with a data median near 12 GeV and a 99th percentile near
67 GeV. An axis stopping at 20 GeV, which the raw spectrum would suggest, leaves
about 21% of data in overflow at unit weight. The coarse map edges must be a subset of the fine
derivation axis; this is checked at import time in `v_reweighting.py`.

## Weight definition and fallback

`background_subtracted` is `False` for every resonance. That matters more for D0
than for the dimuon resonances: the D0 mass window retains combinatorial
background whose `(pT, y, cos theta*)` shape is absorbed into the weights along
with the signal. Worth a sideband study before the D0 weights are used for
anything beyond shape diagnostics.

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

## Kinematic diagnostics

`--kinDiagnostics` is the single flag in both histmakers. It always writes the
three variables the maps are built from:

```text
kinDiagnostics_ptll
kinDiagnostics_yll
kinDiagnostics_cosThetaStarll
```

Each histmaker then adds whatever further diagnostics its final state makes
useful, under the same `kinDiagnostics_` prefix. These are deliberately *not* in
`v_reweighting.py`: what counts as a useful single-body quantity is a property
of the final state, not of the reweighting.

**Every** histogram `--kinDiagnostics` writes carries the leading
`triggerCategory` axis, including the histmaker-specific ones and including the
cases where it is a single trivial bin (J/psi and D0). That keeps the structure
identical across resonances, so a consumer needs no per-resonance or
per-histogram branching, and it means the Upsilon barrel/high split is available
on every diagnostic rather than only on the three map variables.

`dimuon_resonances_calinput.py` adds the two single-muon pT orderings:

```text
kinDiagnostics_ptlead      kinDiagnostics_ptplus
kinDiagnostics_ptsublead   kinDiagnostics_ptminus
```

No map is derived on these, so they show what a map did to a variable it was not
fitted to. `ptlead`/`ptsublead` are pT ordered and are what a leading-pT
selection acts on; `ptplus`/`ptminus` are charge ordered, which for a dimuon is
also a species ordering.

`d0_mass.py` adds the pT-ordered pair only:

```text
kinDiagnostics_ptlead      kinDiagnostics_ptsublead
```

and no charge-ordered pair, because for D0 -> K pi the charge ordering is not a
species ordering (D0 -> K- pi+ and D0bar -> K+ pi-) so each histogram would mix
kaons and pions. The species-resolved equivalents are `kinDiagnostics_K_reco_pt` and
`kinDiagnostics_pi_reco_pt`. In simulation those come from the gen-vs-reco
resolution family the same flag books (`kinDiagnostics_K_qopr`,
`kinDiagnostics_mRK_ratio` and friends), which needs gen-level truth and so is
simulation only; in data they are booked directly, on the same axis and with no
category axis, so the two are directly comparable.

The D0 daughter axis spans 0-80 GeV, sized from the *selected* spectrum: after
the `Dst_pt > 5 GeV` cut the leading daughter has a median near 9 GeV and a
99.9th percentile near 83.

Each carries the same leading `triggerCategory` axis as `vReweightInput` plus
one kinematic axis, on exactly the binning the maps are derived on, so they line
up bin for bin with the derivation input. For J/psi the category axis has a
single bin; for Upsilon it splits barrel and high the same way the maps do, so
consumers need no per-resonance branching. They are filled with the applied
weight.

Unlike `vReweightInput` the flag is not tied to `--vReweightSelection` or to a
single channel, so it works on its own for a plain look at the spectra. Without
a selection the category axis holds one catch-all bin labelled `all`:

```bash
python3 scripts/histmakers/dimuon_resonances_calinput.py \
  --resonance jpsi --era 2016PostVFP --triggers dimuon20_jpsi \
  --kinDiagnostics -j 16 -o OUTDIR
```

Because they can be booked in *both* passes, running with `--kinDiagnostics` in
the derivation and again in the application gives a direct before/after check:
the simulation histograms should move toward data in the second pass, while the
data histograms stay identical. This is the in-job closure test that
`vReweightInput` cannot provide, since it is mutually exclusive with
`--vReweightFile`. The flag adds `kinDiag` to the output filename.

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
