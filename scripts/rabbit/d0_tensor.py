"""Build a rabbit input tensor from the d0_mass.py histmaker output.

This is the D0 -> K pi analogue of scripts/rabbit/lowmass_tensor.py (which
consumes the J/psi calInput histograms). The differences are:

  * The histograms here are 3D (etaK, mRK, D0mass) or, when d0_mass.py is run
    with --pionTemplateAxes, 5D (etaK, mRK, etaPi, mRpi, D0mass), instead of the
    (eta1, eta2, pt1, pt2, mass) J/psi templates. The lineshape ("shape") axis
    is "D0mass" instead of "mass". Use --tensorDim to pick 3D or 5D (or leave it
    at 'auto' to follow the input); a 5D input can be projected down to 3D.
  * Only the muon scale variations, muon resolution variations and a flat
    background are implemented. Pixel multiplicity systematics/stats, tail
    morphing and projection options are intentionally not carried over.

Input histograms produced by d0_mass.py (3D shown; the 5D variant inserts the
etaPi, mRpi axes before D0mass):

  D0_data / output:
    hD0_data                                 (etaK, mRK, D0mass)
  D0_mc / output:
    hD0_nom                                  (etaK, mRK, D0mass)
    nominal_muonScaleSyst_responseWeights    (etaK, mRK, D0mass, unc, downUpVar)
    nominal_muonResolutionSyst_responseWeights (etaK, mRK, D0mass, smearing_variation)
"""

import argparse
import sys
from pathlib import Path

import h5py
import hist
import numpy as np

from rabbit import tensorwriter

WREMNANTS_DIR = Path(__file__).resolve().parent / "WRemnants"
if str(WREMNANTS_DIR) not in sys.path:
    sys.path.insert(0, str(WREMNANTS_DIR))

from wremnants.utilities import common
from wremnants.utilities.io_tools import base_io
from wums import ioutils, output_tools

# The lineshape axis for the D0 templates.
SHAPE_AXIS = "D0mass"

# The pion kinematic axes present only in 5D (--pionTemplateAxes) templates.
# Projecting these away (summing over them) reduces a 5D input to the 3D layout.
PION_AXES = ["etaPi", "mRpi"]

# Muon scale nuisance categories, matching the WRemnants massfit uncertainty
# helper (data_jpsi_crctn_unc_helper). One (A, e, M) triplet per eta bin.
nuisance_categories = ["A", "e", "M"]

# Resolution smearing categories from make_muon_smearing_helpers. The helper
# stores a, b, c, d per eta bin; only a, b, c are propagated as nuisances.
input_res_categories = ["a", "b", "c", "d"]
output_res_categories = ["a", "b", "c"]

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", default="./", help="output directory")
parser.add_argument("--outname", default="d0_tensor", help="output file name")
parser.add_argument(
    "--postfix",
    default=None,
    type=str,
    help="Postfix to append on output file name",
)
parser.add_argument(
    "--sparse",
    default=False,
    action="store_true",
    help="Make sparse tensor",
)
parser.add_argument(
    "--systematicType",
    choices=["log_normal", "normal"],
    default="log_normal",
    help="probability density for systematic variations",
)
parser.add_argument(
    "--d0Hdf5",
    default="/work/submit/havyn/d0_production/d0_mass_2016PostVFP.hdf5",
    help="d0_mass.py output HDF5 file to build the tensor from",
)
parser.add_argument(
    "--channel",
    default="D0",
    help="Name for the D0 fit channel",
)
parser.add_argument(
    "--tensorDim",
    choices=["auto", "3", "5"],
    default="auto",
    help=(
        "Dimensionality of the fit tensor. 'auto' (default) follows the input "
        "histograms. '3' builds a 3D (etaK, mRK, D0mass) tensor, projecting away "
        "the pion axes (etaPi, mRpi) if the input is 5D. '5' builds a 5D "
        "(etaK, mRK, etaPi, mRpi, D0mass) tensor and requires those axes to be "
        "present in the input (produced by d0_mass.py --pionTemplateAxes)."
    ),
)
parser.add_argument(
    "--skipMuonScale",
    default=False,
    action="store_true",
    help="Do not add muon scale systematics to the D0 signal process.",
)
parser.add_argument(
    "--skipMuonResolution",
    default=False,
    action="store_true",
    help="Do not add muon resolution systematics to the D0 signal process.",
)
parser.add_argument(
    "--manualScaleVariations",
    default=False,
    action="store_true",
    help="Use manual scale variations instead of reweights",
)
parser.add_argument(
    "--clip",
    default=0,
    type=float,
    help="clip values in MC and systematics at this threshold",
)
parser.add_argument(
    "--mcEventThresh",
    default=-1,
    type=float,
    help=(
        "Event threshold in raw MC for a kinematic (non-mass) cell being "
        "included in the fit. Disabled by default. Zeros both MC and data in "
        "failing cells if specified."
    ),
)
parser.add_argument(
    "--dataEventThresh",
    default=0,
    type=float,
    help=(
        "Data event threshold for a non-mass cell being included in the fit. "
        "The active mask is applied consistently to data, MC, background and "
        "systematics."
    ),
)
parser.add_argument(
    "--dummyData",
    default=None,
    type=float,
    help=(
        "If set, replace the data with a scaled copy of the nominal MC (hD0_nom) "
        "whose total event count equals this value, instead of using the actual "
        "hD0_data histogram. Produces an Asimov-like dataset of controllable size."
    ),
)
parser.add_argument(
    "--rescaleScaleAndResolution",
    default=1.0,
    type=float,
    help=(
        "Scale the deviation of the D0 muon scale and resolution variations "
        "from nominal when building the tensor. Values >1 inflate the "
        "effective prefit width."
    ),
)
args = parser.parse_args()

if args.rescaleScaleAndResolution <= 0:
    raise ValueError("--rescaleScaleAndResolution must be positive")

if args.dummyData is not None and args.dummyData <= 0:
    raise ValueError("--dummyData must be positive")


# modifies the input histogram variable and then returns it for convenience
def clip_values(dummy_hist, lim=1e-6, variances=True):
    values = np.copy(dummy_hist.values())
    values_copy = np.copy(np.clip(values, lim, np.inf))
    dummy_hist.values()[...] = values_copy
    if variances:
        variances = dummy_hist.variances()
        if variances is not None:
            variances_copy = np.copy(np.clip(variances, lim, np.inf))
            dummy_hist.variances()[...] = variances_copy
    return dummy_hist


def populated_cells(dummy_hist, shape_axis=SHAPE_AXIS, event_thresh=0):
    """Return the nonempty cells after integrating over the lineshape axis."""
    shape_axis_index = dummy_hist.axes.name.index(shape_axis)
    return np.sum(dummy_hist.values(), axis=shape_axis_index) > max(event_thresh, 0)


def zero_inactive_cells(dummy_hist, active_cells, shape_axis=SHAPE_AXIS):
    """Restore exact zeros outside active kinematic cells after clipping."""
    shape_axis_index = dummy_hist.axes.name.index(shape_axis)
    active_bins = np.expand_dims(active_cells, axis=shape_axis_index)
    dummy_hist.values()[...] = np.where(active_bins, dummy_hist.values(), 0.0)
    variances = dummy_hist.variances()
    if variances is not None:
        dummy_hist.variances()[...] = np.where(active_bins, variances, 0.0)
    return dummy_hist


def make_active_background(template_hist, active_cells, shape_axis=SHAPE_AXIS):
    """Build a flat background only in populated kinematic cells.

    The per-bin level is proportional to the physical width of the shape axis
    so the background is flat in physical mass density.
    """
    shape_axis_index = template_hist.axes.name.index(shape_axis)
    active_bins = np.expand_dims(active_cells, axis=shape_axis_index)
    shape_axis_obj = template_hist.axes[shape_axis]

    shape_weights = np.asarray(shape_axis_obj.widths, dtype=float)
    shape_weights = np.where(
        np.isfinite(shape_weights) & (shape_weights > 0.0), shape_weights, 0.0
    )
    weight_shape = [1] * len(template_hist.axes)
    weight_shape[shape_axis_index] = shape_axis_obj.size
    shape_weights = shape_weights.reshape(weight_shape)

    background_hist = hist.Hist(*template_hist.axes, storage=hist.storage.Weight())
    background_hist.values()[...] = np.where(
        active_bins,
        np.broadcast_to(shape_weights, template_hist.values().shape),
        0.0,
    )
    background_hist.variances()[...] = 0
    return background_hist


# numerical stupidity can cause problems - fix by setting vars to nom if they
# differ by a rounding error. modifies the input histogram and then returns it.
def trim_variations(var, nom, thresh=1e-5):
    var_values = var.values()
    nom_values = nom.values()
    indices = np.where(np.abs(var_values - nom_values) < thresh)
    var.values()[indices] = nom_values[indices]
    return var


def scale_variation_around_nominal(var, nom, scale):
    if scale == 1.0:
        return var
    values = nom.values() + scale * (var.values() - nom.values())
    var.values()[...] = values
    variances = var.variances()
    nom_variances = nom.variances()
    if variances is not None and nom_variances is not None:
        var.variances()[...] = nom_variances + scale**2 * (variances - nom_variances)
    return var


def materialize(obj):
    return obj.get() if isinstance(obj, ioutils.H5PickleProxy) else obj


def has_pion_axes(dummy_hist):
    """True if the histogram carries the 5D pion kinematic axes."""
    names = set(dummy_hist.axes.name)
    return all(ax in names for ax in PION_AXES)


def project_out_pion_axes(dummy_hist):
    """Sum a 5D histogram over the pion axes (etaPi, mRpi), returning the 3D
    layout. The remaining axes keep their original order, so the extra
    systematic axes (unc/downUpVar, smearing_variation) survive unchanged."""
    keep = [ax.name for ax in dummy_hist.axes if ax.name not in PION_AXES]
    return dummy_hist.project(*keep)


def resolve_tensor_dim(channel, tensor_dim):
    """Reconcile the requested --tensorDim with the input dimensionality,
    projecting the channel histograms from 5D down to 3D when needed. Modifies
    and returns the channel dict."""
    input_is_5d = has_pion_axes(channel["D0_mc"])
    if tensor_dim == "auto":
        target_5d = input_is_5d
    elif tensor_dim == "5":
        if not input_is_5d:
            raise RuntimeError(
                "--tensorDim 5 requires 5D input histograms with the "
                f"{PION_AXES} axes; input has axes "
                f"{list(channel['D0_mc'].axes.name)}. Run d0_mass.py with "
                "--pionTemplateAxes."
            )
        target_5d = True
    else:  # "3"
        target_5d = False

    if input_is_5d and not target_5d:
        print(
            f"Projecting 5D input down to 3D by summing over {PION_AXES} "
            "(--tensorDim 3)"
        )
        for key in ("D0_data", "D0_mc", "D0_scale_syst", "D0_resolution_syst"):
            if channel.get(key) is not None:
                channel[key] = project_out_pion_axes(channel[key])
    elif not input_is_5d and target_5d:
        # Unreachable: the "5" branch already raised. Guard for clarity.
        raise RuntimeError("Cannot build a 5D tensor from 3D input histograms")

    dim = 5 if target_5d else 3
    print(f"Building {dim}D tensor (axes: {list(channel['D0_mc'].axes.name)})")
    return channel


def load_d0_histograms(path):
    """Load the D0 data and MC histograms from a d0_mass.py output file."""
    with h5py.File(path, "r") as h5file:
        results = base_io.load_results_h5py(h5file)

        datasets = {}
        input_meta_info = None
        for dataset_name, result in results.items():
            if dataset_name == "meta_info":
                # The d0_mass.py command/args/git info; propagated into the
                # tensor output as meta_info_input so provenance is preserved.
                input_meta_info = materialize(result)
                continue
            if not isinstance(result, dict):
                continue
            histograms = {}
            for hist_name, hist_obj in result.get("output", {}).items():
                histograms[hist_name] = materialize(hist_obj)
            datasets[dataset_name] = histograms

    if "D0_data" not in datasets or "D0_mc" not in datasets:
        raise RuntimeError(
            f"Expected 'D0_data' and 'D0_mc' datasets in {path}, "
            f"found {sorted(datasets)}"
        )

    data = datasets["D0_data"]
    mc = datasets["D0_mc"]

    required_data = ["hD0_data"]
    required_mc = ["hD0_nom"]
    if not args.skipMuonScale:
        required_mc.append("nominal_muonScaleSyst_responseWeights")
    if not args.skipMuonResolution:
        required_mc.append("nominal_muonResolutionSyst_responseWeights")
    missing = [n for n in required_data if n not in data] + [
        n for n in required_mc if n not in mc
    ]
    if missing:
        raise RuntimeError(
            f"Missing histograms in {path}: {missing}. "
            "Run d0_mass.py with the muon scale/resolution systematics enabled."
        )

    channel = {
        "name": args.channel,
        "D0_data": data["hD0_data"],
        "D0_mc": mc["hD0_nom"],
        "D0_scale_syst": (
            None
            if args.skipMuonScale
            else (
                mc["nominal_muonScaleSyst_responseWeights"]
                if not args.manualScaleVariations
                else mc["nominal_muonScaleSyst_manual"]
            )
        ),
        "D0_resolution_syst": (
            None
            if args.skipMuonResolution
            else mc["nominal_muonResolutionSyst_responseWeights"]
        ),
        "meta_info_input": input_meta_info,
    }
    return channel


def split_scale_variations(hsyst, hist_mc, scaling_factor, thresh):
    """Split the muon scale response-weight histogram into per-nuisance up/down
    template pairs, one per (category, eta bin)."""
    if "unc" not in hsyst.axes.name or "downUpVar" not in hsyst.axes.name:
        raise RuntimeError(
            "Expected muon scale histogram with 'unc' and 'downUpVar' axes"
        )

    # WRemnants uses downUpVar index 0 for down and 1 for up.
    down_index = 0
    up_index = 1
    if hsyst.axes["unc"].size % len(nuisance_categories):
        raise RuntimeError(
            f"Expected a multiple of {len(nuisance_categories)} uncertainty bins "
            f"in the scale systematic histogram, found {hsyst.axes['unc'].size}"
        )
    num_eta_bins = hsyst.axes["unc"].size // len(nuisance_categories)

    up_variations = []
    down_variations = []
    variations = []
    groups = []
    for param_index, nuisance_category in enumerate(nuisance_categories):
        for i in range(num_eta_bins):
            unc_index = i * len(nuisance_categories) + param_index
            up = hsyst[{"unc": unc_index, "downUpVar": up_index}] * scaling_factor
            down = hsyst[{"unc": unc_index, "downUpVar": down_index}] * scaling_factor
            up = scale_variation_around_nominal(
                up, hist_mc, args.rescaleScaleAndResolution
            )
            down = scale_variation_around_nominal(
                down, hist_mc, args.rescaleScaleAndResolution
            )
            up_variations.append(
                trim_variations(clip_values(up, lim=args.clip), hist_mc, thresh)
            )
            down_variations.append(
                trim_variations(clip_values(down, lim=args.clip), hist_mc, thresh)
            )
            variations.append(f"{nuisance_category}{i}")
            groups.append(nuisance_category)

    return up_variations, down_variations, variations, groups


def split_resolution_variations(hsyst, hist_mc, scaling_factor, thresh):
    """Split the muon resolution smearing histogram into per-nuisance templates,
    one per (category in {a, b, c}, eta bin)."""
    if "smearing_variation" not in hsyst.axes.name:
        raise RuntimeError(
            "Expected muon resolution histogram with 'smearing_variation' axis"
        )

    if hsyst.axes["smearing_variation"].size % len(input_res_categories):
        raise RuntimeError(
            f"Expected a multiple of {len(input_res_categories)} resolution "
            f"variations, found {hsyst.axes['smearing_variation'].size}"
        )
    num_eta_bins = hsyst.axes["smearing_variation"].size // len(input_res_categories)

    variations = []
    names = []
    groups = []
    for param_name in output_res_categories:
        param_index = input_res_categories.index(param_name)
        for ieta in range(num_eta_bins):
            variation_index = ieta * len(input_res_categories) + param_index
            variation = hsyst[{"smearing_variation": variation_index}] * scaling_factor
            variation = scale_variation_around_nominal(
                variation, hist_mc, args.rescaleScaleAndResolution
            )
            variations.append(
                trim_variations(clip_values(variation, lim=args.clip), hist_mc, thresh)
            )
            names.append(f"res_{param_name}{ieta}")
            groups.append(f"res_{param_name}")

    return variations, names, groups


print(args.systematicType)
writer = tensorwriter.TensorWriter(
    sparse=args.sparse,
    systematic_type=args.systematicType,
)

resultdict = load_d0_histograms(args.d0Hdf5)
resultdict = resolve_tensor_dim(resultdict, args.tensorDim)
sample = resultdict["name"]
print(f"processing {sample}")

hist_mc = resultdict["D0_mc"]
hist_data = resultdict["D0_data"]

if args.dummyData is not None:
    # Replace the data with a scaled copy of the nominal MC (hD0_nom) whose total
    # event count equals --dummyData (values scaled by s, variances by s^2). Yields
    # an Asimov-like dataset of controllable size for expected-sensitivity studies.
    mc_total = float(np.sum(hist_mc.values()))
    if mc_total <= 0:
        raise RuntimeError("Nominal MC has no events; cannot build --dummyData")
    dummy_scale = args.dummyData / mc_total
    hist_data = hist_mc * dummy_scale
    print(
        f"--dummyData: data replaced by nominal MC scaled to {args.dummyData:g} "
        f"events (factor {dummy_scale:.6g})"
    )

data_total = float(np.sum(hist_data.values()))
mc_total = float(np.sum(hist_mc.values()))
if not np.isfinite(data_total) or data_total < 0:
    raise RuntimeError(f"Nominal data integral is invalid: {data_total}")
if not np.isfinite(mc_total) or mc_total <= 0:
    raise RuntimeError(f"Nominal MC integral is invalid: {mc_total}")
scaling_factor = data_total / mc_total
print("scaling factor:", scaling_factor)
hist_mc = hist_mc * scaling_factor

# active kinematic (non-mass) cells: nonempty MC, optionally intersected with a
# data occupancy requirement. In 3D these are (etaK, mRK); in 5D also (etaPi, mRpi).
signal_active_cells = populated_cells(hist_mc, event_thresh=args.mcEventThresh)
if args.dataEventThresh > 0:
    signal_active_cells = signal_active_cells & populated_cells(
        hist_data, event_thresh=args.dataEventThresh
    )
print(
    f"Active {sample} cells: "
    f"{np.count_nonzero(signal_active_cells)}/{signal_active_cells.size}"
)

clip_values(hist_mc, lim=args.clip, variances=True)
zero_inactive_cells(hist_mc, signal_active_cells)
if args.mcEventThresh < 0:
    fit_active_cells = signal_active_cells | populated_cells(hist_data)
else:
    fit_active_cells = signal_active_cells
    zero_inactive_cells(hist_data, fit_active_cells)

thresh = 1e-5
if resultdict["D0_scale_syst"] is not None:
    up_variations, down_variations, scale_names, scale_groups = split_scale_variations(
        resultdict["D0_scale_syst"], hist_mc, scaling_factor, thresh
    )
else:
    up_variations, down_variations, scale_names, scale_groups = [], [], [], []

if resultdict["D0_resolution_syst"] is not None:
    resolution_variations, resolution_names, resolution_groups = (
        split_resolution_variations(
            resultdict["D0_resolution_syst"], hist_mc, scaling_factor, thresh
        )
    )
else:
    resolution_variations, resolution_names, resolution_groups = [], [], []

writer.add_channel(hist_mc.axes, sample)
# signal=True means the cross section floats
writer.add_process(hist_mc, f"sig_{sample}", sample, signal=True)

background_hist = make_active_background(hist_mc, fit_active_cells)
writer.add_process(
    background_hist,
    f"background_{sample}",
    sample,
    signal=False,
    variances=np.zeros_like(background_hist.values()),
)

writer.add_data(hist_data, sample)

for up_var, down_var, var, group in zip(
    up_variations, down_variations, scale_names, scale_groups
):
    print(var)
    writer.add_systematic(
        [up_var, down_var],
        var,
        f"sig_{sample}",
        sample,
        groups=[group],
        constrained=True,
    )

for variation, var, group in zip(
    resolution_variations, resolution_names, resolution_groups
):
    print(var)
    writer.add_systematic(
        variation,
        var,
        f"sig_{sample}",
        sample,
        groups=[group],
        constrained=True,
    )

directory = args.output
if directory == "":
    directory = "./"
filename = args.outname
if args.postfix:
    filename += f"_{args.postfix}"

# Propagate provenance into the tensor: this script's command in meta_info and
# the upstream d0_mass.py command (read from the input file) in meta_info_input,
# mirroring setupRabbit.py.
meta = {
    "meta_info": output_tools.make_meta_info_dict(args=args, wd=common.base_dir),
    "meta_info_input": resultdict.get("meta_info_input"),
}
writer.write(outfolder=directory, outfilename=filename, meta_data_dict=meta)
