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

jpsi_channel_names = {
    "dimuon20_jpsi": "JPsi_prompt",
    "doublemu4_jpsitrk_displaced": "JPsi_displaced",
}

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", default="./", help="output directory")
parser.add_argument("--outname", default="test_tensor", help="output file name")
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
    "--symmetrizeAll",
    default=False,
    action="store_true",
    help="Make fully symmetric tensor",
)
parser.add_argument(
    "--skipMaskedChannels",
    default=False,
    action="store_true",
    help="Skip adding masked channels",
)
parser.add_argument(
    "--systematicType",
    choices=["log_normal", "normal"],
    default="log_normal",
    help="probability density for systematic variations",
)
parser.add_argument(
    "--calinputHdf5",
    default=str(WREMNANTS_DIR / "dimuon_resonances_calinput_jpsi_2016PostVFP.hdf5"),
    help="WRemnants calInput-style HDF5 file to use for the JPsi channel",
)
parser.add_argument(
    "--jpsiChannels",
    nargs="+",
    choices=list(jpsi_channel_names.values()),
    default=list(jpsi_channel_names.values()),
    help="JPsi channels to add to the tensor",
)
parser.add_argument(
    "--mcEventThresh",
    default=-1,
    type=float,
    help="event threshold in raw MC for a 4d bin being included in the fit. Disabled by default. Zeros both MC and data in failing bins if specified",
)
parser.add_argument(
    "--clip",
    default=0,
    type=float,
    help="clip values in MC and data at this threshold",
)
parser.add_argument(
    "--dataEventThresh",
    default=0,
    type=float,
    help=(
        "Data event threshold for a non-mass cell being included in the fit. "
        "The active mask is still applied consistently to data, MC, background, "
        "and systematics so sparse mode removes those cells."
    ),
)
parser.add_argument(
    "--mcTailEventThresh",
    default=0,
    type=float,
    help=(
        "Minimum rescaled MC signal yield required in each low/high mass-tail "
        "region of a non-mass cell. If positive, cells with poorly populated "
        "MC tails are removed from data, MC, background, and systematics."
    ),
)
parser.add_argument(
    "--mcTailBins",
    default=3,
    type=int,
    help=(
        "Number of mass bins at each edge used for --mcTailEventThresh. "
        "For example, 3 checks the first three and last three mass bins."
    ),
)
parser.add_argument(
    "--mcTailRequire",
    choices=["both", "either"],
    default="both",
    help=(
        "Use 'both' to require populated low and high MC tails, or 'either' "
        "to keep cells where at least one tail is populated."
    ),
)
parser.add_argument(
    "--massRebin",
    default=1,
    type=int,
    help=(
        "Integer rebin factor for the mass axis applied while building the "
        "tensor. For the nominal 25-bin J/psi histograms, --massRebin 5 "
        "produces 5 mass bins."
    ),
)
parser.add_argument(
    "--massMin",
    default=None,
    type=float,
    help="Minimum mass-bin center to keep while building the tensor.",
)
parser.add_argument(
    "--massMax",
    default=None,
    type=float,
    help="Maximum mass-bin center to keep while building the tensor.",
)
parser.add_argument(
    "--massOnly",
    default=False,
    action="store_true",
    help="Project JPsi channels and systematics to the mass axis only",
)
parser.add_argument(
    "--projectAxes",
    nargs="*",
    default=None,
    choices=["eta1", "eta2", "pt1", "pt2"],
    help=(
        "Project JPsi channels and systematics to these axes plus mass. "
        "For example: --projectAxes eta1 eta2"
    ),
)
parser.add_argument(
    "--skipPixelMultiplicity",
    default=False,
    action="store_true",
    help="Do not add pixel multiplicity systematics to the JPsi signal process.",
)
parser.add_argument(
    "--skipMuonResolution",
    default=False,
    action="store_true",
    help="Do not add muon resolution systematics to the JPsi signal process.",
)
parser.add_argument(
    "--rescaleScaleAndResolution",
    default=1.0,
    type=float,
    help=(
        "Scale the deviation of JPsi muon scale and resolution variations from "
        "nominal when building the tensor. Values >1 inflate the effective "
        "prefit width."
    ),
)
parser.add_argument(
    "--jpsiTailMorph",
    default=False,
    action="store_true",
    help=(
        "Add a J/psi signal-only mass tail morph systematic. The morph stretches "
        "and compresses the mass shape around --jpsiTailMorphCenter while "
        "preserving the signal yield in each non-mass cell."
    ),
)
parser.add_argument(
    "--jpsiTailMorphStrength",
    default=0.02,
    type=float,
    help=(
        "Fractional mass-axis stretch used for --jpsiTailMorph. For example, "
        "0.02 makes up/down templates with +/-2% width changes around the "
        "J/psi mass."
    ),
)
parser.add_argument(
    "--jpsiTailMorphCenter",
    default=3.0969,
    type=float,
    help="Mass value used as the fixed point for --jpsiTailMorph.",
)
args = parser.parse_args()

if args.rescaleScaleAndResolution <= 0:
    raise ValueError("--rescaleScaleAndResolution must be positive")
if args.jpsiTailMorphStrength <= 0:
    raise ValueError("--jpsiTailMorphStrength must be positive")
if args.massRebin <= 0:
    raise ValueError("--massRebin must be a positive integer")
if args.massMin is not None and args.massMax is not None and args.massMin >= args.massMax:
    raise ValueError("--massMin must be smaller than --massMax")
if args.mcTailEventThresh < 0:
    raise ValueError("--mcTailEventThresh cannot be negative")
if args.mcTailBins <= 0:
    raise ValueError("--mcTailBins must be a positive integer")


def rebin_mass(dummy_hist, factor):
    if factor == 1:
        return dummy_hist
    return dummy_hist[{"mass": hist.rebin(factor)}]


def restrict_mass_window(dummy_hist, mass_min=None, mass_max=None):
    if mass_min is None and mass_max is None:
        return dummy_hist

    mass_axis = dummy_hist.axes["mass"]
    centers = np.asarray(mass_axis.centers)
    keep = np.ones_like(centers, dtype=bool)
    if mass_min is not None:
        keep &= centers >= mass_min
    if mass_max is not None:
        keep &= centers <= mass_max
    if not np.any(keep):
        raise ValueError(
            f"Mass window [{mass_min}, {mass_max}] removes all {mass_axis.size} bins"
        )
    start = int(np.argmax(keep))
    stop = int(len(keep) - np.argmax(keep[::-1]))
    if not np.all(keep[start:stop]):
        raise ValueError("Mass window selection is unexpectedly non-contiguous")
    print(
        f"Restrict mass window to centers in [{mass_min}, {mass_max}], "
        f"keeping bins {start}:{stop} of {mass_axis.size}"
    )
    return dummy_hist[{"mass": slice(start, stop)}]


# modifies the input histogram variable and then returns it for convenience
def clip_values(dummy_hist, lim=1e-6, variances=True):
    # return dummy_hist #try getting rid of this
    values = np.copy(dummy_hist.values())
    values_copy = np.copy(np.clip(values, lim, np.inf))
    dummy_hist.values()[...] = values_copy
    if variances:
        variances = dummy_hist.variances()
        if variances is not None:
            variances_copy = np.copy(np.clip(variances, lim, np.inf))
            dummy_hist.variances()[...] = variances_copy
    return dummy_hist


def populated_cells(dummy_hist, shape_axis="mass", event_thresh=0):
    """Return the nonempty cells after integrating over the lineshape axis."""
    shape_axis_index = dummy_hist.axes.name.index(shape_axis)
    return np.sum(dummy_hist.values(), axis=shape_axis_index) > max(event_thresh, 0)


def cells_with_populated_mc_tails(
    dummy_hist,
    shape_axis="mass",
    event_thresh=0,
    tail_bins=3,
    require="both",
):
    """Return cells where the MC signal template populates the mass tails."""
    shape_axis_index = dummy_hist.axes.name.index(shape_axis)
    shape_axis_size = dummy_hist.axes[shape_axis].size
    if 2 * tail_bins > shape_axis_size:
        raise ValueError(
            f"--mcTailBins={tail_bins} is too large for {shape_axis_size} "
            f"{shape_axis} bins"
        )

    values = np.moveaxis(dummy_hist.values(), shape_axis_index, -1)
    low_tail = np.sum(values[..., :tail_bins], axis=-1)
    high_tail = np.sum(values[..., -tail_bins:], axis=-1)
    low_ok = low_tail > event_thresh
    high_ok = high_tail > event_thresh
    if require == "both":
        return low_ok & high_ok
    if require == "either":
        return low_ok | high_ok
    raise ValueError(f"Unsupported tail requirement {require}")


def zero_inactive_cells(dummy_hist, active_cells, shape_axis="mass"):
    """Restore exact zeros outside active kinematic cells after clipping."""
    shape_axis_index = dummy_hist.axes.name.index(shape_axis)
    active_bins = np.expand_dims(active_cells, axis=shape_axis_index)
    dummy_hist.values()[...] = np.where(active_bins, dummy_hist.values(), 0.0)
    variances = dummy_hist.variances()
    if variances is not None:
        dummy_hist.variances()[...] = np.where(active_bins, variances, 0.0)
    return dummy_hist


def make_active_background(template_hist, active_cells, shape_axis="mass"):
    """Build a flat background only in populated kinematic cells."""
    shape_axis_index = template_hist.axes.name.index(shape_axis)
    active_bins = np.expand_dims(active_cells, axis=shape_axis_index)
    shape_axis_obj = template_hist.axes[shape_axis]
    shape_metadata = getattr(shape_axis_obj, "metadata", None)
    if (
        isinstance(shape_metadata, dict)
        and "physical_widths_nd" in shape_metadata
    ):
        shape_weights = np.asarray(shape_metadata["physical_widths_nd"], dtype=float)
        if shape_weights.shape != template_hist.values().shape:
            raise ValueError(
                f"Physical width tensor for shape axis '{shape_axis}' has shape "
                f"{shape_weights.shape}, expected {template_hist.values().shape}"
            )
        shape_weights = np.where(
            np.isfinite(shape_weights) & (shape_weights > 0.0),
            shape_weights,
            0.0,
        )
        print(
            f"Build background using conditional quantile volumes for axis '{shape_axis}'"
        )
    elif (
        isinstance(shape_metadata, dict)
        and "physical_widths" in shape_metadata
    ):
        shape_weights = np.asarray(shape_metadata["physical_widths"], dtype=float)
        if len(shape_weights) != shape_axis_obj.size:
            raise ValueError(
                f"Physical widths for shape axis '{shape_axis}' have length "
                f"{len(shape_weights)}, expected {shape_axis_obj.size}"
            )
        shape_weights = np.where(
            np.isfinite(shape_weights) & (shape_weights > 0.0),
            shape_weights,
            0.0,
        )
        weight_shape = [1] * len(template_hist.axes)
        weight_shape[shape_axis_index] = shape_axis_obj.size
        shape_weights = shape_weights.reshape(weight_shape)
    else:
        shape_weights = np.ones(shape_axis_obj.size, dtype=float)
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


def project_to_axes_with_mass(dummy_hist, keep_axes=None, shape_axis="mass"):
    """Project to selected non-mass axes plus mass."""
    keep_axes = [] if keep_axes is None else list(keep_axes)
    shape_axis_index = dummy_hist.axes.name.index(shape_axis)
    keep_names = set(keep_axes + [shape_axis])
    sum_axes = tuple(
        i for i, axis in enumerate(dummy_hist.axes) if axis.name not in keep_names
    )
    values = np.sum(dummy_hist.values(), axis=sum_axes)
    variances = dummy_hist.variances()
    variances = None if variances is None else np.sum(variances, axis=sum_axes)

    out_axes = []
    out_axes.extend(dummy_hist.axes[name] for name in keep_axes)
    out_axes.append(dummy_hist.axes[shape_axis])
    storage = hist.storage.Weight() if variances is not None else hist.storage.Double()
    out = hist.Hist(*out_axes, storage=storage)
    out.values()[...] = values
    if variances is not None:
        out.variances()[...] = variances
    return out


# def hist_rescale(hist, lumi, xsec, mc_events):
#     return hist * lumi * xsec * 1000 / mc_events


# numerical stupidity can cause problems - fix by setting vars to nom if they differ by a rounding error
# modifies the input histogram and then returns it
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


def make_mass_stretch_variation(
    nominal,
    strength,
    center=3.0969,
    shape_axis="mass",
):
    """Stretch a histogram mass shape around a fixed point, preserving cell yields."""
    shape_axis_index = nominal.axes.name.index(shape_axis)
    mass_axis = nominal.axes[shape_axis]
    mass_centers = np.asarray(mass_axis.centers)
    mass_widths = np.asarray(mass_axis.widths)

    values = np.asarray(nominal.values())
    values_moved = np.moveaxis(values, shape_axis_index, -1)
    flat_values = values_moved.reshape((-1, mass_axis.size))

    source_centers = center + (mass_centers - center) / (1.0 + strength)
    stretched = np.zeros_like(flat_values)
    for irow, row in enumerate(flat_values):
        nominal_sum = np.sum(row)
        if nominal_sum <= 0:
            continue

        density = row / mass_widths
        stretched_density = np.interp(
            source_centers,
            mass_centers,
            density,
            left=0.0,
            right=0.0,
        )
        stretched_row = stretched_density * mass_widths / (1.0 + strength)
        stretched_sum = np.sum(stretched_row)
        if stretched_sum > 0:
            stretched_row *= nominal_sum / stretched_sum
        stretched[irow] = stretched_row

    out = nominal.copy()
    stretched = stretched.reshape(values_moved.shape)
    out.values()[...] = np.moveaxis(stretched, -1, shape_axis_index)

    variances = out.variances()
    nominal_variances = nominal.variances()
    if variances is not None and nominal_variances is not None:
        variances[...] = nominal_variances

    return out


def materialize(obj):
    return obj.get() if isinstance(obj, ioutils.H5PickleProxy) else obj


def load_calinput_datasets(path):
    with h5py.File(path, "r") as h5file:
        results = base_io.load_results_h5py(h5file)

        datasets = {}
        input_meta_info = None
        for dataset_name, result in results.items():
            if dataset_name == "meta_info":
                # The dimuon_resonances_calinput.py command/args/git info;
                # propagated into the tensor output as meta_info_input so
                # provenance is preserved.
                input_meta_info = materialize(result)
                continue
            if not isinstance(result, dict):
                continue
            histograms = {}
            for hist_name, hist_obj in result.get("output", {}).items():
                histograms[hist_name] = materialize(hist_obj)
            datasets[dataset_name] = histograms

    return datasets, input_meta_info


def make_jpsi_channel(dataset_label, histograms):
    required = [
        "JPsi_data",
        "JPsi_mc",
        "nominal_muonScaleSyst_responseWeights",
    ]
    if not args.skipMuonResolution:
        required.append("nominal_muonResolutionSyst_responseWeights")
    if not args.skipPixelMultiplicity:
        required.append("nominal_pixelMultiplicitySyst")
    missing = [name for name in required if name not in histograms]
    if missing:
        raise RuntimeError(
            f"Missing histograms for JPsi channel '{dataset_label}': {missing}. "
            "Run dimuon_resonances_calinput.py with MC scale/pixel systematics enabled."
        )

    channel = {
        "name": jpsi_channel_names.get(
            dataset_label, f"JPsi_{dataset_label}" if dataset_label else "JPsi"
        ),
        "JPsi_data": histograms["JPsi_data"],
        "JPsi_mc": histograms["JPsi_mc"],
        "JPsi_syst": histograms["nominal_muonScaleSyst_responseWeights"],
    }
    if not args.skipMuonResolution:
        channel["JPsi_resolution_syst"] = histograms[
            "nominal_muonResolutionSyst_responseWeights"
        ]
    if not args.skipPixelMultiplicity:
        channel["JPsi_pixel_syst"] = histograms["nominal_pixelMultiplicitySyst"]
    if not args.skipPixelMultiplicity and "nominal_pixelMultiplicityStat" in histograms:
        channel["JPsi_pixel_stat"] = histograms["nominal_pixelMultiplicityStat"]
    else:
        channel["JPsi_pixel_stat"] = None

    return channel


def load_jpsi_channels_from_calinput(path):
    datasets, input_meta_info = load_calinput_datasets(path)

    # Backward compatibility for the original HDF5 layout with one data and
    # one MC dataset. The new layout uses one pair of datasets per trigger.
    if "jpsi_data" in datasets and "jpsi_mc" in datasets:
        histograms = {**datasets["jpsi_data"], **datasets["jpsi_mc"]}
        return [make_jpsi_channel("", histograms)], input_meta_info

    data_prefix = "jpsi_data_"
    mc_prefix = "jpsi_mc_"
    labels_data = {
        name[len(data_prefix) :] for name in datasets if name.startswith(data_prefix)
    }
    labels_mc = {
        name[len(mc_prefix) :] for name in datasets if name.startswith(mc_prefix)
    }
    if labels_data != labels_mc:
        raise RuntimeError(
            f"Mismatched JPsi trigger channels in {path}: "
            f"data-only={sorted(labels_data - labels_mc)}, "
            f"mc-only={sorted(labels_mc - labels_data)}"
        )
    if not labels_data:
        raise RuntimeError(f"No JPsi trigger-channel datasets found in {path}")

    channels = []
    for label in sorted(labels_data):
        histograms = {
            **datasets[f"{data_prefix}{label}"],
            **datasets[f"{mc_prefix}{label}"],
        }
        channels.append(make_jpsi_channel(label, histograms))
    return channels, input_meta_info


def split_calinput_scale_variations(
    hsyst,
    hist_mc,
    scaling_factor,
    thresh,
    projection_axes=None,
    mass_rebin=1,
    mass_min=None,
    mass_max=None,
):
    up_variations = []
    down_variations = []

    if "unc" not in hsyst.axes.name or "downUpVar" not in hsyst.axes.name:
        raise RuntimeError(
            "Expected WRemnants systematic histogram with 'unc' and 'downUpVar' axes"
        )

    # WRemnants uses downUpVar index 0 for down and 1 for up.
    down_index = 0
    up_index = 1
    if hsyst.axes["unc"].size % len(nuisance_categories):
        raise RuntimeError(
            f"Expected a multiple of {len(nuisance_categories)} uncertainty bins "
            f"in the calinput systematic histogram, found {hsyst.axes['unc'].size}"
        )
    num_eta_bins = hsyst.axes["unc"].size // len(nuisance_categories)
    variations = []
    groups = []

    for param_index, nuisance_category in enumerate(nuisance_categories):
        for i in range(num_eta_bins):
            unc_index = i * len(nuisance_categories) + param_index
            up = hsyst[{"unc": unc_index, "downUpVar": up_index}] * scaling_factor
            down = hsyst[{"unc": unc_index, "downUpVar": down_index}] * scaling_factor
            up = rebin_mass(up, mass_rebin)
            down = rebin_mass(down, mass_rebin)
            up = restrict_mass_window(up, mass_min, mass_max)
            down = restrict_mass_window(down, mass_min, mass_max)
            if projection_axes is not None:
                up = project_to_axes_with_mass(up, projection_axes)
                down = project_to_axes_with_mass(down, projection_axes)
            up = scale_variation_around_nominal(
                up, hist_mc, args.rescaleScaleAndResolution
            )
            down = scale_variation_around_nominal(
                down, hist_mc, args.rescaleScaleAndResolution
            )
            up_variations.append(trim_variations(clip_values(up, lim=args.clip), hist_mc, thresh))
            down_variations.append(trim_variations(clip_values(down, lim=args.clip), hist_mc, thresh))
            variations.append(f"{nuisance_category}{i}")
            groups.append(nuisance_category)

    return up_variations, down_variations, variations, groups


def split_calinput_resolution_variations(
    hsyst,
    hist_mc,
    scaling_factor,
    thresh,
    projection_axes=None,
    mass_rebin=1,
    mass_min=None,
    mass_max=None,
):
    if "smearing_variation" not in hsyst.axes.name:
        raise RuntimeError(
            "Expected WRemnants resolution histogram with 'smearing_variation' axis"
        )

    input_res_categories = ["a", "b", "c", "d"]
    output_res_categories = ["a", "b", "c"]
    if hsyst.axes["smearing_variation"].size % len(input_res_categories):
        raise RuntimeError(
            f"Expected a multiple of {len(input_res_categories)} resolution variations, "
            f"found {hsyst.axes['smearing_variation'].size}"
        )
    num_eta_bins = hsyst.axes["smearing_variation"].size // len(input_res_categories)

    resolution_variations = []
    resolution_names = []
    resolution_groups = []
    for param_name in output_res_categories:
        param_index = input_res_categories.index(param_name)
        for ieta in range(num_eta_bins):
            variation_index = ieta * len(input_res_categories) + param_index
            variation = hsyst[{"smearing_variation": variation_index}] * scaling_factor
            variation = rebin_mass(variation, mass_rebin)
            variation = restrict_mass_window(variation, mass_min, mass_max)
            if projection_axes is not None:
                variation = project_to_axes_with_mass(variation, projection_axes)
            variation = scale_variation_around_nominal(
                variation, hist_mc, args.rescaleScaleAndResolution
            )
            resolution_variations.append(
                trim_variations(clip_values(variation, lim=args.clip), hist_mc, thresh)
            )
            resolution_names.append(f"res_{param_name}{ieta}")
            resolution_groups.append(f"res_{param_name}")

    return resolution_variations, resolution_names, resolution_groups


def split_calinput_pixel_multiplicity_variations(
    hsyst,
    hist_mc,
    scaling_factor,
    thresh,
    name_prefix="pixel_multiplicity_syst",
    projection_axes=None,
    mass_rebin=1,
    mass_min=None,
    mass_max=None,
):
    if "var" not in hsyst.axes.name:
        raise RuntimeError(
            f"Expected WRemnants pixel multiplicity histogram with 'var' axis, found {hsyst.axes.name}"
        )

    num_vars = hsyst.axes["var"].size
    variations = []
    names = []
    groups = []
    for ivar in range(num_vars):
        variation = hsyst[{"var": ivar}] * scaling_factor
        variation = rebin_mass(variation, mass_rebin)
        variation = restrict_mass_window(variation, mass_min, mass_max)
        if projection_axes is not None:
            variation = project_to_axes_with_mass(variation, projection_axes)
        variations.append(trim_variations(clip_values(variation, lim=args.clip), hist_mc, thresh))
        names.append(f"{name_prefix}_{ivar}")
        groups.append(name_prefix)

    return variations, names, groups


# set up the global systematics
nuisance_categories = ["A", "e", "M"]

print(args.systematicType)
writer = tensorwriter.TensorWriter(
    sparse=args.sparse,
    systematic_type=args.systematicType,
)
projection_axes = [] if args.massOnly else args.projectAxes

# initialize a global data covariance matrix (flattened for independent errors)
cov_data = np.array([])

# ### add D0

# D0_file = 'histsD0.pkl.lz4'
# with (lz4.frame.open(D0_file, "r")) as openfile:
#     resultdict = pickle.load(openfile)
# print(resultdict)
# hist_mc = resultdict["hD0_nom"]*0.16/2
# hist_mc_pt_up = resultdict["hD0_pt_up"]*0.16/2
# hist_mc_pt_down = resultdict["hD0_pt_down"]*0.16/2
# hist_mc_E_up = resultdict["hD0_E_up"]*0.16/2
# hist_mc_E_down = resultdict["hD0_E_down"]*0.16/2

# print(hist_mc)

# writer.add_channel(hist_mc.axes, 'D0')
# writer.add_process(hist_mc, f"sig_D0", 'D0', signal=False)
# writer.add_data(hist_mc, 'D0')

# # Axis 0 is etaK by how you loaded it (['etaK','mRK','D0mass'])
# eta_axis = hist_mc.axes[0]
# n_eta = eta_axis.size  # number of regular bins (no flow)

# nom = hist_mc.view(flow=False)
# A_up  = hist_mc_pt_up.view(flow=False)
# A_dn  = hist_mc_pt_down.view(flow=False)
# e_up  = hist_mc_E_up.view(flow=False)
# e_dn  = hist_mc_E_down.view(flow=False)

# variations_up = []
# variations_down = []

# # A variations
# for i in range(n_eta):
#     # --- UP: nominal everywhere, but eta-bin i from UP ---
#     h_up_i = hist_mc.copy()
#     v_up_i = h_up_i.view(flow=False)
#     v_up_i[i, :, :] = A_up[i, :, :]
#     variations_up.append(h_up_i)

#     # --- DOWN: nominal everywhere, but eta-bin i from DOWN ---
#     h_dn_i = hist_mc.copy()
#     v_dn_i = h_dn_i.view(flow=False)
#     v_dn_i[i, :, :] = A_dn[i, :, :]
#     variations_down.append(h_dn_i)

# # e variations
# for i in range(n_eta):
#     # --- UP: nominal everywhere, but eta-bin i from UP ---
#     h_up_i = hist_mc.copy()
#     v_up_i = h_up_i.view(flow=False)
#     v_up_i[i, :, :] = e_up[i, :, :]
#     variations_up.append(h_up_i)

#     # --- DOWN: nominal everywhere, but eta-bin i from DOWN ---
#     h_dn_i = hist_mc.copy()
#     v_dn_i = h_dn_i.view(flow=False)
#     v_dn_i[i, :, :] = e_dn[i, :, :]
#     variations_down.append(h_dn_i)

# for (up_var, down_var, var, group) in zip(variations_up, variations_down, variations, groups):
#     #print(up_var.name,var, group)
#     writer.add_systematic(
#                 [up_var, down_var],
#                 var,
#                 f"sig_D0",
#                 "D0",
#                 groups=[group],
#                 #symmetrize="none",
#                 constrained=True #should be True usually
#             )

jpsi_channels, input_meta_info = load_jpsi_channels_from_calinput(args.calinputHdf5)
for resultdict in jpsi_channels:
    sample = resultdict["name"]
    if sample not in args.jpsiChannels:
        print(f"skipping channel {sample}")
        continue
    print(f"processing {sample}")
    hist_mc = resultdict["JPsi_mc"]

    print(np.any(hist_mc.values() <= 0))
    hist_data = resultdict["JPsi_data"]
    scaling_factor = np.sum(hist_data.values()) / np.sum(hist_mc.values())
    signal_active_cells = populated_cells(hist_mc, event_thresh=args.mcEventThresh)
    print("signal active cells:", np.sum(signal_active_cells))
    print(scaling_factor, sample)
    hist_mc = hist_mc * scaling_factor
    hist_mc = rebin_mass(hist_mc, args.massRebin)
    hist_data = rebin_mass(hist_data, args.massRebin)
    hist_mc = restrict_mass_window(hist_mc, args.massMin, args.massMax)
    hist_data = restrict_mass_window(hist_data, args.massMin, args.massMax)

    if projection_axes is not None:
        hist_mc = project_to_axes_with_mass(hist_mc, projection_axes)
        hist_data = project_to_axes_with_mass(hist_data, projection_axes)

    signal_active_cells = populated_cells(hist_mc, event_thresh=args.mcEventThresh)
    if args.dataEventThresh > 0:
        data_active_cells = populated_cells(
            hist_data, event_thresh=args.dataEventThresh
        )
        signal_active_cells = signal_active_cells & data_active_cells
    if args.mcTailEventThresh > 0:
        tail_active_cells = cells_with_populated_mc_tails(
            hist_mc,
            event_thresh=args.mcTailEventThresh,
            tail_bins=args.mcTailBins,
            require=args.mcTailRequire,
        )
        print(
            "MC tail occupancy selection keeps "
            f"{np.count_nonzero(tail_active_cells)}/{tail_active_cells.size} "
            f"cells before other masks"
        )
        signal_active_cells = signal_active_cells & tail_active_cells
    print(
        f"Active {sample} cells after all selections: "
        f"{np.count_nonzero(signal_active_cells)}/{signal_active_cells.size}"
    )
    # fit_active_cells = signal_active_cells | populated_cells(hist_data)
    clip_values(hist_mc, lim=args.clip, variances=True)
    zero_inactive_cells(hist_mc, signal_active_cells)
    if args.mcEventThresh < 0:
        fit_active_cells = signal_active_cells | populated_cells(hist_data)
    else:
        fit_active_cells = signal_active_cells
        zero_inactive_cells(hist_data, fit_active_cells)

    # placing a hack here to tell me which 4d bins to make rabbit selections for
    # s = ""
    # for eta1 in range(24):
    #     for eta2 in range(24):
    #         for pt1 in range(4):
    #             for pt2 in range(4):
    #                 if fit_active_cells[eta1, eta2, pt1, pt2]:
    #                     s += f"-m Select JPsi_prompt eta1:{eta1},eta2:{eta2},pt1:{pt1},pt2:{pt2} "
    # print(s)
    # exit()
    up_variations = []
    down_variations = []
    thresh = 1e-5
    up_variations, down_variations, variations, groups = (
        split_calinput_scale_variations(
            resultdict["JPsi_syst"],
            hist_mc,
            scaling_factor,
            thresh,
            projection_axes=projection_axes,
            mass_rebin=args.massRebin,
            mass_min=args.massMin,
            mass_max=args.massMax,
        )
    )
    if resultdict.get("JPsi_resolution_syst") is not None:
        resolution_variations, resolution_names, resolution_groups = (
            split_calinput_resolution_variations(
                resultdict["JPsi_resolution_syst"],
                hist_mc,
                scaling_factor,
                thresh,
                projection_axes=projection_axes,
                mass_rebin=args.massRebin,
                mass_min=args.massMin,
                mass_max=args.massMax,
            )
        )
    else:
        resolution_variations, resolution_names, resolution_groups = [], [], []
    if resultdict.get("JPsi_pixel_syst") is not None:
        pixel_variations, pixel_names, pixel_groups = (
            split_calinput_pixel_multiplicity_variations(
                resultdict["JPsi_pixel_syst"],
                hist_mc,
                scaling_factor,
                thresh,
                "pixel_multiplicity_syst",
                projection_axes=projection_axes,
                mass_rebin=args.massRebin,
                mass_min=args.massMin,
                mass_max=args.massMax,
            )
        )
    else:
        pixel_variations, pixel_names, pixel_groups = [], [], []
    if resultdict.get("JPsi_pixel_stat") is not None:
        pixel_stat_variations, pixel_stat_names, pixel_stat_groups = (
            split_calinput_pixel_multiplicity_variations(
                resultdict["JPsi_pixel_stat"],
                hist_mc,
                scaling_factor,
                thresh,
                "pixel_multiplicity_stat",
                projection_axes=projection_axes,
                mass_rebin=args.massRebin,
                mass_min=args.massMin,
                mass_max=args.massMax,
            )
        )
    else:
        pixel_stat_variations, pixel_stat_names, pixel_stat_groups = [], [], []

    if args.jpsiTailMorph:
        tail_morph_variations = [
            make_mass_stretch_variation(
                hist_mc,
                -args.jpsiTailMorphStrength,
                center=args.jpsiTailMorphCenter,
            ),
            make_mass_stretch_variation(
                hist_mc,
                args.jpsiTailMorphStrength,
                center=args.jpsiTailMorphCenter,
            ),
        ]
    else:
        tail_morph_variations = None

    writer.add_channel(hist_mc.axes, sample)
    writer.add_process(
        hist_mc, f"sig_{sample}", sample, signal=True
    )  # singal=True means cross sections float
    background_hist = make_active_background(hist_mc, fit_active_cells)
    writer.add_process(
        background_hist,
        f"background_{sample}",
        sample,
        signal=False,
        variances=np.zeros_like(background_hist.values()),
    )
    writer.add_data(hist_data, sample)
    # writer.add_norm_systematic("norm", [f"sig_{sample}"], sample, 1.02)

    # writer.add_lnN_systematic("luminosity", [f"sig_{sample}"], sample, 1.05)

    for up_var, down_var, var, group in zip(
        up_variations, down_variations, variations, groups
    ):
        print(var)
        writer.add_systematic(
            [up_var, down_var],
            var,
            f"sig_{sample}",
            sample,
            groups=[group],
            # symmetrize="none",
            constrained=True,  # should be True usually
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

    for variation, var, group in zip(pixel_variations, pixel_names, pixel_groups):
        print(var)
        writer.add_systematic(
            variation,
            var,
            f"sig_{sample}",
            sample,
            groups=[group],
            constrained=True,
        )

    for variation, var, group in zip(
        pixel_stat_variations, pixel_stat_names, pixel_stat_groups
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

    if tail_morph_variations is not None:
        var = "jpsi_tail_morph"
        print(var)
        writer.add_systematic(
            tail_morph_variations,
            var,
            f"sig_{sample}",
            sample,
            groups=[var],
            constrained=True,
        )

directory = args.output
if directory == "":
    directory = "./"
filename = args.outname
if args.postfix:
    filename += f"_{args.postfix}"

# Propagate provenance into the tensor: this script's command in meta_info and
# the upstream dimuon_resonances_calinput.py command (read from the input file)
# in meta_info_input, mirroring setupRabbit.py and d0_tensor.py.
meta = {
    "meta_info": output_tools.make_meta_info_dict(args=args, wd=common.base_dir),
    "meta_info_input": input_meta_info,
}
writer.write(outfolder=directory, outfilename=filename, meta_data_dict=meta)
