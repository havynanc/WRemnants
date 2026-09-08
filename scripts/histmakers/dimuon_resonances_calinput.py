import os

import hist
import numpy as np
import ROOT

import narf
from wremnants.production import muon_calibration
from wremnants.production.histmaker_tools import write_analysis_output
from wremnants.utilities import common, parsing
from wums import logging

analysis_label = common.analysis_label(os.path.basename(__file__))


local_resonance_files = {
    "jpsi": {
        "data": ["/scratch/submit/cms/emanca/jpsicor_data.root"],
        "mc": [
            "/scratch/submit/cms/emanca/jpsicor_mc.root",
            "/scratch/submit/cms/emanca/jcor_mc_0to8.root",
        ],
    },
    "upsilon": {
        "data": ["/scratch/submit/cms/emanca/upsilon_data.root"],
        "mc": [
            "/scratch/submit/cms/emanca/upsilon_0to8.root",
            "/scratch/submit/cms/emanca/upsilon_8to13.root",
            "/scratch/submit/cms/emanca/upsilon_13toInf.root",
        ],
    },
}

trigger_channels = {
    "jpsi": [
        {
            "label": "dimuon20_jpsi",
            "cut": "HLT_Dimuon20_Jpsi",
            "mc": local_resonance_files["jpsi"]["mc"],
            "layer_corrected": True,
        },
        {
            "label": "doublemu4_jpsitrk_displaced",
            "cut": "HLT_DoubleMu4_JpsiTrk_Displaced",
            "mc": ["/scratch/submit/cms/emanca/BuToJpsiK_BMuonFilter_v2_BPH.root"],
            "layer_corrected": False,
        },
    ],
    "upsilon": [
        {
            "label": "inclusive",
            "cut": "1.",
            "mc": local_resonance_files["upsilon"]["mc"],
            "layer_corrected": True,
        },
    ],
}


parser, initargs = parsing.common_parser(analysis_label)
parser.add_argument(
    "--resonance",
    type=str,
    choices=["jpsi", "upsilon"],
    help="Resonance for selection",
    default="upsilon",
)
parser.add_argument(
    "--triggers",
    nargs="+",
    default=None,
    help="Triggers to process for the resonance (each as a separate channel)",
)
parser.add_argument(
    "--etaBins",
    type=int,
    default=None,
    help="Override the number of bins for each reconstructed muon eta axis",
)
parser.add_argument(
    "--ptBinEdges",
    type=float,
    nargs="+",
    default=None,
    help="Override the default pt binning for the selected resonance",
)
parser.add_argument(
    "--massBinEdges",
    type=float,
    nargs="+",
    default=None,
    help="Override the default mass binning for the selected resonance",
)
parser.add_argument(
    "--applyAeMtoData",
    type=str,
    default=None,
    help="Path to a rabbit fitresult HDF5 file. The fitted AeM parameters are "
    "extracted and used to correct the data muon pts "
    "(via muon_calibration.define_AeM_data_corrections) before filling the "
    "data histograms. MC histograms are left unchanged.",
)
parser.add_argument(
    "--ptBins",
    type=int,
    choices=[2, 3],
    default=None,
    help=(
        "Override the number of bins for each reconstructed muon pt axis. "
        "Currently 2 and 3 are implemented for J/psi fit studies."
    ),
)
parser.add_argument(
    "--quantile4D",
    action="store_true",
    help=(
        "Fill eta1, eta2, pt1, and pt2 as data-derived chained quantile-bin "
        "indices instead of physical bins. The mass axis remains unchanged."
    ),
)
parser.add_argument(
    "--quantile5D",
    action="store_true",
    help=(
        "Fill eta1, eta2, pt1, pt2, and mass as data-derived chained "
        "quantile-bin indices instead of physical bins. This is a fit stress "
        "test: the mass axis is a quantile index, not physical mass."
    ),
)
parser.add_argument(
    "--quantileMass",
    action="store_true",
    help=(
        "Fill only mass as a data-derived quantile-bin index. The kinematic "
        "axes remain physical bins."
    ),
)
parser.add_argument(
    "--pixelHitSelection",
    choices=["all", "bothPixelHits", "anyMissingPixelHits"],
    default="all",
    help=(
        "Optional selection on the number of valid pixel hits in the track refit. "
        "Use bothPixelHits for both muons passing --minValidPixelHits; "
        "use anyMissingPixelHits for the complementary category."
    ),
)
parser.add_argument(
    "--minValidPixelHits",
    type=int,
    default=1,
    help=(
        "Minimum valid pixel hits required per muon when "
        "--pixelHitSelection bothPixelHits is used."
    ),
)
parser.add_argument(
    "--deltaPhiSign",
    choices=["all", "positive", "negative"],
    default="all",
    help=(
        "Optional selection on the signed dimuon azimuthal separation, defined "
        "as TVector2::Phi_mpi_pi(phi_plus - phi_minus). Use negative for "
        "DeltaPhi < 0 and positive for DeltaPhi > 0."
    ),
)
parser.add_argument(
    "--resolutionPrefitUncertainty",
    type=float,
    default=0.3,
    help=(
        "Relative prefit uncertainty assigned to each fitted muon-resolution "
        "parameter when --fitMuonScaleAndResolution is used."
    ),
)
parser.add_argument(
    "--resolutionPrefitUncertaintyA",
    type=float,
    default=None,
    help=(
        "Override the relative prefit uncertainty for only the fitted "
        "muon-resolution a parameter. If omitted, --resolutionPrefitUncertainty "
        "is used for a, b, c, and d."
    ),
)
parser = parsing.set_parser_default(parser, "theoryCorr", [])
parser = parsing.set_parser_default(parser, "scale_A", 5.0)
parser = parsing.set_parser_default(parser, "scale_e", 5.0)

args = parser.parse_args()

if args.etaBins is not None and args.etaBins <= 0:
    raise ValueError("--etaBins must be a positive integer")
if args.ptBins is not None and args.ptBins <= 0:
    raise ValueError("--ptBins must be a positive integer")
if args.minValidPixelHits < 1:
    raise ValueError("--minValidPixelHits must be at least 1")
if args.resolutionPrefitUncertainty <= 0:
    raise ValueError("--resolutionPrefitUncertainty must be positive")
if (
    args.resolutionPrefitUncertaintyA is not None
    and args.resolutionPrefitUncertaintyA <= 0
):
    raise ValueError("--resolutionPrefitUncertaintyA must be positive")
if sum([args.quantile4D, args.quantile5D, args.quantileMass]) > 1:
    raise ValueError(
        "--quantile4D, --quantile5D, and --quantileMass are mutually exclusive"
    )
if args.etaBins is not None and not args.fitMuonScaleAndResolution:
    raise ValueError(
        "--etaBins currently requires --fitMuonScaleAndResolution so scale "
        "uncertainties can be correlated across the requested eta groups"
    )
possible_triggers = [trig["label"] for trig in trigger_channels[args.resonance]]
selected_triggers = possible_triggers if args.triggers is None else args.triggers
if not all(trig in possible_triggers for trig in selected_triggers):
    raise ValueError(
        f"--triggers must contain only valid triggers for the "
        f"selected resonance ({possible_triggers} for resonance "
        f"{args.resonance}), or be None to use all valid triggers"
    )

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)


def read_AeM_corrections(fitresult_path, n_eta_bins=24):
    import rabbit.io_tools

    fitresult = rabbit.io_tools.get_fitresult(fitresult_path)
    h_parms = fitresult["parms"].get()
    labels = [str(label) for label in h_parms.axes["parms"]]
    values = h_parms.values()
    parm_map = dict(zip(labels, values))

    def collect(prefix):
        out = []
        for i in range(n_eta_bins):
            key = f"{prefix}{i}"
            if key not in parm_map:
                raise ValueError(
                    f"Parameter '{key}' not found in fitresult '{fitresult_path}'. "
                    f"Available parameters: {labels}"
                )
            out.append(float(parm_map[key]))
        return out

    return collect("A"), collect("e"), collect("M")


AeM_corrections = None
if args.applyAeMtoData is not None:
    AeM_corrections = read_AeM_corrections(args.applyAeMtoData)
    logger.info(
        f"Loaded A/e/M data pt corrections from fitresult {args.applyAeMtoData}"
    )

if args.muonScaleVariation not in ["onnxReweight", "smearingWeightsSplines"]:
    raise ValueError(
        "dimuon_resonances_calinput.py currently supports "
        "--muonScaleVariation onnxReweight or smearingWeightsSplines"
    )

calib_filepaths = common.calib_filepaths
scale_diff_weights_helper = (
    ROOT.wrem.SplinesDifferentialWeightsHelper(calib_filepaths["tflite_file"])
    if args.muonScaleVariation == "smearingWeightsSplines"
    else None
)
resolution_diff_weights_helper = (
    ROOT.wrem.SplinesDifferentialWeightsHelper(
        calib_filepaths["tflite_file_nosmearing"]
    )
    if args.muonScaleVariation == "smearingWeightsSplines"
    else None
)
(
    _mc_jpsi_crctn_helper,
    _data_jpsi_crctn_helper,
    _mc_jpsi_crctn_unc_helper,
    data_jpsi_crctn_unc_helper,
) = muon_calibration.make_jpsi_crctn_helpers(
    calib_filepaths,
    muon_corr_mc=args.muonCorrMC,
    muon_corr_data=args.muonCorrData,
    scale_var_method=args.muonScaleVariation,
    scale_A=args.scale_A,
    scale_e=args.scale_e,
    scale_M=args.scale_M,
    make_uncertainty_helper=True,
    include_covariance=not args.fitMuonScaleAndResolution,
    smearing=not args.noSmearing,
    fit_muon_scale=args.fitMuonScaleAndResolution,
    variation_eta_bins=args.etaBins,
    # Use mass-aware propagation by default for ONNX reweighting; non-ONNX
    # methods receive None because their helpers do not consume reweight_mass.
    reweight_mass=(
        [0.1056583745, 0.1056583745]
        if args.muonScaleVariation == "onnxReweight"
        else None
    ),
)

if data_jpsi_crctn_unc_helper is None:
    raise ValueError(
        "Muon scale systematics require --muonCorrData massfit or lbl_massfit"
    )

_smearing_helper, smearing_uncertainty_helper = (
    (None, None)
    if args.noSmearing
    else muon_calibration.make_muon_smearing_helpers(
        scale_var_method=args.muonScaleVariation,
        parameter_variations=True,
        fit_muon_resolution=args.fitMuonScaleAndResolution,
        variation_eta_bins=args.etaBins,
        resolution_prefit_uncertainties=[
            args.resolutionPrefitUncertaintyA or args.resolutionPrefitUncertainty,
            args.resolutionPrefitUncertainty,
            args.resolutionPrefitUncertainty,
            args.resolutionPrefitUncertainty,
        ],
        resolution_prefit_uncertainties_mode="relative",
    )
)

(
    _pixel_multiplicity_helper,
    pixel_multiplicity_uncertainty_helper,
    pixel_multiplicity_uncertainty_helper_stat,
) = muon_calibration.make_pixel_multiplicity_helpers()


def limited_files(files):
    if args.maxFiles is not None and args.maxFiles > 0:
        return files[: args.maxFiles]
    return files


def bool_filter(expression):
    return f"static_cast<bool>({expression})"


datasets = []
dataset_channels = {}
for channel in trigger_channels[args.resonance]:
    if channel["label"] not in selected_triggers:
        continue
    for sample_type in ["data", "mc"]:
        dataset_name = f"{args.resonance}_{sample_type}_{channel['label']}"
        dataset_channels[dataset_name] = channel
        datasets.append(
            narf.Dataset(
                name=dataset_name,
                filepaths=limited_files(
                    local_resonance_files[args.resonance]["data"]
                    if sample_type == "data"
                    else channel["mc"]
                ),
                is_data=sample_type == "data",
                xsec=None if sample_type == "data" else 1.0,
                group="Data" if sample_type == "data" else args.resonance.upper(),
            )
        )


resonance_options = {
    "jpsi": {
        "name": "JPsi",
        "default_eta_bins": 24,
        "eta_range": (-2.4, 2.4),
        "pt1_axis": hist.axis.Variable([4.2, 7.0, 10.5, 15.0, 25.0], name="pt1"),
        "pt2_axis": hist.axis.Variable([4.2, 7.0, 10.5, 15.0, 25.0], name="pt2"),
        "mass_axis": hist.axis.Variable(
            [
                2.92,
                2.965,
                3.000,
                3.030,
                3.050,
                3.065,
                3.077,
                3.087,
                3.091,
                3.095,
                3.101,
                3.106,
                3.111,
                3.116,
                3.121,
                3.126,
                3.132,
                3.139,
                3.148,
                3.159,
                3.173,
                3.190,
                3.210,
                3.233,
                3.258,
                3.28,
            ],
            name="mass",
        ),
        "mass_range": (2.8, 3.35),
    },
    "upsilon": {
        "name": "Y",
        "default_eta_bins": 8,
        "eta_range": (-0.8, 0.8),
        "pt1_axis": hist.axis.Variable([4.2, 6.0, 7.9, 10.3, 25.0], name="pt1"),
        "pt2_axis": hist.axis.Variable([4.2, 6.0, 7.9, 10.3, 25.0], name="pt2"),
        "mass_axis": hist.axis.Regular(25, 9.0, 9.7, name="mass"),
        "mass_range": (8.8, 9.6),
    },
}

cfg = resonance_options[args.resonance]
eta_bins = args.etaBins or cfg["default_eta_bins"]
eta_min, eta_max = cfg["eta_range"]
mass_axis = cfg["mass_axis"]

if args.ptBins == 2:
    if args.resonance == "jpsi":
        pt1_axis = hist.axis.Variable([4.2, 10.5, 25.0], name="pt1")
        pt2_axis = hist.axis.Variable([4.2, 10.5, 25.0], name="pt2")
    else:
        pt1_axis = hist.axis.Variable([4.2, 7.9, 25.0], name="pt1")
        pt2_axis = hist.axis.Variable([4.2, 7.9, 25.0], name="pt2")
elif args.ptBins == 3:
    if args.resonance == "jpsi":
        pt1_axis = hist.axis.Variable([4.2, 7.0, 10.5, 25.0], name="pt1")
        pt2_axis = hist.axis.Variable([4.2, 7.0, 10.5, 25.0], name="pt2")
    else:
        pt1_axis = hist.axis.Variable([4.2, 7.9, 10.3, 25.0], name="pt1")
        pt2_axis = hist.axis.Variable([4.2, 7.9, 10.3, 25.0], name="pt2")
else:
    pt1_axis = cfg["pt1_axis"]
    pt2_axis = cfg["pt2_axis"]

calibration_axes = [
    hist.axis.Regular(eta_bins, eta_min, eta_max, name="eta1"),
    hist.axis.Regular(eta_bins, eta_min, eta_max, name="eta2"),
    (
        pt1_axis
        if args.ptBinEdges is None
        else hist.axis.Variable(args.ptBinEdges, name="pt1")
    ),
    (
        pt2_axis
        if args.ptBinEdges is None
        else hist.axis.Variable(args.ptBinEdges, name="pt2")
    ),
    (
        mass_axis
        if args.massBinEdges is None
        else hist.axis.Variable(args.massBinEdges, name="mass")
    ),
]

quantile_kinematic_axes = [
    hist.axis.Regular(eta_bins, 0.0, 1.0, name="eta1", underflow=False, overflow=False),
    hist.axis.Regular(eta_bins, 0.0, 1.0, name="eta2", underflow=False, overflow=False),
    hist.axis.Regular(
        pt1_axis.size, 0.0, 1.0, name="pt1", underflow=False, overflow=False
    ),
    hist.axis.Regular(
        pt2_axis.size, 0.0, 1.0, name="pt2", underflow=False, overflow=False
    ),
]
quantile_dummy_axis = hist.axis.Integer(
    0,
    1,
    name="quantile4d_dummy",
    underflow=False,
    overflow=False,
)
quantile_5d_dummy_axis = hist.axis.Integer(
    0,
    1,
    name="quantile5d_dummy",
    underflow=False,
    overflow=False,
)
quantile_mass_dummy_axis = hist.axis.Integer(
    0,
    1,
    name="quantilemass_dummy",
    underflow=False,
    overflow=False,
)
quantile_calibration_axes = [
    hist.axis.Integer(0, eta_bins, name="eta1", underflow=False, overflow=False),
    hist.axis.Integer(0, eta_bins, name="eta2", underflow=False, overflow=False),
    hist.axis.Integer(0, pt1_axis.size, name="pt1", underflow=False, overflow=False),
    hist.axis.Integer(0, pt2_axis.size, name="pt2", underflow=False, overflow=False),
    mass_axis,
]
quantile_5d_axes = [
    *quantile_kinematic_axes,
    hist.axis.Regular(
        mass_axis.size, 0.0, 1.0, name="mass", underflow=False, overflow=False
    ),
]
quantile_5d_calibration_axes = [
    hist.axis.Integer(0, eta_bins, name="eta1", underflow=False, overflow=False),
    hist.axis.Integer(0, eta_bins, name="eta2", underflow=False, overflow=False),
    hist.axis.Integer(0, pt1_axis.size, name="pt1", underflow=False, overflow=False),
    hist.axis.Integer(0, pt2_axis.size, name="pt2", underflow=False, overflow=False),
    hist.axis.Integer(
        0,
        mass_axis.size,
        name="mass",
        underflow=False,
        overflow=False,
        metadata={
            "physical_centers": np.asarray(mass_axis.centers, dtype=float),
            "physical_widths": np.asarray(mass_axis.widths, dtype=float),
            "physical_axis_name": "mass",
        },
    ),
]
quantile_mass_axis = hist.axis.Regular(
    mass_axis.size, 0.0, 1.0, name="mass", underflow=False, overflow=False
)
quantile_mass_calibration_axes = [
    hist.axis.Regular(eta_bins, eta_min, eta_max, name="eta1"),
    hist.axis.Regular(eta_bins, eta_min, eta_max, name="eta2"),
    pt1_axis,
    pt2_axis,
    hist.axis.Integer(
        0,
        mass_axis.size,
        name="mass",
        underflow=False,
        overflow=False,
        metadata={
            "physical_centers": np.asarray(mass_axis.centers, dtype=float),
            "physical_widths": np.asarray(mass_axis.widths, dtype=float),
            "physical_axis_name": "mass",
        },
    ),
]
quantile_helpers_by_channel = {}
quantile_output_axes_by_channel = {}


def quantile_mass_output_axes_from_hists(centers_hist, volume_hist):
    centers = np.asarray(centers_hist.values()[..., 0], dtype=float)
    widths = np.asarray(volume_hist.values(), dtype=float)
    expected_shape = (eta_bins, eta_bins, pt1_axis.size, pt2_axis.size, mass_axis.size)
    if centers.shape != expected_shape:
        raise RuntimeError(
            "Unexpected conditional mass-quantile center shape "
            f"{centers.shape}, expected {expected_shape}"
        )
    if widths.shape != expected_shape:
        raise RuntimeError(
            "Unexpected conditional mass-quantile volume shape "
            f"{widths.shape}, expected {expected_shape}"
        )
    return [
        hist.axis.Regular(eta_bins, eta_min, eta_max, name="eta1"),
        hist.axis.Regular(eta_bins, eta_min, eta_max, name="eta2"),
        pt1_axis,
        pt2_axis,
        hist.axis.Integer(
            0,
            mass_axis.size,
            name="mass",
            underflow=False,
            overflow=False,
            metadata={
                "physical_centers_nd": centers,
                "physical_widths_nd": widths,
                "physical_axis_names": ["eta1", "eta2", "pt1", "pt2", "mass"],
                "physical_axis_name": "mass",
                "quantile_volume_corrected": True,
            },
        ),
    ]


def calibration_axes_and_cols(df, dataset, channel, cols):
    if not (args.quantile4D or args.quantile5D or args.quantileMass):
        return (
            df,
            calibration_axes,
            [
                cols["plus_eta"],
                cols["minus_eta"],
                cols["plus_pt"],
                cols["minus_pt"],
                cols["mass"],
            ],
        )

    channel_label = channel["label"]
    if args.quantile5D:
        quantile_label = "calib5d"
    elif args.quantileMass:
        quantile_label = "calibmass"
    else:
        quantile_label = "calib4d"
    dummy_col = (
        "quantile5d_dummy"
        if args.quantile5D
        else "quantilemass_dummy" if args.quantileMass else "quantile4d_dummy"
    )
    dummy_axis = (
        quantile_5d_dummy_axis
        if args.quantile5D
        else quantile_mass_dummy_axis if args.quantileMass else quantile_dummy_axis
    )
    df = df.DefinePerSample(dummy_col, "0")
    quantile_input_cols = [
        dummy_col,
        cols["plus_eta"],
        cols["minus_eta"],
        cols["plus_pt"],
        cols["minus_pt"],
    ]
    quantile_axes = quantile_kinematic_axes
    output_axes = quantile_calibration_axes
    if args.quantile5D:
        quantile_input_cols.append(cols["mass"])
        quantile_axes = quantile_5d_axes
        output_axes = quantile_5d_calibration_axes
    elif args.quantileMass:
        quantile_input_cols = [
            cols["plus_eta"],
            cols["minus_eta"],
            cols["plus_pt"],
            cols["minus_pt"],
            cols["mass"],
        ]
        quantile_axes = [quantile_mass_axis]
        output_axes = quantile_mass_calibration_axes
    quantile_helpers = quantile_helpers_by_channel.get(channel_label)
    stored_output_axes = quantile_output_axes_by_channel.get(channel_label)

    if dataset.is_data:
        quantile_helpers, centers_hist, volume_hist = (
            narf.histutils.build_quantile_hists(
                df,
                quantile_input_cols,
                condaxes=calibration_axes[:-1] if args.quantileMass else [dummy_axis],
                quantaxes=quantile_axes,
            )
        )
        quantile_helpers_by_channel[channel_label] = quantile_helpers
        if args.quantileMass:
            output_axes = quantile_mass_output_axes_from_hists(
                centers_hist, volume_hist
            )
            quantile_output_axes_by_channel[channel_label] = output_axes
    elif quantile_helpers is None:
        raise RuntimeError(
            f"Missing data-derived quantile helpers for channel {channel_label}. "
            "Datasets must be processed with data before MC."
        )
    elif stored_output_axes is not None:
        output_axes = stored_output_axes

    df, _quantile_axes, quantile_cols = narf.histutils.define_quantiles(
        df,
        cols=quantile_input_cols,
        quantile_hists=quantile_helpers,
        label=quantile_label,
    )
    if args.quantile5D:
        return df, output_axes, quantile_cols[1:]
    if args.quantileMass:
        return (
            df,
            output_axes,
            [
                cols["plus_eta"],
                cols["minus_eta"],
                cols["plus_pt"],
                cols["minus_pt"],
                quantile_cols[-1],
            ],
        )
    return df, output_axes, [*quantile_cols[1:], cols["mass"]]


def reco_columns(channel):
    suffix = "cor" if channel["layer_corrected"] else ""
    return {
        "plus_pt": f"Muplus{suffix}_pt",
        "minus_pt": f"Muminus{suffix}_pt",
        "plus_eta": f"Muplus{suffix}_eta",
        "minus_eta": f"Muminus{suffix}_eta",
        "plus_phi": f"Muplus{suffix}_phi",
        "minus_phi": f"Muminus{suffix}_phi",
        "mass": f"Jpsi{suffix}_mass",
    }


def reco_selection(cols):
    mass_min, mass_max = cfg["mass_range"]
    return (
        f"{cols['plus_pt']} > 1.0 && {cols['minus_pt']} > 1.0 && "
        f"{cols['plus_pt']} < 100.0 && {cols['minus_pt']} < 100.0 && "
        f"std::fabs({cols['plus_eta']}) < {eta_max} && "
        f"std::fabs({cols['minus_eta']}) < {eta_max} && "
        f"{cols['mass']} > {mass_min} && {cols['mass']} < {mass_max}"
    )


def pixel_hit_selection():
    if args.pixelHitSelection == "all":
        return "1."
    if args.pixelHitSelection == "bothPixelHits":
        return (
            f"Muplus_nvalidpixel >= {args.minValidPixelHits} && "
            f"Muminus_nvalidpixel >= {args.minValidPixelHits}"
        )
    if args.pixelHitSelection == "anyMissingPixelHits":
        return (
            f"Muplus_nvalidpixel < {args.minValidPixelHits} || "
            f"Muminus_nvalidpixel < {args.minValidPixelHits}"
        )
    raise ValueError(f"Unsupported --pixelHitSelection {args.pixelHitSelection}")


def delta_phi_selection(cols):
    if args.deltaPhiSign == "all":
        return "1."

    comparison = "<" if args.deltaPhiSign == "negative" else ">"
    return (
        f"TVector2::Phi_mpi_pi({cols['plus_phi']} - {cols['minus_phi']}) "
        f"{comparison} 0.0"
    )


def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")

    results = []
    channel = dataset_channels[dataset.name]
    reco_cols = reco_columns(channel)

    if dataset.is_data and AeM_corrections is not None:
        if not channel["layer_corrected"]:
            raise ValueError(
                "--applyAeMtoData operates on the layer-corrected data columns "
                f"(Mupluscor_*), but data channel '{channel['label']}' is not "
                "layer corrected"
            )
        A_vals, e_vals, M_vals = AeM_corrections
        df = muon_calibration.define_AeM_data_corrections(df, A_vals, e_vals, M_vals)
        reco_cols["plus_pt"] = "Mupluscor_AeM_pt"
        reco_cols["minus_pt"] = "Muminuscor_AeM_pt"
        reco_cols["mass"] = "Jpsicor_AeM_mass"

    calibration_cols = [
        reco_cols["plus_eta"],
        reco_cols["minus_eta"],
        reco_cols["plus_pt"],
        reco_cols["minus_pt"],
        reco_cols["mass"],
    ]

    df = df.DefinePerSample("weight", "1.0")
    weightsum = df.SumAndCount("weight")

    df = df.Filter(reco_selection(reco_cols))
    df = df.Filter(bool_filter(pixel_hit_selection()))
    df = df.Filter(bool_filter(delta_phi_selection(reco_cols)))
    if not dataset.is_data:
        df = df.Filter(
            "Jpsigen_mass > 0.0 && Muplusgen_pt > 0.0 && Muminusgen_pt > 0.0"
        )
    df = df.Filter(bool_filter(channel["cut"]))

    df, hist_axes, calibration_cols = calibration_axes_and_cols(
        df,
        dataset,
        channel,
        reco_cols,
    )

    pixel_multiplicity_cols = [
        "pixel_triggerCat",
        "pixel_eta",
        "pixel_pt",
        "pixel_charge",
        "pixel_nvalidpixel",
    ]
    if not dataset.is_data:
        df = (
            df
            # The calInput ntuples have no per-leg trigger matching; both selected
            # paths are dimuon triggers, so both candidates use the triggering map.
            .DefinePerSample(
                "pixel_triggerCat",
                "ROOT::VecOps::RVec<wrem::TriggerCat>{wrem::TriggerCat::triggering, wrem::TriggerCat::triggering}",
            )
            .Define(
                "pixel_eta",
                f"ROOT::VecOps::RVec<float>{{float({reco_cols['plus_eta']}), float({reco_cols['minus_eta']})}}",
            )
            .Define(
                "pixel_pt",
                f"ROOT::VecOps::RVec<float>{{float({reco_cols['plus_pt']}), float({reco_cols['minus_pt']})}}",
            )
            .Define("pixel_charge", "ROOT::VecOps::RVec<int>{1, -1}")
            .Define(
                "pixel_nvalidpixel",
                "ROOT::VecOps::RVec<int>{int(Muplus_nvalidpixel), int(Muminus_nvalidpixel)}",
            )
        )

    hist_name = f"{cfg['name']}_{'data' if dataset.is_data else 'mc'}"
    results.append(
        df.HistoBoost(
            hist_name,
            hist_axes,
            [*calibration_cols, "weight"],
        )
    )

    if not dataset.is_data:
        df = (
            df.Define("nominal_weight", "weight")
            .Define(
                "scale_recoPt",
                f"ROOT::VecOps::RVec<float>{{float({reco_cols['plus_pt']}), float({reco_cols['minus_pt']})}}",
            )
            .Define(
                "scale_recoEta",
                f"ROOT::VecOps::RVec<float>{{float({reco_cols['plus_eta']}), float({reco_cols['minus_eta']})}}",
            )
            .Define(
                "scale_recoPhi",
                f"ROOT::VecOps::RVec<float>{{float({reco_cols['plus_phi']}), float({reco_cols['minus_phi']})}}",
            )
            .Define("scale_recoCharge", "ROOT::VecOps::RVec<int>{1, -1}")
            .Define(
                "scale_genPt",
                "ROOT::VecOps::RVec<float>{float(Muplusgen_pt), float(Muminusgen_pt)}",
            )
            .Define(
                "scale_genEta",
                "ROOT::VecOps::RVec<float>{float(Muplusgen_eta), float(Muminusgen_eta)}",
            )
            .Define(
                "scale_genPhi",
                "ROOT::VecOps::RVec<float>{float(Muplusgen_phi), float(Muminusgen_phi)}",
            )
            .Define("scale_genCharge", "ROOT::VecOps::RVec<int>{1, -1}")
            .Define("scale_muon_source", "ROOT::VecOps::RVec<int>{443, 443}")
        )

        input_kinematics = [
            "scale_recoPt",
            "scale_recoEta",
            "scale_recoCharge",
            "scale_genPt",
            "scale_genEta",
            "scale_genCharge",
        ]
        if scale_diff_weights_helper is not None:
            df = df.Define(
                "scale_response_weight",
                scale_diff_weights_helper,
                input_kinematics,
            )
        if resolution_diff_weights_helper is not None:
            df = df.Define(
                "scale_resolution_response_weight",
                resolution_diff_weights_helper,
                input_kinematics,
            )

        df, scale_cols = muon_calibration.muon_reweight_helper_cols(
            df,
            data_jpsi_crctn_unc_helper,
            "scale",
            "scale_response_weight",
        )
        df = df.Define(
            "nominal_muonScaleSyst_responseWeights_tensor",
            data_jpsi_crctn_unc_helper,
            [*scale_cols, "nominal_weight"],
        )
        results.append(
            df.HistoBoost(
                "nominal_muonScaleSyst_responseWeights",
                hist_axes,
                [*calibration_cols, "nominal_muonScaleSyst_responseWeights_tensor"],
                tensor_axes=data_jpsi_crctn_unc_helper.tensor_axes,
                storage=hist.storage.Double(),
            )
        )

        df = muon_calibration.add_resolution_uncertainty(
            df,
            hist_axes,
            results,
            calibration_cols,
            smearing_uncertainty_helper,
            "scale",
            storage_type=hist.storage.Double(),
            response_weight_col="scale_resolution_response_weight",
        )

        df = df.Define(
            "nominal_pixelMultiplicitySyst_tensor",
            pixel_multiplicity_uncertainty_helper,
            [*pixel_multiplicity_cols, "nominal_weight"],
        )
        results.append(
            df.HistoBoost(
                "nominal_pixelMultiplicitySyst",
                hist_axes,
                [*calibration_cols, "nominal_pixelMultiplicitySyst_tensor"],
                tensor_axes=pixel_multiplicity_uncertainty_helper.tensor_axes,
                storage=hist.storage.Double(),
            )
        )

        if args.pixelMultiplicityStat:
            df = df.Define(
                "nominal_pixelMultiplicityStat_tensor",
                pixel_multiplicity_uncertainty_helper_stat,
                [*pixel_multiplicity_cols, "nominal_weight"],
            )
            results.append(
                df.HistoBoost(
                    "nominal_pixelMultiplicityStat",
                    hist_axes,
                    [*calibration_cols, "nominal_pixelMultiplicityStat_tensor"],
                    tensor_axes=pixel_multiplicity_uncertainty_helper_stat.tensor_axes,
                    storage=hist.storage.Double(),
                )
            )

    return results, weightsum


resultdict = narf.build_and_run(datasets, build_graph, event_tree="tree")

fout = f"{os.path.basename(__file__).replace('py', 'hdf5')}"
name_append = [args.resonance, args.era]
if args.etaBins is not None:
    name_append.append(f"etaBins_{args.etaBins}")
if args.ptBins is not None:
    name_append.append(f"ptBins_{args.ptBins}")
if args.quantile4D:
    name_append.append("quantile4D")
if args.quantile5D:
    name_append.append("quantile5D")
if args.quantileMass:
    name_append.append("quantileMass")
if args.deltaPhiSign != "all":
    name_append.append(f"deltaPhi_{args.deltaPhiSign}")
write_analysis_output(resultdict, fout, args, name_append=name_append)
