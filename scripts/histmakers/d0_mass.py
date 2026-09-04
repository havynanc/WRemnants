import glob
import os

import hist
import ROOT

import narf
from wremnants.production import muon_calibration
from wremnants.production.d0_axes import make_d0_template_axes
from wremnants.production.histmaker_tools import write_analysis_output
from wremnants.utilities import common, parsing
from wums import logging

analysis_label = common.analysis_label(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)
parser.add_argument(
    "--dataFile",
    default="/scratch/submit/cms/emanca/DstMiniAODTransientEmulationCVHShared_v1",
    help="Input ROOT file for data",
)
parser.add_argument(
    "--mcFile",
    nargs="+",
    default=[
        "/scratch/submit/cms/emanca/DStarTransientCVHTruth_v1",
        "/scratch/submit/cms/emanca/DStarTransientCVHTruth_prod500M_v1",
    ],
    help="One or more input ROOT files, globs, or directories for MC",
)
parser.add_argument(
    "--treeName",
    default=None,
    help=(
        "Override the input tree path for both data and MC. The selected trees must "
        "still provide either the native or supported legacy D0 branch schema."
    ),
)
parser.add_argument(
    "--dataTreeName",
    default="dstToD0PiProducer/Events",
    help="Input tree name for data",
)
parser.add_argument(
    "--mcTreeName",
    default="dstToD0PiMCTruthProducer/MatchedCandidates",
    help="Input tree name for MC",
)
parser.add_argument(
    "--mcSelection",
    choices=["truthMatched", "dataLike"],
    default="truthMatched",
    help=(
        "MC candidate selection. 'truthMatched' uses directly matched D* chains with "
        "valid CVH refits and basic acceptance; 'dataLike' additionally requires the "
        "producer's full transient-fit data selection."
    ),
)
fit_delta_m_group = parser.add_mutually_exclusive_group()
fit_delta_m_group.add_argument(
    "--applyFitDeltaM",
    dest="applyFitDeltaM",
    action="store_true",
    default=True,
    help=("Apply the fitted DeltaM cut " "abs(deltaM_D0pis_piK - 0.14543) < 0.003."),
)
fit_delta_m_group.add_argument(
    "--noApplyFitDeltaM",
    dest="applyFitDeltaM",
    action="store_false",
    help="Disable the fitted DeltaM cut in the data-like selection",
)
parser.add_argument(
    "--pvalCut",
    type=float,
    default=None,
    help=(
        "Minimum vertex-fit p-value required for both the piK (D0) and D0pis (D*) "
        "vertices. Disabled by default because some ntuples store an unfilled -99 "
        "sentinel in pval_piK/pval_D0pis. Set e.g. 0.005 once valid p-values are "
        "produced. When enabled, sentinel (-99) values are retained and the "
        "pval_piK/pval_D0pis branches are only required in that case."
    ),
)
parser.add_argument(
    "--daughterPtCut",
    type=float,
    default=1.0,
    help="minimum required pt for K and pi",
)
parser.add_argument(
    "--kinDiagnostics",
    action="store_true",
    help=(
        "Save kinematic diagnostic histograms (MC only, filled after gen matching): "
        "the gen and reco pt of the K and pi with their qopr = (reco qop)/(gen qop) "
        "(qop = charge/(pt*cosh(eta))), and the analogous gen and reco mRK with their "
        "ratio mRK_reco/mRK_gen."
    ),
)
parser.add_argument(
    "--genPtReweightSaturation",
    type=float,
    default=None,
    help=(
        "If set, replace the gen pt fed into the scale/resolution reweights with "
        "max(gen pt, VALUE) [GeV]. Use to floor the D0 daughters' gen pt at the "
        "reweight models' training range (~2 GeV) to avoid low-pt extrapolation. "
        "The exact treatment is controlled by --genPtReweightSaturationMode. "
        "Affects only the reweight inputs; the diagnostic, nominal, "
        "and manual A/e/M histograms are unchanged."
    ),
)
parser.add_argument(
    "--genPtReweightSaturationMode",
    type=str,
    default="condition",
    choices=["rescale", "condition"],
    help=(
        "How --genPtReweightSaturation is applied to the ONNX reweight. 'rescale': "
        "scale both reco and gen pt columns by max(1, sat/gen_pt), "
        "preserving qopr (delta_r_kappa is evaluated at the saturated pt). "
        "'condition' (default): feed the real reco/gen pt (so y_raw and "
        "delta_r_kappa use the "
        "true pt) and floor the gen pt only for the network's conditioning input "
        "(log pt_gen). Applied consistently to the scale and resolution ONNX "
        "helpers; it has no effect with the spline backend."
    ),
)
parser.add_argument(
    "--etaBins",
    type=int,
    default=None,
    help="Override the number of eta bins used by fitted scale/resolution variations",
)
parser.add_argument(
    "--templateEtaKBins",
    type=int,
    default=24,
    help="Number of regular etaK template bins over [-2.4, 2.4]",
)
parser.add_argument(
    "--templateMRKBins",
    type=int,
    default=None,
    help=(
        "Use this many regular mRK template bins over [0, 1.8]. If omitted, "
        "retain the legacy 19-bin variable-width axis. Mutually exclusive with "
        "--templateMRKEdges."
    ),
)
parser.add_argument(
    "--templateMRKEdges",
    type=float,
    nargs="+",
    default=None,
    help=(
        "Explicit mRK template bin edges (>=2, strictly increasing). Mutually "
        "exclusive with --templateMRKBins."
    ),
)
parser.add_argument(
    "--pionTemplateAxes",
    action="store_true",
    help=(
        "Save 5D templates (etaK, mRK, etaPi, mRpi, D0mass) instead of the "
        "default 3D (etaK, mRK, D0mass), where etaPi/mRpi are the literal K->pi "
        "swaps of etaK/mRK. Requires, for each of mRK and mRpi, either the bin "
        "count or the bin edges to be specified."
    ),
)
parser.add_argument(
    "--templateEtaPiBins",
    type=int,
    default=None,
    help=(
        "Number of regular etaPi template bins over [-2.4, 2.4] when "
        "--pionTemplateAxes is set. Defaults to --templateEtaKBins."
    ),
)
parser.add_argument(
    "--templateMRpiBins",
    type=int,
    default=None,
    help=(
        "Number of uniform mRpi template bins over [0, 0.20] (the mRpi range, "
        "~(M_pi/M_K)^2 times the mRK range) when --pionTemplateAxes is set. "
        "Mutually exclusive with --templateMRpiEdges."
    ),
)
parser.add_argument(
    "--templateMRpiEdges",
    type=float,
    nargs="+",
    default=None,
    help=(
        "Explicit mRpi template bin edges (>=2, strictly increasing) when "
        "--pionTemplateAxes is set. Mutually exclusive with --templateMRpiBins."
    ),
)
parser.add_argument(
    "--resolutionPrefitUncertainty",
    type=float,
    default=0.3,
    help=(
        "Relative prefit uncertainty assigned to each fitted resolution "
        "parameter when --fitMuonScaleAndResolution is used."
    ),
)
parser.add_argument(
    "--resolutionPrefitUncertaintyA",
    type=float,
    default=None,
    help=(
        "Override the relative prefit uncertainty for only the fitted "
        "resolution a parameter. If omitted, --resolutionPrefitUncertainty "
        "is used for a, b, c, and d."
    ),
)
parser.add_argument(
    "--buildManualScaleVariations",
    action="store_true",
    help="Include manual scale variations in addition to reweights",
)
parser = parsing.set_parser_default(parser, "theoryCorr", [])
parser = parsing.set_parser_default(parser, "scale_A", 1.0)
parser = parsing.set_parser_default(parser, "scale_e", 1.0)
parser = parsing.set_parser_default(parser, "fitMuonScaleAndResolution", True)

args = parser.parse_args()

if args.etaBins is not None and args.etaBins <= 0:
    raise ValueError("--etaBins must be a positive integer")
if args.templateEtaKBins <= 0:
    raise ValueError("--templateEtaKBins must be a positive integer")
if args.templateMRKBins is not None and args.templateMRKBins <= 0:
    raise ValueError("--templateMRKBins must be a positive integer")
if args.templateMRpiBins is not None and args.templateMRpiBins <= 0:
    raise ValueError("--templateMRpiBins must be a positive integer")
if args.templateEtaPiBins is not None and args.templateEtaPiBins <= 0:
    raise ValueError("--templateEtaPiBins must be a positive integer")
# Bins and edges are mutually exclusive per reduced-mass axis.
if args.templateMRKBins is not None and args.templateMRKEdges is not None:
    raise ValueError(
        "--templateMRKBins and --templateMRKEdges are mutually exclusive"
    )
if args.templateMRpiBins is not None and args.templateMRpiEdges is not None:
    raise ValueError(
        "--templateMRpiBins and --templateMRpiEdges are mutually exclusive"
    )
# 5D output requires each reduced-mass axis to be specified as bins OR edges.
if args.pionTemplateAxes:
    if args.templateMRKBins is None and args.templateMRKEdges is None:
        raise ValueError(
            "--pionTemplateAxes (5D output) requires --templateMRKBins or "
            "--templateMRKEdges"
        )
    if args.templateMRpiBins is None and args.templateMRpiEdges is None:
        raise ValueError(
            "--pionTemplateAxes (5D output) requires --templateMRpiBins or "
            "--templateMRpiEdges"
        )
if args.resolutionPrefitUncertainty <= 0:
    raise ValueError("--resolutionPrefitUncertainty must be positive")
if (
    args.resolutionPrefitUncertaintyA is not None
    and args.resolutionPrefitUncertaintyA <= 0
):
    raise ValueError("--resolutionPrefitUncertaintyA must be positive")
if args.muonScaleVariation not in ["onnxReweight", "smearingWeightsSplines"]:
    raise ValueError(
        "d0_mass.py currently supports --muonScaleVariation onnxReweight "
        "or smearingWeightsSplines"
    )

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

# Daughter masses [GeV]. The D0 histmaker always uses the mass-aware energy-loss
# term in the ONNX scale reweight (K mass for the K leg, pi mass for the pi leg),
# so these are passed to the reweight helper below and reused for the four-vectors.
M_K = 0.493677
M_PI = 0.139570

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
    reweight_mass=[M_K, M_PI],
    # "condition" mode: keep the reweight columns real and floor the gen pt only for
    # the network's conditioning input, inside the helper. Off for "rescale"/no-sat.
    cond_pt_gen_min=(
        args.genPtReweightSaturation
        if (
            args.genPtReweightSaturation is not None
            and args.genPtReweightSaturationMode == "condition"
        )
        else None
    ),
)

if data_jpsi_crctn_unc_helper is None:
    raise ValueError("Scale systematics require --muonCorrData massfit or lbl_massfit")

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
        cond_pt_gen_min=(
            args.genPtReweightSaturation
            if (
                args.genPtReweightSaturation is not None
                and args.genPtReweightSaturationMode == "condition"
            )
            else None
        ),
    )
)

ROOT.gInterpreter.Declare("""
    #include <cmath>
    #include <algorithm>
    #include "ROOT/RVec.hxx"

    namespace wrem {
    namespace d0 {

    template <typename T>
    inline double first_or_value(const T &value) {
        return static_cast<double>(value);
    }

    template <typename T>
    inline double first_or_value(const ROOT::VecOps::RVec<T> &values) {
        return values.empty() ? -999.0 : static_cast<double>(values[0]);
    }

    inline double delta_phi(double phi1, double phi2) {
        double dphi = phi1 - phi2;
        while (dphi > M_PI) dphi -= 2.0*M_PI;
        while (dphi <= -M_PI) dphi += 2.0*M_PI;
        return dphi;
    }

    template <typename PtVec, typename EtaVec, typename PhiVec, typename ChargeVec,
              typename PdgIdVec, typename MotherPdgIdVec,
              typename GrandmotherPdgIdVec>
    ROOT::VecOps::RVec<float> matched_gen_kinematics(
        double reco_pt,
        double reco_eta,
        double reco_phi,
        int reco_charge,
        const PtVec &gen_pt,
        const EtaVec &gen_eta,
        const PhiVec &gen_phi,
        const ChargeVec &gen_charge,
        const PdgIdVec &gen_pdgId,
        const MotherPdgIdVec &gen_motherPdgId,
        const GrandmotherPdgIdVec &gen_grandmotherPdgId,
        int abs_pdgid,
        int abs_mother_pdgid,
        int abs_grandmother_pdgid,
        double max_dr = 0.03
    ) {
        const auto size = std::min({gen_pt.size(), gen_eta.size(), gen_phi.size(),
                                    gen_charge.size(), gen_pdgId.size(),
                                    gen_motherPdgId.size(),
                                    gen_grandmotherPdgId.size()});
        const double max_dr2 = max_dr*max_dr;
        double best_dr2 = max_dr2;
        int best = -1;
        for (std::size_t i = 0; i < size; ++i) {
            if (std::abs(static_cast<int>(gen_pdgId[i])) != abs_pdgid) continue;
            if (std::abs(static_cast<int>(gen_motherPdgId[i])) != abs_mother_pdgid) continue;
            if (std::abs(static_cast<int>(gen_grandmotherPdgId[i])) !=
                abs_grandmother_pdgid) continue;
            if (static_cast<int>(gen_charge[i]) != reco_charge) continue;

            const double deta = reco_eta - static_cast<double>(gen_eta[i]);
            const double dphi = delta_phi(reco_phi, static_cast<double>(gen_phi[i]));
            const double dr2 = deta*deta + dphi*dphi;
            if (dr2 < best_dr2) {
                best_dr2 = dr2;
                best = static_cast<int>(i);
            }
        }

        if (best < 0) return ROOT::VecOps::RVec<float>{0.f, 0.f, 0.f, 0.f, 0.f};

        return ROOT::VecOps::RVec<float>{
            static_cast<float>(gen_pt[best]),
            static_cast<float>(gen_eta[best]),
            static_cast<float>(gen_phi[best]),
            static_cast<float>(gen_charge[best]),
            1.f
        };
    }

    }
    }
    """)

# Manual A/e/M scale variations: recompute the D0 mass (and mRK) from the K/pi
# four-vectors under an independent +-1 prefit-width shift of each (eta bin,
# scale parameter), instead of reweighting. The finite shift is evaluated exactly:
# A and M modify curvature, while e shifts total energy before converting back to
# pt with the required 1/cosh(eta). No first-order dpt approximation is used.
# Both the varied mass and mRK are expressed as a shift on top of their authoritative
# nominal values (D0_CVH_mass0 and the nominal mRK column), so a variation that touches
# neither track's eta bin lands exactly on the nominal histogram. Mass and mRK share the
# same shifted momenta (computed once) so they never drift apart.
ROOT.gInterpreter.Declare("""
    #include <cmath>
    #include "ROOT/RVec.hxx"
    #include "Math/Vector4D.h"

    namespace wrem {
    namespace d0 {

    // Per-event varied observables: mass[k], mRK[k], mRpi[k] share the index k.
    struct ScaleVarResult {
        ROOT::VecOps::RVec<double> mass;
        ROOT::VecOps::RVec<double> mRK;
        ROOT::VecOps::RVec<double> mRpi;
    };

    class ScaleVariationsHelper {
    public:
        ScaleVariationsHelper(unsigned int n_eta_bins, double eta_min, double eta_max,
                              unsigned int n_scale_params,
                              double width_A, double width_e, double width_M,
                              double mK, double mPi)
          : n_eta_bins_(n_eta_bins), n_scale_params_(n_scale_params),
            nvars_(n_eta_bins * n_scale_params),
            eta_min_(eta_min), eta_max_(eta_max),
            width_A_(width_A), width_e_(width_e), width_M_(width_M),
            mK_(mK), mPi_(mPi) {}

        int eta_bin(double eta) const {
            if (eta < eta_min_ || eta >= eta_max_) return -1;
            int ib = static_cast<int>((eta - eta_min_) / (eta_max_ - eta_min_) * n_eta_bins_);
            if (ib < 0 || ib >= static_cast<int>(n_eta_bins_)) return -1;
            return ib;
        }

        // pt after a finite scale-parameter shift. The shared implementation
        // applies A/M in curvature and e in total energy, including /cosh(eta).
        double shifted_pt(double pt, double q, double eta, double mass, int track_ieta,
                          unsigned int ieta, unsigned int iparm, double sgn) const {
            if (track_ieta != static_cast<int>(ieta)) return pt;
            const double AShift = iparm == 0 ? sgn * width_A_ : 0.0;
            const double eShift = iparm == 1 ? sgn * width_e_ : 0.0;
            const double MShift = iparm == 2 ? sgn * width_M_ : 0.0;
            return wrem::calculateShiftedPtExact(
                pt, eta, static_cast<int>(q), AShift, eShift, MShift, mass);
        }

        // mRK observable, matching the nominal column M_K^2 * (pi_E/K_E) / mass.
        double mrk(double eK, double ePi, double mass) const {
            return mK_ * mK_ * (ePi / eK) / mass;
        }

        // mRpi: literal K<->pi swap of mRK, matching M_pi^2 * (K_E/pi_E) / mass.
        double mrpi(double eK, double ePi, double mass) const {
            return mPi_ * mPi_ * (eK / ePi) / mass;
        }

        // Returns nvars*2 varied mass, mRK, mRpi values, ordered (ieta, iparm, {down, up})
        // so the flattened index is ivar*2 + isign with ivar = ieta*n_scale_params + iparm.
        ScaleVarResult operator()(
            double ptK, double etaK, double phiK, double qK,
            double ptPi, double etaPi, double phiPi, double qPi,
            double d0_mass_nom, double mRK_nom, double mRpi_nom) const {

            const int biK = eta_bin(etaK);
            const int biPi = eta_bin(etaPi);

            const double coshK = std::cosh(etaK);
            const double coshPi = std::cosh(etaPi);
            const auto energy = [](double pt, double cosh_eta, double m) {
                return std::sqrt(pt * cosh_eta * pt * cosh_eta + m * m);
            };

            const ROOT::Math::PtEtaPhiMVector v0K(ptK, etaK, phiK, mK_);
            const ROOT::Math::PtEtaPhiMVector v0Pi(ptPi, etaPi, phiPi, mPi_);
            const double m_nom = (v0K + v0Pi).M();
            const double eK_nom = energy(ptK, coshK, mK_);
            const double ePi_nom = energy(ptPi, coshPi, mPi_);
            const double mrk_nom_recomp = mrk(eK_nom, ePi_nom, m_nom);
            const double mrpi_nom_recomp = mrpi(eK_nom, ePi_nom, m_nom);

            ScaleVarResult res;
            res.mass.reserve(nvars_ * 2);
            res.mRK.reserve(nvars_ * 2);
            res.mRpi.reserve(nvars_ * 2);
            const double signs[2] = {-1.0, 1.0};  // down, up
            for (unsigned int ieta = 0; ieta < n_eta_bins_; ++ieta) {
                for (unsigned int iparm = 0; iparm < n_scale_params_; ++iparm) {
                    for (int isign = 0; isign < 2; ++isign) {
                        const double sgn = signs[isign];
                        const double ptKs =
                            shifted_pt(ptK, qK, etaK, mK_, biK, ieta, iparm, sgn);
                        const double ptPis =
                            shifted_pt(ptPi, qPi, etaPi, mPi_, biPi, ieta, iparm, sgn);
                        const ROOT::Math::PtEtaPhiMVector vK(ptKs, etaK, phiK, mK_);
                        const ROOT::Math::PtEtaPhiMVector vPi(ptPis, etaPi, phiPi, mPi_);
                        const double m_var = (vK + vPi).M();
                        const double eK_var = energy(ptKs, coshK, mK_);
                        const double ePi_var = energy(ptPis, coshPi, mPi_);
                        const double mrk_var = mrk(eK_var, ePi_var, m_var);
                        const double mrpi_var = mrpi(eK_var, ePi_var, m_var);
                        res.mass.push_back(d0_mass_nom + (m_var - m_nom));
                        res.mRK.push_back(mRK_nom + (mrk_var - mrk_nom_recomp));
                        res.mRpi.push_back(mRpi_nom + (mrpi_var - mrpi_nom_recomp));
                    }
                }
            }
            return res;
        }

    private:
        unsigned int n_eta_bins_, n_scale_params_, nvars_;
        double eta_min_, eta_max_, width_A_, width_e_, width_M_, mK_, mPi_;
    };

    // Constant per-fill index vectors matching the (ivar, {down, up}) ordering above.
    inline ROOT::VecOps::RVec<int> scale_var_indices(unsigned int nvars) {
        ROOT::VecOps::RVec<int> out;
        out.reserve(nvars * 2);
        for (unsigned int ivar = 0; ivar < nvars; ++ivar) {
            out.push_back(static_cast<int>(ivar));
            out.push_back(static_cast<int>(ivar));
        }
        return out;
    }
    // down -> -1.0 (downUpVar bin 0), up -> +1.0 (downUpVar bin 1)
    inline ROOT::VecOps::RVec<double> updown_coords(unsigned int nvars) {
        ROOT::VecOps::RVec<double> out;
        out.reserve(nvars * 2);
        for (unsigned int ivar = 0; ivar < nvars; ++ivar) {
            out.push_back(-1.0);
            out.push_back(1.0);
        }
        return out;
    }

    }
    }
    """)


def limited_files(files):
    if args.maxFiles is not None and args.maxFiles > 0:
        return files[: args.maxFiles]
    return files


def input_files(paths):
    """Expand all roots, then deduplicate and apply the global file limit."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    resolved_files = []
    for path in paths:
        if os.path.isdir(path):
            matches = glob.glob(os.path.join(path, "**", "*.root"), recursive=True)
        else:
            matches = glob.glob(path)
        files = sorted(
            os.path.realpath(match) for match in matches if os.path.isfile(match)
        )
        if not files:
            raise ValueError(f"No input ROOT files found for {path}")
        resolved_files.extend(files)

    return limited_files(sorted(set(resolved_files)))


datasets = [
    narf.Dataset(
        name="D0_data",
        filepaths=input_files(args.dataFile),
        is_data=True,
        xsec=None,
        group="Data",
    ),
    narf.Dataset(
        name="D0_mc",
        filepaths=input_files(args.mcFile),
        is_data=False,
        xsec=1.0,
        group="D0",
    ),
]

template_axes = make_d0_template_axes(
    eta_bins=args.templateEtaKBins,
    mrk_bins=args.templateMRKBins,
    mrk_edges=args.templateMRKEdges,
    pion_axes=args.pionTemplateAxes,
    eta_pi_bins=args.templateEtaPiBins,
    mrpi_bins=args.templateMRpiBins,
    mrpi_edges=args.templateMRpiEdges,
)
d0_axes = list(template_axes)

# Coordinate columns matching d0_axes, in the same order. In 5D mode the pion
# analogues (pi_CVH_eta0, mRpi) are inserted before the D0 mass (shape) axis.
if args.pionTemplateAxes:
    coord_cols = ["K_CVH_eta0", "mRK", "pi_CVH_eta0", "mRpi", "D0_mass"]
else:
    coord_cols = ["K_CVH_eta0", "mRK", "D0_mass"]

# Axes for the optional --kinDiagnostics histograms. mRK uses the wide mR axis;
# mRpi gets its own ~12x-compressed axis (the K<->pi swap scales the reduced mass
# by ~(M_pi/M_K)^2 ~ 0.08, with an energy-ratio tail extending a bit past that, so
# [0, 0.20] covers the bulk with headroom). The mR ratio gets its own axis (like qopr).
axis_diag_pt = hist.axis.Regular(100, 0.0, 20.0, name="pt")
axis_diag_eta = hist.axis.Regular(96, -2.4, 2.4, name="eta")
axis_diag_mR = hist.axis.Regular(100, 0.0, 1.80, name="mR")
axis_diag_mRpi = hist.axis.Regular(100, 0.0, 0.20, name="mRpi")
axis_diag_qopr = hist.axis.Regular(200, 0.5, 1.5, name="qopr")
axis_diag_mR_ratio = hist.axis.Regular(200, 0.0, 2.0, name="mRr")

# Manual A/e/M scale-variation histogram. The per-(eta bin, parameter) construction
# below is only valid for the diagonal prefit-width configuration (no covariance /
# eigen-decomposition), i.e. --fitMuonScaleAndResolution, which is the default here.
# The scale-variation and down/up axes are reused verbatim from the response-weight
# uncertainty helper so the manual histogram is structurally identical to
# nominal_muonScaleSyst_responseWeights and the "unc" index maps to the same nuisance.
build_manual_scale_variations = args.buildManualScaleVariations
d0_scale_var_helper = None
manual_scale_axes = None
if build_manual_scale_variations:
    if not args.fitMuonScaleAndResolution:
        raise ValueError(
            "--buildManualScaleVariations requires "
            "--fitMuonScaleAndResolution so the nuisance axis is the direct "
            "eta-bin x (A,e,M) basis"
        )
    _scale_var_axis, _updown_axis = data_jpsi_crctn_unc_helper.tensor_axes
    _nvars = _scale_var_axis.size
    if _nvars % 3 != 0:
        raise ValueError(
            f"Expected the scale unc axis size to be a multiple of 3 (A, e, M per "
            f"eta bin), got {_nvars}"
        )
    _n_eta_bins_scale = _nvars // 3
    d0_scale_var_helper = ROOT.wrem.d0.ScaleVariationsHelper(
        _n_eta_bins_scale,
        -2.4,
        2.4,
        3,
        1e-3 * args.scale_A,
        1e-2 * args.scale_e,
        1e-4 * args.scale_M,
        M_K,
        M_PI,
    )
    manual_scale_axes = [_scale_var_axis, _updown_axis]
    logger.info(
        f"Manual scale variations enabled: {_nvars} nuisances "
        f"({_n_eta_bins_scale} eta bins x 3 parameters), down/up each."
    )


def bool_filter(expression):
    return f"static_cast<bool>({expression})"


def scalarize(df, names):
    for name in names:
        df = df.Define(f"{name}0", f"wrem::d0::first_or_value({name})")
    return df


def available_columns(df):
    return {str(col) for col in df.GetColumnNames()}


def require_columns(columns, names):
    missing = [name for name in names if name not in columns]
    if missing:
        raise ValueError(f"Missing required D0 ntuple branches: {', '.join(missing)}")


def available_first(columns, names, label):
    available = [name for name in names if name in columns]
    if not available:
        raise ValueError(
            f"Missing D0 ntuple branch for {label}; tried {', '.join(names)}"
        )
    return available


def adapt_ntuple_schema(df, dataset):
    """Map the native ALCARECO data/MC trees onto one calibration schema."""
    aliases = {
        "K_CVH_pt": ["CVH_K_CVH_pt"],
        "K_CVH_eta": ["CVH_K_CVH_eta"],
        "K_CVH_phi": ["CVH_K_CVH_phi"],
        "K_charge": ["CVH_K_charge", "MC_reco_K_charge"],
        "pi_CVH_pt": ["CVH_pi_CVH_pt"],
        "pi_CVH_eta": ["CVH_pi_CVH_eta"],
        "pi_CVH_phi": ["CVH_pi_CVH_phi"],
        "pi_charge": ["CVH_pi_charge", "MC_reco_pi_charge"],
        "pis_CVH_pt": ["CVH_pis_CVH_pt"],
        "chi2_piK": ["MC_D0_fit_chi2", "CVH_chi2_piK"],
        "pval_piK": ["MC_D0_fit_pval", "CVH_pval_piK"],
        "mass_piK": ["MC_D0_fit_mass", "CVH_mass_piK"],
        "chi2_D0pis": ["MC_Dst_fit_chi2", "CVH_chi2_D0pis"],
        "pval_D0pis": ["MC_Dst_fit_pval", "CVH_pval_D0pis"],
        "mass_D0pis": ["MC_Dst_fit_mass", "CVH_mass_D0pis"],
        "deltaM_D0pis_piK": ["deltaMass_D0pis"],
        "Dst_pt": ["CVH_Dst_pt", "truth_Dst_pt"],
        "Dst_iso": ["CVH_Dst_iso"],
        "pis_dR_D0": ["CVH_pis_dR_D0", "CVH_pis_dR_D0raw"],
        "D0_CVH_mass": ["CVH_D0_CVH_mass"],
        "D0_CVH_valid": ["CVH_D0_CVH_valid"],
        "pis_CVH_edmval": ["CVH_pis_CVH_edmval"],
    }
    columns = available_columns(df)
    for target, sources in aliases.items():
        if target in columns:
            continue
        source = next((name for name in sources if name in columns), None)
        if source is not None:
            df = df.Define(target, source)
            columns.add(target)

    # The data production predates the explicit CVH-valid boolean. A positive mass
    # is the corresponding persisted validity condition there.
    if "D0_CVH_valid" not in columns and "D0_CVH_mass" in columns:
        df = df.Define("D0_CVH_valid", "D0_CVH_mass > 0.0")
        columns.add("D0_CVH_valid")

    # The matched-candidate tree stores both fitted masses but not their difference.
    if (
        "deltaM_D0pis_piK" not in columns
        and "mass_D0pis" in columns
        and "mass_piK" in columns
    ):
        df = df.Define("deltaM_D0pis_piK", "mass_D0pis - mass_piK")

    logger.info(
        f"Using {'data' if dataset.is_data else 'MC'} tree schema for {dataset.name}"
    )
    return df


def define_reco(df):
    columns = available_columns(df)
    require_columns(
        columns,
        [
            "K_CVH_pt",
            "K_CVH_eta",
            "K_CVH_phi",
            "K_charge",
            "pi_CVH_pt",
            "pi_CVH_eta",
            "pi_CVH_phi",
            "pi_charge",
            "pis_CVH_pt",
            "chi2_piK",
            "chi2_D0pis",
            "Dst_pt",
            "Dst_iso",
            "pis_dR_D0",
            "D0_CVH_valid",
        ],
    )
    # The vertex-fit p-value branches are only needed when the cut is enabled;
    # they are unfilled (-99) in current productions, so do not require them by default.
    pval_cols = ["pval_piK", "pval_D0pis"] if args.pvalCut is not None else []
    if pval_cols:
        require_columns(columns, pval_cols)
    d0_fit_mass_cols = available_first(
        columns, ["mass_piK", "D0_fit_mass"], "transient D0 fitted mass"
    )
    d0_cvh_mass_cols = available_first(
        columns, ["D0_CVH_mass", "mass_piK", "D0_fit_mass"], "CVH D0 mass"
    )
    dst_mass_cols = available_first(
        columns,
        ["mass_D0pis", "Dst_fit_mass", "Dst_CVH3_mass", "Dst_CVH_mass"],
        "D* fitted mass",
    )
    delta_m_cols = [
        name for name in ["deltaM_D0pis_piK", "Dst_fit_deltaM"] if name in columns
    ]

    reco_cols = list(
        dict.fromkeys(
            [
                "K_CVH_pt",
                "K_CVH_eta",
                "K_CVH_phi",
                "K_charge",
                "pi_CVH_pt",
                "pi_CVH_eta",
                "pi_CVH_phi",
                "pi_charge",
                *pval_cols,
                "pis_CVH_pt",
                "chi2_piK",
                "chi2_D0pis",
                "Dst_pt",
                "Dst_iso",
                "pis_dR_D0",
                "D0_CVH_valid",
                *d0_fit_mass_cols,
                *d0_cvh_mass_cols,
                *dst_mass_cols,
                *delta_m_cols,
            ]
        )
    )
    df = scalarize(df, reco_cols)
    d0_fit_expr = f"{d0_fit_mass_cols[-1]}0"
    for name in reversed(d0_fit_mass_cols[:-1]):
        d0_fit_expr = f"({name}0 > 0.0 ? {name}0 : {d0_fit_expr})"
    d0_cvh_expr = f"{d0_cvh_mass_cols[-1]}0"
    for name in reversed(d0_cvh_mass_cols[:-1]):
        d0_cvh_expr = f"({name}0 > 0.0 ? {name}0 : {d0_cvh_expr})"
    dst_expr = f"{dst_mass_cols[-1]}0"
    for name in reversed(dst_mass_cols[:-1]):
        dst_expr = f"({name}0 > 0.0 ? {name}0 : {dst_expr})"
    if delta_m_cols:
        delta_m_expr = f"{delta_m_cols[-1]}0"
        for name in reversed(delta_m_cols[:-1]):
            delta_m_expr = f"({name}0 > 0.0 ? {name}0 : {delta_m_expr})"
    else:
        delta_m_expr = "Dst_fit_mass_for_selection - D0_fit_mass_for_selection"

    df = (
        df.Define(
            "K_E0",
            f"std::sqrt(std::pow(K_CVH_pt0*std::cosh(K_CVH_eta0), 2) + {M_K}*{M_K})",
        )
        .Define(
            "pi_E0",
            f"std::sqrt(std::pow(pi_CVH_pt0*std::cosh(pi_CVH_eta0), 2) + {M_PI}*{M_PI})",
        )
        .Define("D0_fit_mass_for_selection", d0_fit_expr)
        .Define("Dst_fit_mass_for_selection", dst_expr)
        .Define("Dst_fit_deltaM_for_selection", delta_m_expr)
        .Define("D0_mass", d0_cvh_expr)
        .Define("mRK", f"{M_K}*{M_K}*(pi_E0/K_E0)/D0_mass")
        # Literal K<->pi swap of mRK (mass prefactor and energy ratio both flip).
        .Define("mRpi", f"{M_PI}*{M_PI}*(K_E0/pi_E0)/D0_mass")
    )
    if args.applyFitDeltaM and not delta_m_cols:
        logger.warning(
            "No fitted DeltaM branch found; using D* fitted mass minus D0 fitted mass"
        )
    return df


def data_selection():
    selection = [
        f"K_CVH_pt0 > {args.daughterPtCut}",
        f"pi_CVH_pt0 > {args.daughterPtCut}",
        "pis_CVH_pt0 > 0.35",
        "K_E0 > 0.0",
        "pi_E0 > 0.0",
        "std::fabs(K_CVH_eta0) < 2.4",
        "std::fabs(pi_CVH_eta0) < 2.4",
        "D0_CVH_valid0 > 0.5",
        "D0_fit_mass_for_selection > 0.0",
        "Dst_fit_mass_for_selection > 0.0",
        "Dst_fit_deltaM_for_selection > 0.0",
        "chi2_piK0 > 0.0",
        "chi2_D0pis0 > 0.0",
        "std::fabs(D0_fit_mass_for_selection - 1.86483) < 0.035",
        "pis_dR_D00 < 0.12",
        "Dst_pt0 > 5.0",
        "Dst_iso0 > 0.20",
    ]
    if args.pvalCut is not None:
        # Guard against the unfilled -99 sentinel: reject a candidate only when its
        # p-value is valid (> -1) and below threshold. Sentinel values are kept.
        selection.append(f"(pval_piK0 < -1.0 || pval_piK0 > {args.pvalCut})")
        selection.append(f"(pval_D0pis0 < -1.0 || pval_D0pis0 > {args.pvalCut})")
    if args.applyFitDeltaM:
        selection.append("std::fabs(Dst_fit_deltaM_for_selection - 0.14543) < 0.003")
    return " && ".join(selection)


def truth_matched_mc_selection():
    return " && ".join(
        [
            f"K_CVH_pt0 > {args.daughterPtCut}",
            f"pi_CVH_pt0 > {args.daughterPtCut}",
            "pis_CVH_pt0 > 0.35",
            "std::fabs(K_CVH_eta0) < 2.4",
            "std::fabs(pi_CVH_eta0) < 2.4",
            "D0_CVH_valid0 > 0.5",
            "D0_mass > 0.0",
        ]
    )


def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")

    results = []
    df = df.DefinePerSample("weight", "1.0")
    weightsum = df.SumAndCount("weight")
    df = adapt_ntuple_schema(df, dataset)
    df = define_reco(df)
    if dataset.is_data:
        df = df.Filter(bool_filter(data_selection()))
    elif args.mcSelection == "dataLike":
        columns = available_columns(df)
        if "MC_passDataSelection" not in columns:
            raise ValueError(
                "--mcSelection dataLike requires MC_passDataSelection in the MC tree"
            )
        df = df.Filter("MC_passDataSelection").Filter(bool_filter(data_selection()))
    else:
        df = df.Filter(bool_filter(truth_matched_mc_selection()))

    if dataset.is_data:
        results.append(
            df.HistoBoost(
                "hD0_data",
                d0_axes,
                coord_cols + ["weight"],
            )
        )
    else:
        # Gen-pt saturation for the reweights (see --genPtReweightSaturation[Mode]).
        # In "rescale" mode we scale BOTH the reco and gen pt columns by the per-leg
        # factor max(1, sat/gen_pt): this floors gen pt at sat while preserving
        # kappa_reco/kappa_gen (qopr), so the network sees an in-domain point instead of
        # an artificial residual. In "condition" mode (or no saturation) the columns
        # stay real and the gen-pt floor is applied inside the helper to the network's
        # conditioning input only. Only the scale_* reweight inputs are ever affected;
        # eta/phi are untouched so the angular residuals are preserved.
        if (
            args.genPtReweightSaturation is not None
            and args.genPtReweightSaturationMode == "rescale"
        ):
            sat = args.genPtReweightSaturation
            factor_K = f"std::max<float>(1.0f, {sat}f / K_gen_match[0])"
            factor_pi = f"std::max<float>(1.0f, {sat}f / pi_gen_match[0])"
            reco_pt_K = f"float(K_CVH_pt0) * {factor_K}"
            reco_pt_pi = f"float(pi_CVH_pt0) * {factor_pi}"
            gen_pt_K = f"std::max<float>(K_gen_match[0], {sat}f)"
            gen_pt_pi = f"std::max<float>(pi_gen_match[0], {sat}f)"
            logger.info(
                f"Saturating reweight gen pt at {sat} GeV and scaling reco pt by the "
                f"same factor to preserve qopr (scale_recoPt, scale_genPt)"
            )
        else:
            reco_pt_K = "float(K_CVH_pt0)"
            reco_pt_pi = "float(pi_CVH_pt0)"
            gen_pt_K = "K_gen_match[0]"
            gen_pt_pi = "pi_gen_match[0]"
        df = (
            df.Define("nominal_weight", "weight")
            .Define(
                "scale_recoEta",
                "ROOT::VecOps::RVec<float>{float(K_CVH_eta0), float(pi_CVH_eta0)}",
            )
            .Define(
                "scale_recoPhi",
                "ROOT::VecOps::RVec<float>{float(K_CVH_phi0), float(pi_CVH_phi0)}",
            )
            .Define(
                "scale_recoCharge",
                "ROOT::VecOps::RVec<int>{int(K_charge0), int(pi_charge0)}",
            )
        )
        direct_truth_columns = {
            "truth_K_pt",
            "truth_K_eta",
            "truth_K_phi",
            "truth_K_charge",
            "truth_pi_pt",
            "truth_pi_eta",
            "truth_pi_phi",
            "truth_pi_charge",
        }
        if direct_truth_columns.issubset(available_columns(df)):
            logger.info("Using direct K/pi truth from the matched-candidate MC tree")
            df = df.Define(
                "K_gen_match",
                "ROOT::VecOps::RVec<float>{float(truth_K_pt), "
                "float(truth_K_eta), float(truth_K_phi), "
                "float(truth_K_charge), 1.f}",
            ).Define(
                "pi_gen_match",
                "ROOT::VecOps::RVec<float>{float(truth_pi_pt), "
                "float(truth_pi_eta), float(truth_pi_phi), "
                "float(truth_pi_charge), 1.f}",
            )
        else:
            logger.warning(
                "Direct matched truth is unavailable; falling back to ancestry-aware "
                "DeltaR matching against the legacy gen-particle vectors"
            )
            legacy_truth_columns = {
                "gen_pt",
                "gen_eta",
                "gen_phi",
                "gen_charge",
                "gen_pdgId",
                "gen_motherPdgId",
                "gen_grandmotherPdgId",
            }
            missing_truth = sorted(legacy_truth_columns - available_columns(df))
            if missing_truth:
                raise ValueError(
                    "Legacy MC truth matching requires branches: "
                    + ", ".join(missing_truth)
                )
            df = df.Define(
                "K_gen_match",
                "wrem::d0::matched_gen_kinematics("
                "K_CVH_pt0, K_CVH_eta0, K_CVH_phi0, int(K_charge0), "
                "gen_pt, gen_eta, gen_phi, gen_charge, gen_pdgId, "
                "gen_motherPdgId, gen_grandmotherPdgId, 321, 421, 413)",
            ).Define(
                "pi_gen_match",
                "wrem::d0::matched_gen_kinematics("
                "pi_CVH_pt0, pi_CVH_eta0, pi_CVH_phi0, int(pi_charge0), "
                "gen_pt, gen_eta, gen_phi, gen_charge, gen_pdgId, "
                "gen_motherPdgId, gen_grandmotherPdgId, 211, 421, 413)",
            )
        df = (
            df.Define(
                "scale_recoPt",
                f"ROOT::VecOps::RVec<float>{{{reco_pt_K}, {reco_pt_pi}}}",
            )
            .Define(
                "scale_genPt",
                f"ROOT::VecOps::RVec<float>{{{gen_pt_K}, {gen_pt_pi}}}",
            )
            .Define(
                "scale_genEta",
                "ROOT::VecOps::RVec<float>{K_gen_match[1], pi_gen_match[1]}",
            )
            .Define(
                "scale_genPhi",
                "ROOT::VecOps::RVec<float>{K_gen_match[2], pi_gen_match[2]}",
            )
            .Define(
                "scale_genCharge",
                "ROOT::VecOps::RVec<int>{int(K_gen_match[3]), int(pi_gen_match[3])}",
            )
            .Define(
                "scale_hasGenMatch",
                "ROOT::VecOps::RVec<int>{int(K_gen_match[4]), int(pi_gen_match[4])}",
            )
            .Define("scale_muon_source", "ROOT::VecOps::RVec<int>{443, 443}")
            .Filter("K_gen_match[4] > 0.5 && pi_gen_match[4] > 0.5")
        )

        results.append(
            df.HistoBoost(
                "hD0_nom",
                d0_axes,
                coord_cols + ["weight"],
            )
        )

        if args.kinDiagnostics:
            # gen/reco pt and qopr of each daughter, plus gen/reco mRK and their ratio.
            # qopr = (reco qop)/(gen qop), with p = pt*cosh(eta), qop = charge/p.
            df = (
                df.Define("K_gen_pt", "K_gen_match[0]")
                .Define("pi_gen_pt", "pi_gen_match[0]")
                .Define("K_reco_p", "K_CVH_pt0 * std::cosh(K_CVH_eta0)")
                .Define("K_gen_p", "K_gen_match[0] * std::cosh(K_gen_match[1])")
                .Define("K_qop_reco", "K_charge0 / K_reco_p")
                .Define("K_qop_gen", "K_gen_match[3] / K_gen_p")
                .Define("K_qopr", "K_qop_reco / K_qop_gen")
                .Define("pi_reco_p", "pi_CVH_pt0 * std::cosh(pi_CVH_eta0)")
                .Define("pi_gen_p", "pi_gen_match[0] * std::cosh(pi_gen_match[1])")
                .Define("pi_qop_reco", "pi_charge0 / pi_reco_p")
                .Define("pi_qop_gen", "pi_gen_match[3] / pi_gen_p")
                .Define("pi_qopr", "pi_qop_reco / pi_qop_gen")
                # gen mRK: same form as the reco mRK column but from gen kinematics,
                # with the gen D0 mass = invariant mass of the gen K/pi four-vectors.
                # reco mRK is the existing analysis column "mRK".
                .Define("K_E_gen", f"std::sqrt(K_gen_p*K_gen_p + {M_K}*{M_K})")
                .Define("pi_E_gen", f"std::sqrt(pi_gen_p*pi_gen_p + {M_PI}*{M_PI})")
                .Define(
                    "D0_mass_gen",
                    "(ROOT::Math::PtEtaPhiMVector("
                    f"K_gen_match[0], K_gen_match[1], K_gen_match[2], {M_K}) + "
                    "ROOT::Math::PtEtaPhiMVector("
                    f"pi_gen_match[0], pi_gen_match[1], pi_gen_match[2], {M_PI})).M()",
                )
                .Define("mRK_gen", f"{M_K}*{M_K}*(pi_E_gen/K_E_gen)/D0_mass_gen")
                .Define("mRK_ratio", "mRK / mRK_gen")
                # gen/reco etaK, etaPi and the pion analogues of the mRK diagnostics
                # (mRpi = literal K<->pi swap). reco mRpi is the existing "mRpi" column.
                .Define("K_gen_eta", "K_gen_match[1]")
                .Define("pi_gen_eta", "pi_gen_match[1]")
                .Define("mRpi_gen", f"{M_PI}*{M_PI}*(K_E_gen/pi_E_gen)/D0_mass_gen")
                .Define("mRpi_ratio", "mRpi / mRpi_gen")
            )
            for name, col, axis in [
                ("hK_gen_pt", "K_gen_pt", axis_diag_pt),
                ("hK_reco_pt", "K_CVH_pt0", axis_diag_pt),
                ("hpi_gen_pt", "pi_gen_pt", axis_diag_pt),
                ("hpi_reco_pt", "pi_CVH_pt0", axis_diag_pt),
                ("hK_qopr", "K_qopr", axis_diag_qopr),
                ("hpi_qopr", "pi_qopr", axis_diag_qopr),
                ("hmRK_gen", "mRK_gen", axis_diag_mR),
                ("hmRK_reco", "mRK", axis_diag_mR),
                ("hmRK_ratio", "mRK_ratio", axis_diag_mR_ratio),
                ("hK_gen_eta", "K_gen_eta", axis_diag_eta),
                ("hK_reco_eta", "K_CVH_eta0", axis_diag_eta),
                ("hpi_gen_eta", "pi_gen_eta", axis_diag_eta),
                ("hpi_reco_eta", "pi_CVH_eta0", axis_diag_eta),
                ("hmRpi_gen", "mRpi_gen", axis_diag_mRpi),
                ("hmRpi_reco", "mRpi", axis_diag_mRpi),
                ("hmRpi_ratio", "mRpi_ratio", axis_diag_mR_ratio),
            ]:
                results.append(df.HistoBoost(name, [axis], [col, "weight"]))

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
                d0_axes,
                coord_cols + ["nominal_muonScaleSyst_responseWeights_tensor"],
                tensor_axes=data_jpsi_crctn_unc_helper.tensor_axes,
                storage=hist.storage.Double(),
            )
        )

        if build_manual_scale_variations:
            df = df.Define(
                "d0_scale_var",
                d0_scale_var_helper,
                [
                    "K_CVH_pt0",
                    "K_CVH_eta0",
                    "K_CVH_phi0",
                    "K_charge0",
                    "pi_CVH_pt0",
                    "pi_CVH_eta0",
                    "pi_CVH_phi0",
                    "pi_charge0",
                    "D0_mass",
                    "mRK",
                    "mRpi",
                ],
            )
            df = df.Define("d0_scale_var_mass", "d0_scale_var.mass")
            df = df.Define("d0_scale_var_mRK", "d0_scale_var.mRK")
            df = df.Define("d0_scale_var_mRpi", "d0_scale_var.mRpi")
            df = df.Define(
                "d0_scale_var_idx",
                f"wrem::d0::scale_var_indices({manual_scale_axes[0].size})",
            )
            df = df.Define(
                "d0_scale_updown",
                f"wrem::d0::updown_coords({manual_scale_axes[0].size})",
            )
            # Coordinate columns matching d0_axes: the varied observables (mRK, mRpi,
            # D0 mass) are per-variation RVecs; etaK/etaPi are fixed per-event scalars.
            if args.pionTemplateAxes:
                manual_coord_cols = [
                    "K_CVH_eta0",
                    "d0_scale_var_mRK",
                    "pi_CVH_eta0",
                    "d0_scale_var_mRpi",
                    "d0_scale_var_mass",
                ]
            else:
                manual_coord_cols = [
                    "K_CVH_eta0",
                    "d0_scale_var_mRK",
                    "d0_scale_var_mass",
                ]
            results.append(
                df.HistoBoost(
                    "nominal_muonScaleSyst_manual",
                    d0_axes + manual_scale_axes,
                    manual_coord_cols
                    + [
                        "d0_scale_var_idx",
                        "d0_scale_updown",
                        "nominal_weight",
                    ],
                    storage=hist.storage.Double(),
                )
            )

        df = muon_calibration.add_resolution_uncertainty(
            df,
            d0_axes,
            results,
            coord_cols,
            smearing_uncertainty_helper,
            "scale",
            storage_type=hist.storage.Double(),
            response_weight_col="scale_resolution_response_weight",
        )

    return results, weightsum


data_tree = args.treeName if args.treeName is not None else args.dataTreeName
mc_tree = args.treeName if args.treeName is not None else args.mcTreeName
resultdict = narf.build_and_run([datasets[0]], build_graph, event_tree=data_tree)
resultdict.update(narf.build_and_run([datasets[1]], build_graph, event_tree=mc_tree))
fout = f"{os.path.basename(__file__).replace('py', 'hdf5')}"
write_analysis_output(resultdict, fout, args, name_append=[args.era])
