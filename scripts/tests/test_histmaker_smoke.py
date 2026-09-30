"""Smoke tests that actually build and run each histmaker's event loop.

These exist because the pure-Python tests in ``test_v_reweighting.py`` cannot
see a whole class of failure: a selection or a histogram booking that references
a column nothing defines fails only when RDataFrame just-in-time compiles the
graph, which needs real input files and a real event loop. An Upsilon selection
once cut on a column that had been moved out of the reweighting core, and no
unit test noticed.

They are slow (a minute or two each) and need the ntuples on disk, so they are
opt-in and skip themselves when either condition is not met::

    WREM_SMOKE_TESTS=1 python -m pytest scripts/tests/test_histmaker_smoke.py -v

Every case runs with ``--maxFiles 1`` and ``--kinDiagnostics``, because the
diagnostics path is the one that touches the most booking code.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DIMUON = REPO / "scripts" / "histmakers" / "dimuon_resonances_calinput.py"
D0 = REPO / "scripts" / "histmakers" / "d0_mass.py"

# One representative input per case. The histmakers glob directories, so the
# existence of the first file is enough to tell whether the sample is present.
JPSI_INPUT = Path("/scratch/submit/cms/emanca/jpsicor_data.root")
UPSILON_INPUT = Path("/scratch/submit/cms/emanca/upsilon_data.root")
D0_INPUT = Path("/scratch/submit/cms/emanca/D0Data_LayerJacobian_v5")

COMMON_DIMUON = [
    "--etaBins", "24",
    "--scale_A", "1.0",
    "--scale_e", "1.0",
    "--fitMuonScaleAndResolution",
    "--maxFiles", "1",
    "-j", "4",
]

CASES = [
    pytest.param(
        "jpsi",
        JPSI_INPUT,
        [
            sys.executable, str(DIMUON),
            "--resonance", "jpsi",
            "--triggers", "dimuon20_jpsi",
            "--vReweightSelection", "inclusive",
            "--makeVReweightInputs", "--kinDiagnostics",
            *COMMON_DIMUON,
        ],
        id="jpsi",
    ),
    pytest.param(
        # The Upsilon case is the point of this file: it is the only one whose
        # category axis is non-trivial, and the only one whose selection module
        # defines columns its category expression then cuts on.
        "upsilon",
        UPSILON_INPUT,
        [
            sys.executable, str(DIMUON),
            "--resonance", "upsilon",
            "--triggers", "inclusive",
            "--vReweightSelection", "muon_eta",
            "--makeVReweightInputs", "--kinDiagnostics",
            *COMMON_DIMUON,
        ],
        id="upsilon",
    ),
    pytest.param(
        "d0",
        D0_INPUT,
        [
            sys.executable, str(D0),
            "--era", "2016PostVFP",
            "--mcSelection", "dataLike",
            "--vReweightSelection", "inclusive",
            "--makeVReweightInputs", "--kinDiagnostics",
            "--maxFiles", "1", "-j", "4",
        ],
        id="d0",
    ),
]


@pytest.mark.skipif(
    os.environ.get("WREM_SMOKE_TESTS") != "1",
    reason="slow; set WREM_SMOKE_TESTS=1 to enable",
)
@pytest.mark.parametrize("label,required_input,command", CASES)
def test_histmaker_builds_and_runs(label, required_input, command, tmp_path):
    """The graph must compile and the event loop must complete."""

    if not required_input.exists():
        pytest.skip(f"input sample not present: {required_input}")

    result = subprocess.run(
        [*command, "-o", str(tmp_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    output = result.stdout + result.stderr
    # A just-in-time failure is the signature this file exists to catch, so name
    # it explicitly rather than leaving the reader to scan a long traceback.
    for marker in ("use of undeclared identifier", "error occurred during just-in-time"):
        assert marker not in output, (
            f"{label}: RDataFrame failed to compile the graph.\n"
            + "\n".join(
                line for line in output.splitlines() if "error" in line.lower()
            )[:2000]
        )
    assert result.returncode == 0, f"{label} exited {result.returncode}\n{output[-3000:]}"
    assert list(tmp_path.glob("*.hdf5")), f"{label} wrote no output file"
