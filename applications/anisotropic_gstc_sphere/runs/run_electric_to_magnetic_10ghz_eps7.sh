#!/usr/bin/env bash
# ==============================================================================
# Pipeline Runner: Electric Dipole to Magnetic-Type Radiation Metasurface
# Case: 10 GHz, eps_r = 7 interior, reciprocal lossless general tensor
#
# Steps:
#   1. Run SIE solver with synthesized GSTC surface parameters.
#   2. Plot physical 2D field maps and 1D radial cuts (dB scale).
#   3. Plot spherical surface constitutive parameters (Y, Z, chi, condition number).
#
# Usage:
#   ./applications/anisotropic_gstc_sphere/runs/run_electric_to_magnetic_10ghz_eps7.sh [options]
#
# Options:
#   --plots-only   Skip SIE solve and regenerate all plots from existing data.
#   --force        Delete existing results and re-run SIE solve from scratch.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${APP_DIR}/../.." && pwd)"

# Ensure repository root is on PYTHONPATH
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Python environment
if [[ -f "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Error: Python 3 not found." >&2
    exit 1
fi

# Paths
CONFIG_FILE="${APP_DIR}/configurations/dipole_pol_rot/e2m_eps7.json"
MESH_FILE="${APP_DIR}/meshes/example_1_omega_matching_10ghz_eps7/sphere.obj"
RESULTS_DIR="${APP_DIR}/results/dipole_pol_rot/e2m_eps7"
if [ ! -d "${RESULTS_DIR}" ] && [ -d "${APP_DIR}/results/dipole_pol_rot/electric_to_magnetic_10ghz_eps7_reciprocal_lossless" ]; then
    RESULTS_DIR="${APP_DIR}/results/dipole_pol_rot/electric_to_magnetic_10ghz_eps7_reciprocal_lossless"
fi
PARAM_PLOTS_DIR="${RESULTS_DIR}/parameter_plots"

# Parse CLI flags
PLOTS_ONLY=false
FORCE=false

for arg in "$@"; do
    case "$arg" in
        --plots-only)
            PLOTS_ONLY=true
            ;;
        --force)
            FORCE=true
            ;;
        -h|--help)
            echo "Usage: $0 [--plots-only] [--force]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

echo "=================================================================="
echo " Polarization Rotation: Electric Dipole to Magnetic Radiation"
echo " Configuration: e2m_eps7"
echo " Results Directory: ${RESULTS_DIR}"
echo "=================================================================="

# ------------------------------------------------------------------------------
# Step 1: Run Spherical GSTC SIE Solver
# ------------------------------------------------------------------------------
if [[ "${PLOTS_ONLY}" == true ]]; then
    echo -e "\n---> [SKIPPED] Step 1: SIE Solve (--plots-only requested)"
elif [[ -f "${RESULTS_DIR}/batch_metrics.json" ]] && [[ "${FORCE}" != true ]]; then
    echo -e "\n---> [SKIPPED] Step 1: SIE Solve (results already exist; pass --force to re-solve)"
else
    if [[ "${FORCE}" == true ]] && [[ -d "${RESULTS_DIR}" ]]; then
        echo -e "\n---> [CLEAN] Removing existing results directory: ${RESULTS_DIR}"
        rm -rf "${RESULTS_DIR}"
    fi

    echo -e "\n---> [RUNNING] Step 1: Prepare Tensors and Solve Spherical GSTC SIE"
    "${PYTHON_BIN}" "${APP_DIR}/sie_gstc_sphere.py" \
        --config "${CONFIG_FILE}" \
        --run-dir "${RESULTS_DIR}" \
        --use-existing-mesh
    echo "---> [COMPLETED] Step 1: SIE Solve"
fi

# ------------------------------------------------------------------------------
# Step 2: Plot Physical Fields (2D Maps & 1D Radial Cuts)
# ------------------------------------------------------------------------------
echo -e "\n---> [RUNNING] Step 2: Plot Physical Fields (2D Maps & Radial Cuts)"

# 2a. Total |E| and exterior TE donut E_phi (20-98 percentile range)
"${PYTHON_BIN}" "${APP_DIR}/plot_fields.py" \
    --run-dir "${RESULTS_DIR}" \
    --case electric_incidence \
    --reference-transformation electric_to_magnetic \
    --map-components E E_phi \
    --map-scale db \
    --map-percentiles 20 98

# 2b. Interior TM dipole field E_theta (82-98 percentile range for dipole lobes)
"${PYTHON_BIN}" "${APP_DIR}/plot_fields.py" \
    --run-dir "${RESULTS_DIR}" \
    --case electric_incidence \
    --reference-transformation electric_to_magnetic \
    --map-components E_theta \
    --map-scale db \
    --map-percentiles 82 98

echo "---> [COMPLETED] Step 2: Field Plots"

# ------------------------------------------------------------------------------
# Step 3: Plot Spherical Surface Parameters (Y, Z, chi, condition number)
# ------------------------------------------------------------------------------
echo -e "\n---> [RUNNING] Step 3: Plot Constitutive Surface Parameters (Y, Z, chi)"

"${PYTHON_BIN}" "${APP_DIR}/plot_surface_parameters.py" \
    --tensors "${RESULTS_DIR}/face_tensors.bin" \
    --mesh "${MESH_FILE}" \
    --output-dir "${PARAM_PLOTS_DIR}" \
    --title-suffix "Electric to Magnetic (10 GHz, eps=7)" \
    --csv

echo "---> [COMPLETED] Step 3: Parameter Plots"

echo -e "\n=================================================================="
echo " Pipeline Finished Successfully!"
echo " Results and plots saved to:"
echo "   Fields & Plots:    ${RESULTS_DIR}/cases/electric_incidence/plots"
echo "   Surface Parameters: ${PARAM_PLOTS_DIR}"
echo "=================================================================="

