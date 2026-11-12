#!/usr/bin/env bash
# ==============================================================================
# Pipeline Runner: Dual / Complementary Polarization Metasurface
# Case: 10 GHz, eps_r = 7 interior, reciprocal lossless general tensor
#
# Transformations:
#   1. Electric dipole (interior) -> Magnetic radiation (exterior)
#   2. Magnetic dipole (interior) -> Electric radiation (exterior)
#
# Solves both cases simultaneously in C++ batch mode:
#   - Case 1: electric_incidence
#   - Case 2: magnetic_incidence
#
# Usage:
#   ./applications/anisotropic_gstc_sphere/runs/run_2transf_10ghz_eps7.sh [options]
#
# Options:
#   --plots-only   Skip SIE solve and regenerate all plots from existing data.
#   --force        Delete existing results and re-run SIE solve from scratch.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${APP_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -f "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Error: Python 3 not found." >&2
    exit 1
fi

CONFIG_FILE="${APP_DIR}/configurations/dipole_pol_rot/dual_pol_eps7.json"
MESH_FILE="${APP_DIR}/meshes/example_1_omega_matching_10ghz_eps7/sphere_ppw6.obj"
RESULTS_DIR="${APP_DIR}/results/dipole_pol_rot/dual_pol_eps7"
if [ ! -d "${RESULTS_DIR}" ] && [ -d "${APP_DIR}/results/dipole_pol_rot/dual_pol_rot_10ghz_eps7_reciprocal_lossless" ]; then
    RESULTS_DIR="${APP_DIR}/results/dipole_pol_rot/dual_pol_rot_10ghz_eps7_reciprocal_lossless"
fi
PARAM_PLOTS_DIR="${RESULTS_DIR}/parameter_plots"

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
echo " Dual Complementary Polarization Rotation (10 GHz, eps=7)"
echo " Transformations: Electric <-> Magnetic Duality"
echo " Configuration: dual_pol_eps7"
echo " Results Directory: ${RESULTS_DIR}"
echo "=================================================================="

# ------------------------------------------------------------------------------
# Step 1: Run Spherical GSTC SIE Solver (Dual RHS Batch Solve)
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

    echo -e "\n---> [RUNNING] Step 1: Prepare Tensors and Solve Spherical GSTC SIE (Both Cases)"
    "${PYTHON_BIN}" "${APP_DIR}/sie_gstc_sphere.py" \
        --config "${CONFIG_FILE}" \
        --run-dir "${RESULTS_DIR}" \
        --use-existing-mesh
    echo "---> [COMPLETED] Step 1: Dual SIE Batch Solve"
fi

# ------------------------------------------------------------------------------
# Step 2: Plot Physical Fields for Both Cases
# ------------------------------------------------------------------------------
echo -e "\n---> [RUNNING] Step 2A: Plot Fields for Case 1 (electric_incidence: E_int -> M_ext)"
"${PYTHON_BIN}" "${APP_DIR}/plot_fields.py" \
    --run-dir "${RESULTS_DIR}" \
    --case electric_incidence \
    --reference-transformation electric_to_magnetic \
    --map-components E E_phi \
    --map-scale db \
    --map-percentiles 20 98

"${PYTHON_BIN}" "${APP_DIR}/plot_fields.py" \
    --run-dir "${RESULTS_DIR}" \
    --case electric_incidence \
    --reference-transformation electric_to_magnetic \
    --map-components E_theta \
    --map-scale db \
    --map-percentiles 82 98

echo -e "\n---> [RUNNING] Step 2B: Plot Fields for Case 2 (magnetic_incidence: M_int -> E_ext)"
"${PYTHON_BIN}" "${APP_DIR}/plot_fields.py" \
    --run-dir "${RESULTS_DIR}" \
    --case magnetic_incidence \
    --reference-transformation magnetic_to_electric \
    --map-components E E_theta \
    --map-scale db \
    --map-percentiles 20 98

"${PYTHON_BIN}" "${APP_DIR}/plot_fields.py" \
    --run-dir "${RESULTS_DIR}" \
    --case magnetic_incidence \
    --reference-transformation magnetic_to_electric \
    --map-components E_phi \
    --map-scale db \
    --map-percentiles 82 98

echo "---> [COMPLETED] Step 2: Field Plots for Both Simulation Cases"

# ------------------------------------------------------------------------------
# Step 3: Plot Constitutive Surface Parameters
# ------------------------------------------------------------------------------
echo -e "\n---> [RUNNING] Step 3: Plot Constitutive Surface Parameters"
"${PYTHON_BIN}" "${APP_DIR}/plot_surface_parameters.py" \
    --tensors "${RESULTS_DIR}/face_tensors.bin" \
    --mesh "${MESH_FILE}" \
    --output-dir "${PARAM_PLOTS_DIR}" \
    --title-suffix "Dual Rotations: E<->M (10 GHz, eps=7)" \
    --csv

echo "---> [COMPLETED] Step 3: Parameter Plots"

# ------------------------------------------------------------------------------
# Step 4: Evaluate & Plot Far-Field Radiation Patterns for Both Cases
# ------------------------------------------------------------------------------
echo -e "\n---> [RUNNING] Step 4: Far-Field Radiation Patterns (Both Cases)"
"${PYTHON_BIN}" "${APP_DIR}/plot_far_field.py" \
    --run-dir "${RESULTS_DIR}" \
    --cases electric_incidence magnetic_incidence

echo "---> [COMPLETED] Step 4: Far-Field Plots & Cuts"

echo -e "\n=================================================================="
echo " Dual Pipeline Finished Successfully!"
echo " Results and plots saved to:"
echo "   Case 1 Plots:       ${RESULTS_DIR}/cases/electric_incidence/plots"
echo "   Case 1 Far-Field:   ${RESULTS_DIR}/cases/electric_incidence/far_field"
echo "   Case 2 Plots:       ${RESULTS_DIR}/cases/magnetic_incidence/plots"
echo "   Case 2 Far-Field:   ${RESULTS_DIR}/cases/magnetic_incidence/far_field"
echo "   Surface Parameters: ${PARAM_PLOTS_DIR}"
echo "=================================================================="

