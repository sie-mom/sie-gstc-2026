#!/usr/bin/env bash
# ==============================================================================
# Publication Figure Runner: Example 2 Part A
# Case: Electric Dipole -> Magnetic Radiation Metasurface (10 GHz, eps_r = 7)
#
# Generates IEEE-style publication figures:
#   1. figure_1_part_a_2d_field_maps.pdf / .png (2D |E_theta| vs |E_phi| in dB)
#   2. figure_2_part_a_radial_cut_transformation.pdf / .png (Target vs SIE vs SVWF)
#   3. surface_parameters/ (Im(Y), Im(Z), Re(chi), condition number in PDF/PNG/CSV)
#
# Usage:
#   ./applications/anisotropic_gstc_sphere/publication/run_example_2_part_a.sh [options]
#
# Options:
#   --plots-only        Skip SIE solve and generate plots from existing results.
#   --force             Delete existing results and re-run SIE solve from scratch.
#   --skip-parameters   Skip metasurface constitutive parameter plots.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${APP_DIR}/../.." && pwd)"

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

CONFIG_FILE="${APP_DIR}/configurations/dipole_pol_rot/e2m_eps7.json"
RESULTS_DIR="${APP_DIR}/results/dipole_pol_rot/e2m_eps7"
if [ ! -d "${RESULTS_DIR}" ] && [ -d "${APP_DIR}/results/dipole_pol_rot/electric_to_magnetic_10ghz_eps7_reciprocal_lossless" ]; then
    RESULTS_DIR="${APP_DIR}/results/dipole_pol_rot/electric_to_magnetic_10ghz_eps7_reciprocal_lossless"
fi
OUTPUT_DIR="${SCRIPT_DIR}/figures"

PLOTS_ONLY=false
FORCE=false
SKIP_PARAMETERS=false

for arg in "$@"; do
    case "$arg" in
        --plots-only)
            PLOTS_ONLY=true
            ;;
        --force)
            FORCE=true
            ;;
        --skip-parameters)
            SKIP_PARAMETERS=true
            ;;
        -h|--help)
            echo "Usage: $0 [--plots-only] [--force] [--skip-parameters]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

# Resolve configured mesh path
MESH_REL_PATH="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${CONFIG_FILE}'))['sie']['mesh']['path'])")"
if [[ "${MESH_REL_PATH}" = /* ]]; then
    MESH_FULL_PATH="${MESH_REL_PATH}"
else
    MESH_FULL_PATH="${APP_DIR}/${MESH_REL_PATH}"
fi

echo "=================================================================="
echo " Publication Figures: Example 2 Part A (Electric to Magnetic)"
echo " Configuration: e2m_eps7"
echo " Mesh:          ${MESH_FULL_PATH}"
echo " Results Directory: ${RESULTS_DIR}"
echo " Figures Directory: ${OUTPUT_DIR}"
echo "=================================================================="

# Step 1: Ensure SIE solve is present
if [[ "${PLOTS_ONLY}" == true ]]; then
    echo -e "\n---> [SKIPPED] Step 1: SIE Solve (--plots-only requested)"
elif [[ -f "${RESULTS_DIR}/batch_metrics.json" ]] && [[ "${FORCE}" != true ]]; then
    echo -e "\n---> [SKIPPED] Step 1: SIE Solve (results exist; pass --force to re-solve)"
else
    if [[ "${FORCE}" == true ]] && [[ -d "${RESULTS_DIR}" ]]; then
        echo -e "\n---> [CLEAN] Removing existing results directory: ${RESULTS_DIR}"
        rm -rf "${RESULTS_DIR}"
    fi

    MESH_OPT=""
    if [[ -f "${MESH_FULL_PATH}" ]]; then
        MESH_OPT="--use-existing-mesh"
    fi

    echo -e "\n---> [RUNNING] Step 1: Solve Spherical GSTC SIE"
    "${PYTHON_BIN}" "${APP_DIR}/sie_gstc_sphere.py" \
        --config "${CONFIG_FILE}" \
        --run-dir "${RESULTS_DIR}" \
        ${MESH_OPT}
fi

# Step 2: Ensure SVWF validation dataset is present
SVWF_CSV="${RESULTS_DIR}/cases/electric_incidence/fields/svwf_radial.csv"
if [[ ! -f "${SVWF_CSV}" ]]; then
    echo -e "\n---> [RUNNING] Step 2: Generating SVWF Collocation Benchmark Data"
    "${PYTHON_BIN}" -c "
import numpy as np
import pandas as pd
from pathlib import Path
from applications.anisotropic_gstc_sphere.workflow import load_workflow_configuration
from sie_mom.gstc import extract_reciprocal_lossless_least_squares, form_boundary_quantities
from sie_mom.excitation import evaluate_dipole_fields
from spherical_expansion.solver import solve_collocation_general
from spherical_expansion.svwf_sphere import compute_fields
from spherical_expansion.coefficients import save_coefficients
from sie_mom.postprocessing import load_field_csv

cfg = load_workflow_configuration(Path('${CONFIG_FILE}'))
case = cfg.case
a = case.radius_m

th = np.array([np.deg2rad(45.0)])
ph = np.array([0.0])

E_int, H_int = evaluate_dipole_fields(case.k_interior, a, th, ph, case.eta_interior, effective_moment=(0.0, 0.0, 1.0), dipole_type='electric')
E_ext, H_ext = evaluate_dipole_fields(case.k_exterior, a, th, ph, case.eta_exterior, effective_moment=(0.0, 0.0, 11.386035931884505), dipole_type='magnetic')

bq = form_boundary_quantities(E_int[1:].conj(), H_int[1:].conj(), E_ext[1:].conj(), H_ext[1:].conj())
res_ext = extract_reciprocal_lossless_least_squares([bq], reference_impedance=case.eta_exterior)
params_svwf = res_ext.parameters

tensors = {
    'Y': params_svwf.Y_ee[:, :, 0],
    'Chi': params_svwf.Y_em[:, :, 0],
    'Gamma': params_svwf.Y_me[:, :, 0],
    'Z': params_svwf.Y_mm[:, :, 0],
}

def incident_field(theta, phi):
    E_inc, H_inc = evaluate_dipole_fields(case.k_interior, a, theta, phi, case.eta_interior, effective_moment=(0.0, 0.0, 1.0), dipole_type='electric')
    return E_inc[1].conj(), E_inc[2].conj(), H_inc[1].conj(), H_inc[2].conj()

res = solve_collocation_general(
    n_max=8, k0=case.k_exterior, eta0=case.eta_exterior,
    k1=case.k_interior, eta1=case.eta_interior, radius=a,
    gstc_params=tensors, n_theta=36, n_phi=72, return_metrics=True,
    incident_field_func=incident_field, is_interior_source=True,
)
ext_m, ext_n, int_m, int_n, _ = res

case_dir = Path('${RESULTS_DIR}') / 'cases' / 'electric_incidence'
fields_dir = case_dir / 'fields'
save_coefficients(case_dir / 'svwf_coefficients.npz', ext_m, ext_n, int_m, int_n)

data_rad = load_field_csv(fields_dir / 'sie_physical_radial.csv')
x_r, y_r, z_r = data_rad['x'], data_rad['y'], data_rad['z']
r_pts = np.sqrt(x_r**2 + y_r**2 + z_r**2)
th_r = np.arccos(np.clip(z_r / r_pts, -1.0, 1.0))
ph_r = np.arctan2(y_r, x_r)
r_norm = r_pts / a

int_mask = r_norm < 1.0
ext_mask = r_norm >= 1.0

E_svwf_sph = np.zeros((3, len(r_pts)), dtype=complex)
H_svwf_sph = np.zeros((3, len(r_pts)), dtype=complex)

E_ext_svwf, H_ext_svwf = compute_fields(case.k_exterior, r_pts[ext_mask], th_r[ext_mask], ph_r[ext_mask], ext_m, ext_n, kind=3, eta=case.eta_exterior, grid=False)
E_svwf_sph[:, ext_mask] = np.conjugate(np.asarray(E_ext_svwf))
H_svwf_sph[:, ext_mask] = np.conjugate(np.asarray(H_ext_svwf))

E_inc_int, H_inc_int = evaluate_dipole_fields(case.k_interior, r_pts[int_mask], th_r[int_mask], ph_r[int_mask], case.eta_interior, effective_moment=(0.0, 0.0, 1.0), dipole_type='electric')
E_refl_int, H_refl_int = compute_fields(case.k_interior, r_pts[int_mask], th_r[int_mask], ph_r[int_mask], int_m, int_n, kind=1, eta=case.eta_interior, grid=False)
E_svwf_sph[:, int_mask] = np.conjugate(np.conjugate(E_inc_int) + np.asarray(E_refl_int))
H_svwf_sph[:, int_mask] = np.conjugate(np.conjugate(H_inc_int) + np.asarray(H_refl_int))

def sph_to_cart(v_sph, th, ph):
    vr, vth, vph = v_sph[0], v_sph[1], v_sph[2]
    sin_th, cos_th = np.sin(th), np.cos(th)
    sin_ph, cos_ph = np.sin(ph), np.cos(ph)
    vx = vr * sin_th * cos_ph + vth * cos_th * cos_ph - vph * sin_ph
    vy = vr * sin_th * sin_ph + vth * cos_th * sin_ph + vph * cos_ph
    vz = vr * cos_th - vth * sin_th
    return vx, vy, vz

Ex, Ey, Ez = sph_to_cart(E_svwf_sph, th_r, ph_r)
Hx, Hy, Hz = sph_to_cart(H_svwf_sph, th_r, ph_r)

df_out = pd.DataFrame({
    'point_index': np.arange(len(r_pts)),
    'x': x_r, 'y': y_r, 'z': z_r,
    'region': data_rad['region'].astype(int),
    'Ex_real': Ex.real, 'Ex_imag': Ex.imag,
    'Ey_real': Ey.real, 'Ey_imag': Ey.imag,
    'Ez_real': Ez.real, 'Ez_imag': Ez.imag,
    'Hx_real': Hx.real, 'Hx_imag': Hx.imag,
    'Hy_real': Hy.real, 'Hy_imag': Hy.imag,
    'Hz_real': Hz.real, 'Hz_imag': Hz.imag,
})
df_out.to_csv(fields_dir / 'svwf_radial.csv', index=False)
print('SVWF radial benchmark generated successfully.')
"
fi

# Step 3: Generate publication figures
echo -e "\n---> [RUNNING] Step 3: Generate Publication Figures (PDF & PNG)"
"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_example_2_part_a.py" \
    --config "${CONFIG_FILE}" \
    --results-dir "${RESULTS_DIR}" \
    --case "electric_incidence" \
    --output-dir "${OUTPUT_DIR}"

# Step 4: Generate metasurface constitutive parameter plots (Y, Z, chi, condition number)
if [[ "${SKIP_PARAMETERS}" != true ]]; then
    PARAM_OUTPUT_DIR="${OUTPUT_DIR}/surface_parameters"
    mkdir -p "${PARAM_OUTPUT_DIR}"
    if [[ -f "${RESULTS_DIR}/face_tensors.bin" ]] && [[ -f "${MESH_FULL_PATH}" ]]; then
        echo -e "\n---> [RUNNING] Step 4: Plot Metasurface Constitutive Parameters (Y, Z, chi, cond)"
        "${PYTHON_BIN}" "${APP_DIR}/plot_surface_parameters.py" \
            --tensors "${RESULTS_DIR}/face_tensors.bin" \
            --mesh "${MESH_FULL_PATH}" \
            --output-dir "${PARAM_OUTPUT_DIR}" \
            --title-suffix "Part A: Electric to Magnetic (10 GHz, eps=7)" \
            --csv
        echo "---> [COMPLETED] Step 4: Surface parameters saved to ${PARAM_OUTPUT_DIR}"
    fi
fi

echo -e "\n=================================================================="
echo " Figures and Postprocessing Generated Successfully!"
echo " Output files in ${OUTPUT_DIR}:"
ls -la "${OUTPUT_DIR}"
if [[ -d "${OUTPUT_DIR}/surface_parameters" ]]; then
    echo -e "\n Surface parameter figures in ${OUTPUT_DIR}/surface_parameters:"
    ls -la "${OUTPUT_DIR}/surface_parameters"
fi
echo "=================================================================="

