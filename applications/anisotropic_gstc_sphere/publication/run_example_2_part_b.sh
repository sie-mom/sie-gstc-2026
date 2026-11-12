#!/usr/bin/env bash
# ==============================================================================
# Publication Figure Runner: Example 2 Part B
# Case: Dual Polarization Rotator Metasurface (10 GHz, eps_r = 7)
#       Simultaneous Electric -> Magnetic & Magnetic -> Electric transformations
#
# Generates IEEE-style publication figures:
#   1. figure_1_part_b_case1_electric_incidence_2d_field_maps.pdf / .png
#   2. figure_1_part_b_case2_magnetic_incidence_2d_field_maps.pdf / .png
#   3. figure_1_part_b_combined_2d_field_maps.pdf / .png (4-panel dual comparison)
#   4. figure_2_part_b_case1_electric_incidence_radial_cut.pdf / .png
#   5. figure_2_part_b_case2_magnetic_incidence_radial_cut.pdf / .png
#   6. figure_2_part_b_radial_cut_transformation.pdf / .png (2-panel dual comparison)
#   7. surface_parameters_part_b/ (Im(Y), Im(Z), Re(chi), condition number in PDF/PNG/CSV)
#
# Usage:
#   ./applications/anisotropic_gstc_sphere/publication/run_example_2_part_b.sh [options]
#
# Options:
#   --plots-only        Skip SIE solve and generate plots from existing results.
#   --force             Delete existing results and re-run SIE solve from scratch.
#   --skip-parameters   Skip metasurface constitutive parameter plots.
#   --vmin-db <val>     Colorbar minimum in normalized dB (default: -112.0).
#   --vmax-db <val>     Colorbar maximum in normalized dB (default: -58.0).
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

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_example_2_part_b.py" "$@"

