#!/usr/bin/env bash
#SBATCH --job-name=e2m-case-a
#SBATCH --account=jordan_budhu
#SBATCH --partition=normal_q
#SBATCH --qos=tc_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=230G
#SBATCH --time=03:45:00
#SBATCH --output=slurm-e2m-casea-%j.out
#SBATCH --error=slurm-e2m-casea-%j.out

# ==============================================================================
# TinkerCliffs COMSOL Slurm Driver: Case A (Zero-Thickness Sheet GSTC)
# Electric-to-Magnetic Dipole Polarization Rotation (a = 0.7 * lambda0, er = 7)
#
# Enforces two-sided GSTC jump conditions directly on r = a via SurfaceCurrent
# (scu1) + SurfaceMagneticCurrentDensity (smcd1) with broken-H(curl) up()/down()
# field traces, solved using COMSOL 6.1 and the MUMPS direct solver.
#
# Usage:
#   sbatch applications/anisotropic_gstc_sphere/comsol/runs/tinkercliffs_electric_to_magnetic_case_a.sbatch
#   PRESET=case2 sbatch applications/anisotropic_gstc_sphere/comsol/runs/tinkercliffs_electric_to_magnetic_case_a.sbatch
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="${SIE_MOM_ROOT:-/home/germanf/repo/sie_mom}"
cd "$PROJECT_ROOT"

COMSOL_VERSION="${COMSOL_VERSION:-6.1}"

# Load TinkerCliffs modules & License server
if [ "$COMSOL_VERSION" = "5.4" ]; then
    # Campus license server for COMSOL 5.4 (5 seats)
    export LM_LICENSE_FILE="1718@comsol.software.vt.edu"
    module reset
    module load COMSOL/5.4.0.388
else
    # Jordan Budhu group license server for COMSOL 6.1 (1 RF seat)
    export LM_LICENSE_FILE="1718@128.173.92.65"
    module reset
    module use /projects/jordan_budhu/modules/tinkercliffs-rome/all
    module load comsol/6.1
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH=.

PRESET="${PRESET:-case1}"

# ------------------------------------------------------------------------------
# Controlled Mesh Convergence Presets (Case A Zero-Thickness Sheet)
# ------------------------------------------------------------------------------
case "$PRESET" in
    case1)
        # Baseline Case A: matches Case B case9 bulk/surface resolution without volumetric shell
        SHEET_TAN_DIV="25.0"
        GLOBAL_DIV="20.0"
        INTERIOR_DIV="20.0"
        INNER_SHEET_BAND_WIDTH="0.0"
        INNER_SHEET_BAND_DIV="0.0"
        OUTER_SHEET_BAND_WIDTH="0.0"
        OUTER_SHEET_BAND_DIV="0.0"
        SOURCE_BALL_RADIUS="0.1"
        SOURCE_BALL_DIV="60.0"
        FF_OFFSET_LAM="0.10"
        ;;
    case2)
        # Conformal Band Refinement: 1:1 aspect ratio volume tetrahedra at r = a
        # Preserves stable source ball (0.1 / 60) while refining sheet and adjacent bands
        SHEET_TAN_DIV="30.0"
        GLOBAL_DIV="20.0"
        INTERIOR_DIV="20.0"
        INNER_SHEET_BAND_WIDTH="0.08"
        INNER_SHEET_BAND_DIV="30.0"
        OUTER_SHEET_BAND_WIDTH="0.10"
        OUTER_SHEET_BAND_DIV="30.0"
        SOURCE_BALL_RADIUS="0.1"
        SOURCE_BALL_DIV="60.0"
        FF_OFFSET_LAM="0.10"
        ;;
    case3)
        # Ultra-fine conformal band study (tan_div=35.0, in/out band=35.0)
        SHEET_TAN_DIV="35.0"
        GLOBAL_DIV="20.0"
        INTERIOR_DIV="20.0"
        INNER_SHEET_BAND_WIDTH="0.08"
        INNER_SHEET_BAND_DIV="35.0"
        OUTER_SHEET_BAND_WIDTH="0.10"
        OUTER_SHEET_BAND_DIV="35.0"
        SOURCE_BALL_RADIUS="0.1"
        SOURCE_BALL_DIV="60.0"
        FF_OFFSET_LAM="0.10"
        ;;
    *)
        echo "Unknown PRESET '$PRESET'. Using custom environment settings or case1 defaults."
        SHEET_TAN_DIV="${SHEET_TAN_DIV:-25.0}"
        GLOBAL_DIV="${GLOBAL_DIV:-20.0}"
        INTERIOR_DIV="${INTERIOR_DIV:-20.0}"
        INNER_SHEET_BAND_WIDTH="${INNER_SHEET_BAND_WIDTH:-0.0}"
        INNER_SHEET_BAND_DIV="${INNER_SHEET_BAND_DIV:-0.0}"
        OUTER_SHEET_BAND_WIDTH="${OUTER_SHEET_BAND_WIDTH:-0.0}"
        OUTER_SHEET_BAND_DIV="${OUTER_SHEET_BAND_DIV:-0.0}"
        SOURCE_BALL_RADIUS="${SOURCE_BALL_RADIUS:-0.1}"
        SOURCE_BALL_DIV="${SOURCE_BALL_DIV:-60.0}"
        FF_OFFSET_LAM="${FF_OFFSET_LAM:-0.10}"
        ;;
esac

# At f0 = 10 GHz, lambda0 = c / 10 GHz = 0.0299792458 m
# a = 0.7 * lambda0 = 0.020985472 m, air-extent b = 0.95 * lambda0 = 0.02848028351 m
AIR_EXTENT_M="${AIR_EXTENT_M:-0.02848028351}"
LINEAR_SOLVER="${LINEAR_SOLVER:-mumps}"
DISCRETIZATION="${DISCRETIZATION:-linear}"
CORES="${SLURM_CPUS_PER_TASK:-64}"

RADIAL_POINTS="${RADIAL_POINTS:-300}"
RADIAL_MAX_RADII="${RADIAL_MAX_RADII:-2.0}"
ZX_NX="${ZX_GRID:-200}"
ZX_NZ="${ZX_GRID:-200}"
ZX_SPAN_RADII="${ZX_SPAN_RADII:-2.0}"

CASE_DIR="applications/anisotropic_gstc_sphere/results/dipole_pol_rot/e2m_eps7"
if [ ! -d "$CASE_DIR" ]; then
    CASE_DIR="applications/anisotropic_gstc_sphere/results/dipole_pol_rot/electric_to_magnetic_10ghz_eps7_reciprocal_lossless"
fi
TENSORS="$CASE_DIR/face_tensors.bin"
MESH="applications/anisotropic_gstc_sphere/meshes/example_1_omega_matching_10ghz_eps7/sphere_ppw6.obj"
CONFIG="applications/anisotropic_gstc_sphere/configurations/dipole_pol_rot/e2m_eps7.json"
SIE_DIR="$CASE_DIR/cases/electric_incidence"

BAND_TAG=""
EXTRA_BAND_ARGS=()
if [ "${INNER_SHEET_BAND_WIDTH:-0.0}" != "0.0" ] && [ "${INNER_SHEET_BAND_WIDTH:-0.0}" != "0" ]; then
    BAND_TAG="${BAND_TAG}_inband${INNER_SHEET_BAND_DIV}"
    EXTRA_BAND_ARGS+=(--inner-sheet-band-width "$INNER_SHEET_BAND_WIDTH" --inner-sheet-band-hmax-divisor "$INNER_SHEET_BAND_DIV")
fi
if [ "${OUTER_SHEET_BAND_WIDTH:-0.0}" != "0.0" ] && [ "${OUTER_SHEET_BAND_WIDTH:-0.0}" != "0" ]; then
    BAND_TAG="${BAND_TAG}_outband${OUTER_SHEET_BAND_DIV}"
    EXTRA_BAND_ARGS+=(--outer-sheet-band-width "$OUTER_SHEET_BAND_WIDTH" --outer-sheet-band-hmax-divisor "$OUTER_SHEET_BAND_DIV")
fi

if [ "$SOURCE_BALL_RADIUS" != "0.0" ] && [ "$SOURCE_BALL_RADIUS" != "0" ]; then
    RUN_NAME="casea_${PRESET}_sheet${SHEET_TAN_DIV}${BAND_TAG}_glob${GLOBAL_DIV}_int${INTERIOR_DIV}_ff${FF_OFFSET_LAM}_src${SOURCE_BALL_RADIUS}_div${SOURCE_BALL_DIV}_${COMSOL_VERSION}_slurm${SLURM_JOB_ID}"
else
    RUN_NAME="casea_${PRESET}_sheet${SHEET_TAN_DIV}${BAND_TAG}_glob${GLOBAL_DIV}_int${INTERIOR_DIV}_ff${FF_OFFSET_LAM}_${COMSOL_VERSION}_slurm${SLURM_JOB_ID}"
fi
OUT_DIR="$CASE_DIR/comsol/${RUN_NAME}"
mkdir -p "$OUT_DIR/logs"

echo "================================================================================"
echo "COMSOL $COMSOL_VERSION Case A (Zero-Thickness Sheet) Validation: Electric to Magnetic (Job ${SLURM_JOB_ID})"
echo "Host             : $(hostname)"
echo "Date             : $(date --iso-8601=seconds)"
echo "Preset           : $PRESET"
echo "CPUs allocated   : $CORES"
echo "Linear Solver    : $LINEAR_SOLVER"
echo "Tensors file     : $TENSORS"
echo "Mesh file        : $MESH"
echo "Configuration    : $CONFIG"
echo "Output Directory : $OUT_DIR"
echo "Sheet Tan Div    : $SHEET_TAN_DIV (Huygens FF offset = ${FF_OFFSET_LAM} * lambda0)"
echo "================================================================================"

if [ ! -f "$TENSORS" ]; then
    echo "ERROR: Tensors file not found at $TENSORS"
    echo "Please ensure SIE results are generated or synced to TinkerCliffs."
    exit 1
fi

# Execute Bianisotropic Case A COMSOL Validation Pipeline
python -u applications/anisotropic_gstc_sphere/comsol/bianisotropic_sheet_validation.py \
    --tensors "$TENSORS" \
    --mesh "$MESH" \
    --config "$CONFIG" \
    --output-dir "$OUT_DIR" \
    --version "$COMSOL_VERSION" \
    --cores "$CORES" \
    --linear-solver "$LINEAR_SOLVER" \
    --discretization "$DISCRETIZATION" \
    --sheet-tangential-divisor "$SHEET_TAN_DIV" \
    --global-mesh-divisor "$GLOBAL_DIV" \
    --interior-hmax-divisor "$INTERIOR_DIV" \
    --source-ball-radius "$SOURCE_BALL_RADIUS" \
    --source-ball-hmax-divisor "$SOURCE_BALL_DIV" \
    --far-field-offset-lambda "$FF_OFFSET_LAM" \
    "${EXTRA_BAND_ARGS[@]}" \
    --air-extent "$AIR_EXTENT_M" \
    --radial-points "$RADIAL_POINTS" \
    --radial-max-radii "$RADIAL_MAX_RADII" \
    --radial-theta 45.0 \
    --radial-phi 0.0 \
    --zx-nx "$ZX_NX" \
    --zx-nz "$ZX_NZ" \
    --zx-span-radii "$ZX_SPAN_RADII"

echo ""
echo "--- Running Benchmark Comparison Against SIE Reference ---"
if [ -d "$SIE_DIR" ]; then
    if [ ! -f "$SIE_DIR/far_field/far_field_cuts.csv" ] && [ ! -f "$SIE_DIR/far_field_cuts.csv" ]; then
        echo "Generating missing SIE reference far-field cuts..."
        python -u applications/anisotropic_gstc_sphere/plot_far_field.py \
            --run-dir "$CASE_DIR" \
            --cases electric_incidence || true
    fi
    python -u applications/anisotropic_gstc_sphere/comsol/compare_sie_comsol.py \
        --sie-dir "$SIE_DIR" \
        --comsol-dir "$OUT_DIR" || true
fi

echo ""
echo "--- Generating Physical Field Comparison Plots via plot_fields.py (Spherical Basis) ---"
# 1. Full spherical component suite (norms and all spherical vector components)
MPLCONFIGDIR=/tmp/mpl python -u applications/anisotropic_gstc_sphere/plot_fields.py \
    --run-dir "$OUT_DIR" \
    --case electric_incidence \
    --reference-transformation electric_to_magnetic \
    --line-basis spherical \
    --map-basis spherical \
    --map-percentiles 20 98 \
    --magnitude-scale db || true

# 2. Dedicated mode rotation comparison (E_theta TM mode vs E_phi TE mode)
MPLCONFIGDIR=/tmp/mpl python -u applications/anisotropic_gstc_sphere/plot_fields.py \
    --run-dir "$OUT_DIR" \
    --case electric_incidence \
    --reference-transformation electric_to_magnetic \
    --line-basis spherical \
    --map-basis spherical \
    --line-components E_theta E_phi \
    --map-components E_theta E_phi \
    --map-percentiles 20 98 \
    --magnitude-scale db || true

echo ""
echo "================================================================================"
echo "SUMMARY METRICS:"
echo "================================================================================"
python -c "
import json
from pathlib import Path

out_dir = Path('$OUT_DIR')
comsol_metrics_path = out_dir / 'comsol_metrics.json'

if comsol_metrics_path.is_file():
    with comsol_metrics_path.open() as f:
        cm = json.load(f)
    mesh = cm.get('mesh', {})
    solver = cm.get('solver', {})
    print(f'Total Mesh Elements : {mesh.get(\"total_elements\", \"N/A\"):,}')
    print(f'Sheet Triangles     : {mesh.get(\"sheet_triangles\", \"N/A\"):,}')
    print(f'Volume Tetrahedra   : {mesh.get(\"tetrahedra\", \"N/A\"):,}')
    print(f'Solve Duration      : {solver.get(\"solve_duration_seconds\", 0.0):.2f} s')
    print(f'Total Execution     : {cm.get(\"total_execution_seconds\", 0.0):.2f} s')
"
echo "================================================================================"
echo "COMSOL 6.1 Case A Job ${SLURM_JOB_ID} finished successfully at $(date --iso-8601=seconds)"

