#!/usr/bin/env bash
#SBATCH --job-name=e-to-m-comsol
#SBATCH --account=jordan_budhu
#SBATCH --partition=normal_q
#SBATCH --qos=tc_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-e-to-m-%j.out
#SBATCH --error=slurm-e-to-m-%j.out

# ==============================================================================
# TinkerCliffs COMSOL Slurm Driver: Electric-to-Magnetic Polarization Rotation
#
# Validates the anisotropic metasurface (reciprocal, lossless) transforming
# an interior electric dipole into an exterior magnetic dipole at 10 GHz (er=7 core)
# using COMSOL 6.1 with MUMPS direct solver on TinkerCliffs.
#
# Usage:
#   sbatch applications/anisotropic_gstc_sphere/comsol/runs/tinkercliffs_electric_to_magnetic_swept.sbatch
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="${SIE_MOM_ROOT:-/home/germanf/repo/sie_mom}"
cd "$PROJECT_ROOT"

# Load TinkerCliffs modules & Jordan Budhu group license server (1 RF seat)
export LM_LICENSE_FILE="1718@128.173.92.65"
module reset
module use /projects/jordan_budhu/modules/tinkercliffs-rome/all
module load comsol/6.1

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH=.

PRESET="${PRESET:-case12}"

# ------------------------------------------------------------------------------
# Presets Matching Convergence Benchmarks
# ------------------------------------------------------------------------------
case "$PRESET" in
    case4)
        SHELL_LAYERS=2
        SHELL_TAN_DIV="20.0"
        GLOBAL_DIV="15.0"
        INTERIOR_DIV="15.0"
        SOURCE_BALL_RADIUS="0.0"
        SOURCE_BALL_DIV="30.0"
        TAU_DIV="100.0"
        ;;
    case6)
        # Baseline: 3 swept prism layers, tan_div=25.0, glob_div=20.0 (~6 min solve)
        SHELL_LAYERS=3
        SHELL_TAN_DIV="25.0"
        GLOBAL_DIV="20.0"
        INTERIOR_DIV="20.0"
        SOURCE_BALL_RADIUS="0.0"
        SOURCE_BALL_DIV="30.0"
        TAU_DIV="100.0"
        ;;
    case7)
        # Layer study: identical to case6 but with 4 swept prism layers
        SHELL_LAYERS=4
        SHELL_TAN_DIV="25.0"
        GLOBAL_DIV="20.0"
        INTERIOR_DIV="20.0"
        SOURCE_BALL_RADIUS="0.0"
        SOURCE_BALL_DIV="30.0"
        TAU_DIV="100.0"
        ;;
    case8)
        # Source ball study: 4 swept layers + source ball (r=0.1 lambda0, div=30.0)
        SHELL_LAYERS=4
        SHELL_TAN_DIV="25.0"
        GLOBAL_DIV="20.0"
        INTERIOR_DIV="20.0"
        SOURCE_BALL_RADIUS="0.1"
        SOURCE_BALL_DIV="30.0"
        TAU_DIV="100.0"
        ;;
    case9)
        # Source ball resolution study: 4 swept layers + source ball (r=0.1 lambda0, div=60.0, hmax=0.5mm)
        SHELL_LAYERS=4
        SHELL_TAN_DIV="25.0"
        GLOBAL_DIV="20.0"
        INTERIOR_DIV="20.0"
        SOURCE_BALL_RADIUS="0.1"
        SOURCE_BALL_DIV="60.0"
        TAU_DIV="100.0"
        ;;
    case10)
        # Source ball ultra-refinement: 4 swept layers + source ball (r=0.1 lambda0, div=120.0, hmax=0.25mm)
        SHELL_LAYERS=4
        SHELL_TAN_DIV="25.0"
        GLOBAL_DIV="20.0"
        INTERIOR_DIV="20.0"
        SOURCE_BALL_RADIUS="0.1"
        SOURCE_BALL_DIV="120.0"
        TAU_DIV="100.0"
        ;;
    case11)
        # Metasurface tangential resolution study: tan_div=35.0 (2x surface triangles), 5 swept layers, r_src=0.1, div=60.0
        SHELL_LAYERS=5
        SHELL_TAN_DIV="35.0"
        GLOBAL_DIV="20.0"
        INTERIOR_DIV="20.0"
        SOURCE_BALL_RADIUS="0.1"
        SOURCE_BALL_DIV="60.0"
        TAU_DIV="100.0"
        ;;
    case12)
        # Shell thickness study: tau = lambda0 / 200 (0.15 mm), 5 swept layers, tan_div=35.0, r_src=0.1, div=60.0
        SHELL_LAYERS=5
        SHELL_TAN_DIV="35.0"
        GLOBAL_DIV="20.0"
        INTERIOR_DIV="20.0"
        SOURCE_BALL_RADIUS="0.1"
        SOURCE_BALL_DIV="60.0"
        TAU_DIV="200.0"
        ;;
    *)
        echo "Unknown PRESET '$PRESET'. Using custom settings or default case12."
        SHELL_LAYERS="${SHELL_LAYERS:-5}"
        SHELL_TAN_DIV="${SHELL_TAN_DIV:-35.0}"
        GLOBAL_DIV="${GLOBAL_DIV:-20.0}"
        INTERIOR_DIV="${INTERIOR_DIV:-20.0}"
        SOURCE_BALL_RADIUS="${SOURCE_BALL_RADIUS:-0.1}"
        SOURCE_BALL_DIV="${SOURCE_BALL_DIV:-60.0}"
        TAU_DIV="${TAU_DIV:-200.0}"
        ;;
esac

# At f0 = 10 GHz, lambda0 = c / 10 GHz = 0.0299792458 m
# air-extent = 0.95 * lambda0 = 0.02848028351 m
AIR_EXTENT_M="0.02848028351"
LAMBDA0_M="0.0299792458"
TAU_DIV="${TAU_DIV:-100.0}"
TAU_M=$(awk "BEGIN {printf \"%.17g\", $LAMBDA0_M / $TAU_DIV}")
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

if [ "$SOURCE_BALL_RADIUS" != "0.0" ] && [ "$SOURCE_BALL_RADIUS" != "0" ]; then
    RUN_NAME="bianiso_${PRESET}_shell${SHELL_TAN_DIV}_glob${GLOBAL_DIV}_lay${SHELL_LAYERS}_tau${TAU_DIV}_src${SOURCE_BALL_RADIUS}_div${SOURCE_BALL_DIV}_6.1_slurm${SLURM_JOB_ID}"
else
    RUN_NAME="bianiso_${PRESET}_shell${SHELL_TAN_DIV}_glob${GLOBAL_DIV}_lay${SHELL_LAYERS}_tau${TAU_DIV}_6.1_slurm${SLURM_JOB_ID}"
fi
OUT_DIR="$CASE_DIR/comsol/${RUN_NAME}"
mkdir -p "$OUT_DIR/logs"

echo "================================================================================"
echo "COMSOL 6.1 Anisotropic Validation: Electric to Magnetic Rotation (Job ${SLURM_JOB_ID})"
echo "Host             : $(hostname)"
echo "Date             : $(date --iso-8601=seconds)"
echo "Preset           : $PRESET"
echo "CPUs allocated   : $CORES"
echo "Tensors file     : $TENSORS"
echo "Mesh file        : $MESH"
echo "Configuration    : $CONFIG"
echo "Output Directory : $OUT_DIR"
echo "Shell Thickness  : $TAU_M m (lambda0 / $TAU_DIV)"
echo "================================================================================"

if [ ! -f "$TENSORS" ]; then
    echo "ERROR: Tensors file not found at $TENSORS"
    echo "Please scp the results directory from local machine to TinkerCliffs."
    exit 1
fi

# Execute Bianisotropic COMSOL Validation Pipeline
python -u applications/anisotropic_gstc_sphere/comsol/bianisotropic_shell_validation.py \
    --tensors "$TENSORS" \
    --mesh "$MESH" \
    --config "$CONFIG" \
    --output-dir "$OUT_DIR" \
    --version 6.1 \
    --cores "$CORES" \
    --linear-solver "$LINEAR_SOLVER" \
    --discretization "$DISCRETIZATION" \
    --shell-mesh swept \
    --shell-layers "$SHELL_LAYERS" \
    --shell-tangential-divisor "$SHELL_TAN_DIV" \
    --global-mesh-divisor "$GLOBAL_DIV" \
    --interior-hmax-divisor "$INTERIOR_DIV" \
    --source-ball-radius "$SOURCE_BALL_RADIUS" \
    --source-ball-hmax-divisor "$SOURCE_BALL_DIV" \
    --tau "$TAU_M" \
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
    print(f'Swept Prisms        : {mesh.get(\"prisms\", \"N/A\"):,}')
    print(f'Solve Duration      : {solver.get(\"solve_duration_seconds\", 0.0):.2f} s')
    print(f'Total Execution     : {cm.get(\"total_execution_seconds\", 0.0):.2f} s')
"
echo "================================================================================"
echo "COMSOL 6.1 Job ${SLURM_JOB_ID} finished successfully at $(date --iso-8601=seconds)"

