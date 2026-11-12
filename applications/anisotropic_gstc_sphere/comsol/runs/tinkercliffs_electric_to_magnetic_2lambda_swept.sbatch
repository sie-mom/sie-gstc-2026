#!/usr/bin/env bash
#SBATCH --job-name=e2m-2lam
#SBATCH --account=jordan_budhu
#SBATCH --partition=normal_q
#SBATCH --qos=tc_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-e2m-2lam-%j.out
#SBATCH --error=slurm-e2m-2lam-%j.out

# ==============================================================================
# TinkerCliffs COMSOL Slurm Driver: 2*lambda0 Electric-to-Magnetic Metasurface
#
# Validates the anisotropic metasurface (reciprocal, lossless) transforming
# an interior electric dipole into an exterior magnetic dipole at 10 GHz (er=7 core)
# for radius a = 2*lambda0 (~60 mm) using COMSOL 6.1 on TinkerCliffs normal_q.
#
# Memory Sizing:
#   At a = 2*lambda0, the volume is ~23x larger than a = 0.7*lambda0.
#   By using the localized source ball (r=0.1*lambda0, div=60) for the dipole,
#   the bulk core mesh is safely set to 10 elements/wavelength (INTERIOR_DIV=10.0),
#   keeping the total mesh under 3.5M elements and peak RAM under 70 GB.
#   Runs safely on normal_q (120 GB node) in ~6-8 minutes without high-mem queue!
#
# Usage:
#   sbatch applications/anisotropic_gstc_sphere/comsol/runs/tinkercliffs_electric_to_magnetic_2lambda_swept.sbatch
#   PRESET=case1 sbatch applications/anisotropic_gstc_sphere/comsol/runs/tinkercliffs_electric_to_magnetic_2lambda_swept.sbatch
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

PRESET="${PRESET:-case3}"

# ------------------------------------------------------------------------------
# Presets Sized for standard normal_q nodes (120 GB limit, ~6-8 min runtime)
# ------------------------------------------------------------------------------
case "$PRESET" in
    case1)
        # Full 10 PPW: 3 swept prism layers, tan_div=20.0, glob_div=10.0, int_div=10.0
        # ~2.8M elements, peak MUMPS RAM ~106 GB (solves in ~12 min)
        SHELL_LAYERS=3
        SHELL_TAN_DIV=20
        GLOBAL_DIV="10.0"
        INTERIOR_DIV="10.0"
        SOURCE_BALL_RADIUS="0.1"
        SOURCE_BALL_DIV="60.0"
        TAU_DIV="100.0"
        ;;
    case2)
        # Higher Shell Tangential Resolution: 3 swept layers, tan_div=25, glob_div=10.0, int_div=10.0
        # ~3.5M elements, peak MUMPS RAM ~135-145 GB (safe within 200 GB allocation)
        SHELL_LAYERS=3
        SHELL_TAN_DIV=25
        GLOBAL_DIV="10.0"
        INTERIOR_DIV="10.0"
        SOURCE_BALL_RADIUS="0.1"
        SOURCE_BALL_DIV="60.0"
        TAU_DIV="100.0"
        ;;
    case3)
        # Thicker shell: tau = lambda0 / 50.0 (0.60 mm), 3 swept layers, tan_div=25, glob_div=10.0, int_div=10.0
        # Halves effective shell permittivity (reduces stiffness of hyperbolic medium)
        SHELL_LAYERS=3
        SHELL_TAN_DIV=25
        GLOBAL_DIV="10.0"
        INTERIOR_DIV="10.0"
        SOURCE_BALL_RADIUS="0.1"
        SOURCE_BALL_DIV="60.0"
        TAU_DIV="50.0"
        ;;
    case4)
        # Thinner shell study: tau = lambda0 / 200, 4 swept layers
        SHELL_LAYERS=4
        SHELL_TAN_DIV="20.0"
        GLOBAL_DIV="10.0"
        INTERIOR_DIV="10.0"
        SOURCE_BALL_RADIUS="0.1"
        SOURCE_BALL_DIV="60.0"
        TAU_DIV="200.0"
        ;;
    *)
        echo "Unknown PRESET '$PRESET'. Using custom settings or default case1."
        SHELL_LAYERS="${SHELL_LAYERS:-3}"
        SHELL_TAN_DIV="${SHELL_TAN_DIV:-20.0}"
        GLOBAL_DIV="${GLOBAL_DIV:-10.0}"
        INTERIOR_DIV="${INTERIOR_DIV:-10.0}"
        SOURCE_BALL_RADIUS="${SOURCE_BALL_RADIUS:-0.1}"
        SOURCE_BALL_DIV="${SOURCE_BALL_DIV:-60.0}"
        TAU_DIV="${TAU_DIV:-100.0}"
        ;;
esac

# Physical parameters at 10 GHz:
# lambda0 = c / 10 GHz = 0.0299792458 m
LAMBDA0_M="0.0299792458"
RADIUS_A_M=$(awk "BEGIN {printf \"%.17g\", 2.0 * $LAMBDA0_M}")
# In COMSOL, air_extent (b) is the distance from the origin (0,0,0) to the inner PML face.
# It must be larger than the sphere radius a (60 mm): b = a + air_buffer = 2.0*lambda0 + 0.35*lambda0 = 2.35*lambda0 (~70.5 mm).
AIR_BUFFER_M=$(awk "BEGIN {printf \"%.17g\", 0.35 * $LAMBDA0_M}")
AIR_EXTENT_M=$(awk "BEGIN {printf \"%.17g\", $RADIUS_A_M + $AIR_BUFFER_M}")
TAU_M=$(awk "BEGIN {printf \"%.17g\", $LAMBDA0_M / $TAU_DIV}")
LINEAR_SOLVER="${LINEAR_SOLVER:-mumps}"
DISCRETIZATION="${DISCRETIZATION:-linear}"
CORES="${SLURM_CPUS_PER_TASK:-64}"

RADIAL_POINTS="${RADIAL_POINTS:-300}"
RADIAL_MAX_RADII="${RADIAL_MAX_RADII:-2.0}"
ZX_NX="${ZX_GRID:-200}"
ZX_NZ="${ZX_GRID:-200}"
ZX_SPAN_RADII="${ZX_SPAN_RADII:-2.0}"

CASE_DIR="applications/anisotropic_gstc_sphere/results/dipole_pol_rot/e2m_eps7_2lambda"
if [ ! -d "$CASE_DIR" ]; then
    CASE_DIR="applications/anisotropic_gstc_sphere/results/dipole_pol_rot/electric_to_magnetic_10ghz_eps7_2lambda_reciprocal_lossless"
fi
TENSORS="$CASE_DIR/face_tensors.bin"
MESH="applications/anisotropic_gstc_sphere/meshes/electric_to_magnetic_10ghz_eps7_2lambda/sphere_ppw2.obj"
CONFIG="applications/anisotropic_gstc_sphere/configurations/dipole_pol_rot/e2m_eps7_2lambda.json"
SIE_DIR="$CASE_DIR/cases/electric_incidence"

RUN_NAME="bianiso_2lam_${PRESET}_shell${SHELL_TAN_DIV}_glob${GLOBAL_DIV}_lay${SHELL_LAYERS}_tau${TAU_DIV}_src${SOURCE_BALL_RADIUS}_div${SOURCE_BALL_DIV}_6.1_slurm${SLURM_JOB_ID}"
OUT_DIR="$CASE_DIR/comsol/${RUN_NAME}"
mkdir -p "$OUT_DIR/logs"

echo "================================================================================"
echo "COMSOL 6.1 2*Lambda Anisotropic Validation: Electric to Magnetic (Job ${SLURM_JOB_ID})"
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
    echo "Tensors file not found at $TENSORS"
    echo "Generating GSTC metasurface tensors on the fly via sie_gstc_sphere.py --prepare-only..."
    MESH_OPT=""
    if [ -f "$MESH" ]; then
        MESH_OPT="--use-existing-mesh"
    fi
    python -u applications/anisotropic_gstc_sphere/sie_gstc_sphere.py \
        --config "$CONFIG" \
        --run-dir "$CASE_DIR" \
        --prepare-only \
        $MESH_OPT
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
MPLCONFIGDIR=/tmp/mpl python -u applications/anisotropic_gstc_sphere/plot_fields.py \
    --run-dir "$OUT_DIR" \
    --case electric_incidence \
    --reference-transformation electric_to_magnetic \
    --line-basis spherical \
    --map-basis spherical \
    --map-percentiles 20 98 \
    --magnitude-scale db || true

MPLCONFIGDIR=/tmp/mpl python -u applications/anisotropic_gstc_sphere/plot_fields.py \
    --run-dir "$OUT_DIR" \
    --case electric_incidence \
    --reference-transformation electric_to_magnetic \
    --line-basis spherical \
    --map-basis spherical \
    --line-components E_r E_theta E_phi \
    --map-components E_r E_theta E_phi \
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

