#!/usr/bin/env bash
#SBATCH --job-name=omega-bianiso-val
#SBATCH --account=jordan_budhu
#SBATCH --partition=normal_q
#SBATCH --qos=tc_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-omega-bianiso-%j.out
#SBATCH --error=slurm-omega-bianiso-%j.out

# ==============================================================================
# TinkerCliffs COMSOL Slurm Driver: Bi-Isotropic Validation via Bianisotropic Pipeline
#
# Validates the 10-parameter bianisotropic formulation and Weak Form against the
# known-good bi-isotropic omega sphere benchmark (budhu_10ghz_eps7) by directly
# loading its SIE-extracted face_tensors.bin and passing it through
# bianisotropic_shell_validation.py on COMSOL 6.1.
#
# Usage:
#   sbatch applications/anisotropic_gstc_sphere/comsol/runs/tinkercliffs_biisotropic_tensor_swept.sbatch
#
# Presets:
#   PRESET=case6 (default): 3 radial prism layers, tan_div=25.0, glob_div=20.0 (~20% L2 error target)
#   PRESET=case4          : 2 radial prism layers, tan_div=20.0, glob_div=15.0 (~132% L2 error reference)
#   PRESET=case7          : 4 radial prism layers, tan_div=30.0, glob_div=25.0 (further refinement)
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

PRESET="${PRESET:-case6}"

# ------------------------------------------------------------------------------
# Presets Matching Bi-Isotropic Convergence Benchmark
# ------------------------------------------------------------------------------
case "$PRESET" in
    case4)
        SHELL_LAYERS=2
        SHELL_TAN_DIV="20.0"
        GLOBAL_DIV="15.0"
        INTERIOR_DIV="15.0"
        SOURCE_BALL_RADIUS="0.0"
        ;;
    case6)
        # Default: Exact settings of Job 8359821 (356s runtime, 20.32% radial L2 error)
        SHELL_LAYERS=3
        SHELL_TAN_DIV="25.0"
        GLOBAL_DIV="20.0"
        INTERIOR_DIV="20.0"
        SOURCE_BALL_RADIUS="0.0"
        ;;
    case7)
        SHELL_LAYERS=4
        SHELL_TAN_DIV="30.0"
        GLOBAL_DIV="25.0"
        INTERIOR_DIV="25.0"
        SOURCE_BALL_RADIUS="0.0"
        ;;
    *)
        echo "Unknown PRESET '$PRESET'. Using custom settings or default case6."
        SHELL_LAYERS="${SHELL_LAYERS:-3}"
        SHELL_TAN_DIV="${SHELL_TAN_DIV:-25.0}"
        GLOBAL_DIV="${GLOBAL_DIV:-20.0}"
        INTERIOR_DIV="${INTERIOR_DIV:-20.0}"
        SOURCE_BALL_RADIUS="${SOURCE_BALL_RADIUS:-0.0}"
        ;;
esac

# At f0 = 10 GHz, lambda0 = c / 10 GHz = 0.0299792458 m
# air-extent = 0.95 * lambda0 = 0.02848028351 m (matching Job 8359821)
AIR_EXTENT_M="0.02848028351"
LINEAR_SOLVER="${LINEAR_SOLVER:-mumps}"
DISCRETIZATION="${DISCRETIZATION:-linear}"
CORES="${SLURM_CPUS_PER_TASK:-64}"

RADIAL_POINTS="${RADIAL_POINTS:-501}"
RADIAL_MAX_RADII="${RADIAL_MAX_RADII:-1.9}"
ZX_NX="${ZX_NX:-300}"
ZX_NZ="${ZX_NZ:-300}"
ZX_SPAN_RADII="${ZX_SPAN_RADII:-1.3}"

# ------------------------------------------------------------------------------
# Input Files and Destination Directories
# ------------------------------------------------------------------------------
TENSORS="applications/omega_biisotropic_sphere/results/budhu_10ghz_eps7/sie/least_squares/face_tensors.bin"
MESH="applications/omega_biisotropic_sphere/meshes/budhu_10ghz_eps7/sphere.obj"
CONFIG="applications/anisotropic_gstc_sphere/configurations/example_1_omega_matching_10ghz_eps7.json"
SIE_DIR="applications/omega_biisotropic_sphere/results/budhu_10ghz_eps7/sie/least_squares"

RUN_NAME="bianiso_${PRESET}_shell${SHELL_TAN_DIV}_glob${GLOBAL_DIV}_lay${SHELL_LAYERS}_6.1_slurm${SLURM_JOB_ID}"
OUT_DIR="applications/anisotropic_gstc_sphere/results/budhu_10ghz_eps7/comsol/${RUN_NAME}"
mkdir -p "$OUT_DIR/logs"

echo "================================================================================"
echo "COMSOL 6.1 Bianisotropic Validation on Bi-Isotropic Benchmark (Job ${SLURM_JOB_ID})"
echo "Host             : $(hostname)"
echo "Date             : $(date --iso-8601=seconds)"
echo "Preset           : $PRESET"
echo "CPUs allocated   : $CORES"
echo "Tensors file     : $TENSORS"
echo "Mesh file        : $MESH"
echo "Configuration    : $CONFIG"
echo "Output Directory : $OUT_DIR"
echo "================================================================================"

# Ensure tensor input file exists (results/ is gitignored so generate on the fly if missing)
if [ ! -f "$TENSORS" ]; then
    echo "Tensors file not found at $TENSORS."
    echo "Generating canonical bi-isotropic SIE tensors via prepare_sie_inputs.py..."
    python -u applications/omega_biisotropic_sphere/prepare_sie_inputs.py \
        --config "applications/omega_biisotropic_sphere/configurations/budhu_10ghz_eps7.json" \
        --use-existing-mesh
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
    --air-extent "$AIR_EXTENT_M" \
    --radial-points "$RADIAL_POINTS" \
    --radial-max-radii "$RADIAL_MAX_RADII" \
    --radial-theta 45.0 \
    --radial-phi 0.0 \
    --zx-nx "$ZX_NX" \
    --zx-nz "$ZX_NZ" \
    --zx-span-radii "$ZX_SPAN_RADII"

echo ""
echo "--- Running Benchmark Comparison Against SIE-BEM Reference ---"
python -u applications/anisotropic_gstc_sphere/comsol/compare_sie_comsol.py \
    --sie-dir "$SIE_DIR" \
    --comsol-dir "$OUT_DIR" || true

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

