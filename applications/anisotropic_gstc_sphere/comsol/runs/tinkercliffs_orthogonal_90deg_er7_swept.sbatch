#!/usr/bin/env bash
#SBATCH --job-name=comsol-bianiso-swept
#SBATCH --account=jordan_budhu
#SBATCH --partition=normal_q
#SBATCH --qos=tc_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-comsol-bianiso-%j.out
#SBATCH --error=slurm-comsol-bianiso-%j.out

# ==============================================================================
# COMSOL Slurm Batch Driver: Bianisotropic Spherical Metasurface (Case B)
#
# Target: TinkerCliffs cluster (normal_q, 64 CPU cores, no GPU needed)
# Case  : Orthogonal 90-degree dipole rotation, er=7 core, 1 m offset, swept shell
#
# Usage:
#   sbatch applications/anisotropic_gstc_sphere/comsol/runs/tinkercliffs_orthogonal_90deg_er7_swept.sbatch
#
# Override parameters via environment variables at submit time, e.g.:
#   COMSOL_VERSION=6.1 TAG=mesh_refined sbatch applications/anisotropic_gstc_sphere/comsol/runs/tinkercliffs_orthogonal_90deg_er7_swept.sbatch
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="${SIE_MOM_ROOT:-/home/germanf/repo/sie_mom}"
cd "$PROJECT_ROOT"

# ------------------------------------------------------------------------------
# 1. Environment & License Selection (COMSOL 5.4 vs 6.1)
# ------------------------------------------------------------------------------
COMSOL_VERSION="${COMSOL_VERSION:-6.1}"

if [ "$COMSOL_VERSION" = "6.1" ]; then
    # Jordan Budhu group license server (1 RF seat)
    export LM_LICENSE_FILE="1718@128.173.92.65"
    module reset
    module use /projects/jordan_budhu/modules/tinkercliffs-rome/all
    module load comsol/6.1
else
    # Campus/ARC license server for COMSOL 5.4 (Wave Optics)
    export LM_LICENSE_FILE="1718@comsol.software.vt.edu"
    module reset
    module load COMSOL/5.4.0.388
    if [ -z "${JAVA_HOME:-}" ] && command -v java >/dev/null 2>&1; then
        export JAVA_HOME="$(dirname $(dirname $(readlink -f $(which java))))"
    fi
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "================================================================================"
echo "COMSOL $COMSOL_VERSION Job ${SLURM_JOB_ID} started on $(hostname) at $(date --iso-8601=seconds)"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-64}"
echo "================================================================================"

# ------------------------------------------------------------------------------
# 2. Case and Simulation Directories
# ------------------------------------------------------------------------------
BASE="applications/anisotropic_gstc_sphere"
RESULT="$BASE/results/dipole_rotation"
RUN="${RUN:-orthogonal_90deg_er7_offset_1m_phase_aligned_smooth002_norm_400iter}"
NAME="${NAME:-orthogonal_90deg_er7_offset_1m_swept}"
TAG="${TAG:-tc_${SLURM_JOB_ID}}"

MESH="${MESH:-$BASE/meshes/reflectionless_100mhz_radius2m_ppw10/sphere.obj}"
CONFIG="${CONFIG:-$BASE/configurations/moment_rot/rot90_eps7.json}"
SIE_DIR="${SIE_DIR:-$RESULT/$RUN/simulation_iter400/cases/electric_incidence}"

OUT="$RESULT/comsol/${NAME}_${TAG}"
mkdir -p "$OUT/logs"

# Redirect stdout and stderr into the run output log
exec > >(tee -a "$OUT/logs/slurm-${SLURM_JOB_ID}.log") 2>&1

# ------------------------------------------------------------------------------
# 3. Solver & Concentric Mesh Refinement Parameters
# ------------------------------------------------------------------------------
SHELL_MESH="${SHELL_MESH:-swept}"
SHELL_LAYERS="${SHELL_LAYERS:-5}"
GRID_OVERSAMPLE="${GRID_OVERSAMPLE:-3.0}"
GRID_METHOD="${GRID_METHOD:-cubic}"
SHELL_TAN_DIV="${SHELL_TAN_DIV:-35.0}"
GLOBAL_DIV="${GLOBAL_DIV:-20.0}"
INTERIOR_DIV="${INTERIOR_DIV:-25.0}"
SRC_RADIUS="${SRC_RADIUS:-0.15}"
SRC_DIV="${SRC_DIV:-60.0}"
INNER_WIDTH="${INNER_WIDTH:-0.05}"
INNER_DIV="${INNER_DIV:-40.0}"
OUTER_WIDTH="${OUTER_WIDTH:-0.05}"
OUTER_DIV="${OUTER_DIV:-40.0}"
LINEAR_SOLVER="${LINEAR_SOLVER:-pardiso}"
DISCRETIZATION="${DISCRETIZATION:-linear}"
CORES="${SLURM_CPUS_PER_TASK:-64}"

# Record exact parameters
cat <<EOF > "$OUT/params.txt"
date=$(date --iso-8601=seconds)
slurm_job_id=${SLURM_JOB_ID}
host=$(hostname)
git_commit=$(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')
COMSOL_VERSION=$COMSOL_VERSION
SHELL_MESH=$SHELL_MESH
SHELL_LAYERS=$SHELL_LAYERS
GRID_OVERSAMPLE=$GRID_OVERSAMPLE
GRID_METHOD=$GRID_METHOD
SHELL_TAN_DIV=$SHELL_TAN_DIV
GLOBAL_DIV=$GLOBAL_DIV
INTERIOR_DIV=$INTERIOR_DIV
SRC_RADIUS=$SRC_RADIUS
SRC_DIV=$SRC_DIV
INNER_WIDTH=$INNER_WIDTH
INNER_DIV=$INNER_DIV
OUTER_WIDTH=$OUTER_WIDTH
OUTER_DIV=$OUTER_DIV
LINEAR_SOLVER=$LINEAR_SOLVER
DISCRETIZATION=$DISCRETIZATION
CORES=$CORES
EOF

# ------------------------------------------------------------------------------
# 4. Execute COMSOL Validation Pipeline
# ------------------------------------------------------------------------------
python "$BASE/comsol/bianisotropic_shell_validation.py" \
    --thetas "$RESULT/$RUN/optimized_thetas.bin" \
    --mesh "$MESH" \
    --config "$CONFIG" \
    --version "$COMSOL_VERSION" \
    --shell-mesh "$SHELL_MESH" \
    --shell-layers "$SHELL_LAYERS" \
    --grid-oversample "$GRID_OVERSAMPLE" \
    --grid-method "$GRID_METHOD" \
    --shell-tangential-divisor "$SHELL_TAN_DIV" \
    --global-mesh-divisor "$GLOBAL_DIV" \
    --interior-hmax-divisor "$INTERIOR_DIV" \
    --source-ball-radius "$SRC_RADIUS" \
    --source-ball-hmax-divisor "$SRC_DIV" \
    --inner-shell-band-width "$INNER_WIDTH" \
    --inner-shell-band-hmax-divisor "$INNER_DIV" \
    --outer-shell-band-width "$OUTER_WIDTH" \
    --outer-shell-band-hmax-divisor "$OUTER_DIV" \
    --linear-solver "$LINEAR_SOLVER" \
    --discretization "$DISCRETIZATION" \
    --cores "$CORES" \
    --output-dir "$OUT"

# ------------------------------------------------------------------------------
# 5. Compare Directly Against SIE-BEM Benchmark
# ------------------------------------------------------------------------------
echo "================================================================================"
echo "Comparing COMSOL Results with SIE-BEM Reference..."
echo "================================================================================"

python "$BASE/comsol/compare_sie_comsol.py" \
    --sie-dir "$SIE_DIR" \
    --comsol-dir "$OUT" \
    --output-dir "$OUT/comparison"

echo "COMSOL Job ${SLURM_JOB_ID} finished successfully at $(date --iso-8601=seconds)"
