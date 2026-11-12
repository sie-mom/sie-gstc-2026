#!/usr/bin/env bash
#SBATCH --job-name=omega-comsol-volumetric
#SBATCH --account=jordan_budhu
#SBATCH --partition=normal_q
#SBATCH --qos=tc_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-omega-volumetric-%j.out
#SBATCH --error=slurm-omega-volumetric-%j.out

# ==============================================================================
# COMSOL Slurm Batch Driver: Bi-Isotropic Omega Metasurface Sphere (Case B)
#
# Target: Virginia Tech TinkerCliffs cluster (normal_q, 64 CPU cores)
# Memory: 200 GB (fits on standard AMD Rome 256 GB nodes without high-mem delay)
#
# Presets (selected via PRESET environment variable, default: case4):
#   - smoke  : Fast smoke test (Global=15, Shell=20, Layers=2, ~950k DOFs, ~2 min)
#   - case4  : Benchmark Case 4 (Global=15, Shell=25, Layers=2, ~993k DOFs, ~2 min)
#   - case12 : High-res Case 12 (Global=20, Shell=45, Layers=5, ~2.75M DOFs, ~5 min)
#   - case19 : Publication Checkpoint 19 (Global=20, Interior=35, Shell=90, Layers=5, ~5.5M DOFs, ~14 min)
#
# Usage:
#   # 1. Run default fast benchmark (Case 4):
#   sbatch applications/omega_biisotropic_sphere/comsol/tinkercliffs_volumetric_swept.sbatch
#
#   # 2. Run publication checkpoint 19:
#   PRESET=case19 sbatch applications/omega_biisotropic_sphere/comsol/tinkercliffs_volumetric_swept.sbatch
#
#   # 3. Custom parameter overrides:
#   GLOBAL_DIV=20 SHELL_TAN_DIV=60 sbatch applications/omega_biisotropic_sphere/comsol/tinkercliffs_volumetric_swept.sbatch
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="${SIE_MOM_ROOT:-/home/germanf/repo/sie_mom}"
cd "$PROJECT_ROOT"

# ------------------------------------------------------------------------------
# 1. Configuration & Preset Selection
# ------------------------------------------------------------------------------
COMSOL_VERSION="${COMSOL_VERSION:-6.1}"
PRESET="${PRESET:-case4}"
CONFIG="${CONFIG:-applications/omega_biisotropic_sphere/configurations/budhu_10ghz_eps7.json}"
STAGE="omega_sphere"
SHELL_MESH="swept"
DISCRETIZATION="linear"
AIR_EXTENT="${AIR_EXTENT:-0.95}"
TAU_THICKNESS="${TAU_THICKNESS:-0.01}"
LINEAR_SOLVER="${LINEAR_SOLVER:-mumps}"

# Preset sizing definitions from historical benchmark table
case "$PRESET" in
    smoke)
        DEFAULT_GLOBAL_DIV="15.0"
        DEFAULT_SHELL_TAN_DIV="20.0"
        DEFAULT_SHELL_LAYERS="2"
        DEFAULT_INTERIOR_DIV=""
        ;;
    case4)
        DEFAULT_GLOBAL_DIV="15.0"
        DEFAULT_SHELL_TAN_DIV="25.0"
        DEFAULT_SHELL_LAYERS="2"
        DEFAULT_INTERIOR_DIV=""
        ;;
    case6)
        DEFAULT_GLOBAL_DIV="20.0"
        DEFAULT_SHELL_TAN_DIV="25.0"
        DEFAULT_SHELL_LAYERS="3"
        DEFAULT_INTERIOR_DIV=""
        ;;
    case7)
        DEFAULT_GLOBAL_DIV="20.0"
        DEFAULT_SHELL_TAN_DIV="25.0"
        DEFAULT_SHELL_LAYERS="4"
        DEFAULT_INTERIOR_DIV=""
        ;;
    case8)
        DEFAULT_GLOBAL_DIV="20.0"
        DEFAULT_SHELL_TAN_DIV="30.0"
        DEFAULT_SHELL_LAYERS="4"
        DEFAULT_INTERIOR_DIV=""
        ;;
    case12)
        DEFAULT_GLOBAL_DIV="20.0"
        DEFAULT_SHELL_TAN_DIV="45.0"
        DEFAULT_SHELL_LAYERS="5"
        DEFAULT_INTERIOR_DIV=""
        ;;
    case19)
        DEFAULT_GLOBAL_DIV="20.0"
        DEFAULT_SHELL_TAN_DIV="90.0"
        DEFAULT_SHELL_LAYERS="5"
        DEFAULT_INTERIOR_DIV="35.0"
        ;;
    custom)
        DEFAULT_GLOBAL_DIV="15.0"
        DEFAULT_SHELL_TAN_DIV="25.0"
        DEFAULT_SHELL_LAYERS="2"
        DEFAULT_INTERIOR_DIV=""
        ;;
    *)
        echo "Unknown PRESET='$PRESET'. Valid: smoke, case4, case12, case19, custom" >&2
        exit 1
        ;;
esac

GLOBAL_DIV="${GLOBAL_DIV:-$DEFAULT_GLOBAL_DIV}"
SHELL_TAN_DIV="${SHELL_TAN_DIV:-$DEFAULT_SHELL_TAN_DIV}"
SHELL_LAYERS="${SHELL_LAYERS:-$DEFAULT_SHELL_LAYERS}"
INTERIOR_DIV="${INTERIOR_DIV:-$DEFAULT_INTERIOR_DIV}"

# Extract case_id from config
CASE_ID=$(python3 -c "import json; print(json.load(open('$CONFIG'))['case_id'])")
RUN_NAME="${RUN_NAME:-omega_${PRESET}_shell${SHELL_TAN_DIV}_glob${GLOBAL_DIV}_lay${SHELL_LAYERS}_${COMSOL_VERSION}_slurm${SLURM_JOB_ID}}"
OUT_DIR="$PROJECT_ROOT/applications/omega_biisotropic_sphere/results/$CASE_ID/comsol/$RUN_NAME"
mkdir -p "$OUT_DIR/logs"

# ------------------------------------------------------------------------------
# 2. Modules & License Setup
# ------------------------------------------------------------------------------
echo "================================================================================"
echo "COMSOL $COMSOL_VERSION Bi-Isotropic Job ${SLURM_JOB_ID} on $(hostname) at $(date --iso-8601=seconds)"
echo "Preset: $PRESET (GlobalDiv=$GLOBAL_DIV, ShellTanDiv=$SHELL_TAN_DIV, Layers=$SHELL_LAYERS)"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-64}, Memory requested: 200 GB"
echo "Output Directory: $OUT_DIR"
echo "================================================================================"

if [ "$COMSOL_VERSION" = "6.1" ]; then
    # Jordan Budhu group custom COMSOL 6.1 (1 RF Module seat)
    export LM_LICENSE_FILE="1718@128.173.92.65"
    module reset
    module use /projects/jordan_budhu/modules/tinkercliffs-rome/all
    module load comsol/6.1
    PHYSICS_INTERFACE="emw"
else
    # Campus/ARC system COMSOL 5.4 (Wave Optics ewfd)
    export LM_LICENSE_FILE="1718@comsol.software.vt.edu"
    module reset
    module load COMSOL/5.4.0.388
    PHYSICS_INTERFACE="ewfd"
    if [ -z "${JAVA_HOME:-}" ] && command -v java >/dev/null 2>&1; then
        export JAVA_HOME="$(dirname $(dirname $(readlink -f $(which java))))"
    fi
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1

module list

# ------------------------------------------------------------------------------
# 3. Assemble Execution Command
# ------------------------------------------------------------------------------
CMD=(
    python -u applications/omega_biisotropic_sphere/comsol/volumetric_impedance_matching.py
    --stage "$STAGE"
    --version "$COMSOL_VERSION"
    --physics-interface "$PHYSICS_INTERFACE"
    --discretization "$DISCRETIZATION"
    --config "$CONFIG"
    --case-id "$CASE_ID"
    --run-name "$RUN_NAME"
    --run-dir "$OUT_DIR"
    --cores "${SLURM_CPUS_PER_TASK:-64}"
    --linear-solver "$LINEAR_SOLVER"
    --shell-mesh "$SHELL_MESH"
    --shell-radial-layers "$SHELL_LAYERS"
    --shell-tangential-divisor "$SHELL_TAN_DIV"
    --global-hmax-divisor "$GLOBAL_DIV"
    --air-extent "$AIR_EXTENT"
    --tau-thickness "$TAU_THICKNESS"
    --radial-theta 45.0
    --radial-phi 0.0
    --radial-points 501
    --radial-max-radii 1.9
    --zx-grid-size 300
    --zx-span-radii 1.3
)

if [ -n "$INTERIOR_DIV" ]; then
    CMD+=(--interior-hmax-divisor "$INTERIOR_DIV")
fi

echo -e "\nExecuting Command:\n${CMD[*]}\n"
"${CMD[@]}" 2>&1 | tee "$OUT_DIR/logs/comsol_execution.log"

# ------------------------------------------------------------------------------
# 4. Manifest & Error Validation
# ------------------------------------------------------------------------------
echo -e "\n--- Validating Results Manifest ---"
python -c '
import json, sys, pathlib
manifest_path = pathlib.Path(sys.argv[1])
if not manifest_path.is_file():
    print(f"ERROR: Manifest missing at {manifest_path}", file=sys.stderr)
    sys.exit(1)
manifest = json.loads(manifest_path.read_text())
status = manifest.get("status")
if status != "complete":
    print(f"ERROR: Run status is \"{status}\"", file=sys.stderr)
    sys.exit(1)

case_id = manifest.get("case_id")
n_elems = manifest.get("mesh_elements", 0)
n_prisms = manifest.get("mesh_prisms", 0)
elapsed = manifest.get("elapsed_seconds", 0.0)
err_e_rad = manifest.get("relative_l2_error_electric_radial_buffered", float("nan"))
err_e_zx = manifest.get("relative_l2_error_electric_zx_buffered", float("nan"))
err_h_rad = manifest.get("relative_l2_error_magnetic_radial_buffered", float("nan"))

print("\n=== Validation Metrics Summary ===")
print(f"Case ID           : {case_id}")
print(f"Mesh Elements     : {n_elems:,}")
print(f"Swept Prisms      : {n_prisms:,}")
print(f"Elapsed Time      : {elapsed:.1f} s")
print(f"Electric Radial L2: {err_e_rad * 100:.2f}% (buffered)")
print(f"Electric ZX L2    : {err_e_zx * 100:.2f}% (buffered)")
print(f"Magnetic Radial L2: {err_h_rad * 100:.2f}% (buffered)")
print("===================================\n")
' "$OUT_DIR/run.json"

echo "COMSOL $COMSOL_VERSION Bi-Isotropic Job ${SLURM_JOB_ID} completed successfully at $(date --iso-8601=seconds)"

