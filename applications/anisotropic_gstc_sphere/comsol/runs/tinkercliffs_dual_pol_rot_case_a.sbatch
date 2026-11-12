#!/usr/bin/env bash
#SBATCH --job-name=dual-case-a
#SBATCH --account=jordan_budhu
#SBATCH --partition=normal_q
# (For runs > 4 hours, do not specify tc_normal_short)
# #SBATCH --qos=tc_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=230G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-dual-casea-%j.out
#SBATCH --error=slurm-dual-casea-%j.out

# ==============================================================================
# TinkerCliffs COMSOL Slurm Driver: Case A (Zero-Thickness Sheet GSTC)
# Dual Complementary Polarization Rotation (a = 0.7 * lambda0, er = 7)
#
# Solves BOTH complementary transformations on the same bianisotropic sheet
# (electric_incidence: p_z -> m_z; magnetic_incidence: m_z -> p_z) inside a
# single COMSOL 6.1 study by reusing a single MUMPS LU factorization across
# both RHS excitations.
#
# Enforces two-sided GSTC jump conditions directly on r = a via SurfaceCurrent
# (scu1) + SurfaceMagneticCurrentDensity (smcd1) with broken-H(curl) up()/down()
# field traces, using the Case 2 1:1 conformal band refinement around r = a.
#
# Usage:
#   sbatch applications/anisotropic_gstc_sphere/comsol/runs/tinkercliffs_dual_pol_rot_case_a.sbatch
#   PRESET=case1 sbatch applications/anisotropic_gstc_sphere/comsol/runs/tinkercliffs_dual_pol_rot_case_a.sbatch
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

PRESET="${PRESET:-case2}"

# ------------------------------------------------------------------------------
# Controlled Mesh Convergence Presets (Case A Zero-Thickness Sheet)
# ------------------------------------------------------------------------------
case "$PRESET" in
    case1)
        # Baseline Case A: bulk/surface resolution without conformal near-sheet bands
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
        # Conformal Band Refinement (Default): 1:1 aspect ratio volume tetrahedra at r = a
        # Validated in Job 8378521 (~2.24M tets, ~218 GB peak RSS, ~20 min MUMPS LU)
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
        echo "Unknown PRESET '$PRESET'. Using custom environment settings or case2 defaults."
        SHEET_TAN_DIV="${SHEET_TAN_DIV:-30.0}"
        GLOBAL_DIV="${GLOBAL_DIV:-20.0}"
        INTERIOR_DIV="${INTERIOR_DIV:-20.0}"
        INNER_SHEET_BAND_WIDTH="${INNER_SHEET_BAND_WIDTH:-0.08}"
        INNER_SHEET_BAND_DIV="${INNER_SHEET_BAND_DIV:-30.0}"
        OUTER_SHEET_BAND_WIDTH="${OUTER_SHEET_BAND_WIDTH:-0.10}"
        OUTER_SHEET_BAND_DIV="${OUTER_SHEET_BAND_DIV:-30.0}"
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

CASE_DIR="applications/anisotropic_gstc_sphere/results/dipole_pol_rot/dual_pol_eps7"
if [ ! -d "$CASE_DIR" ]; then
    CASE_DIR="applications/anisotropic_gstc_sphere/results/dipole_pol_rot/dual_pol_rot_10ghz_eps7_reciprocal_lossless"
fi
TENSORS="$CASE_DIR/face_tensors.bin"
MESH="applications/anisotropic_gstc_sphere/meshes/example_1_omega_matching_10ghz_eps7/sphere_ppw6.obj"
CONFIG="applications/anisotropic_gstc_sphere/configurations/dipole_pol_rot/dual_pol_eps7.json"

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

CASE_ID="${CASE_ID:-}"
EXTRA_CASE_ARGS=()
if [ -n "$CASE_ID" ]; then
    EXTRA_CASE_ARGS+=(--case-id "$CASE_ID")
    TARGET_CASES=("$CASE_ID")
else
    TARGET_CASES=(electric_incidence magnetic_incidence)
fi

if [ "$SOURCE_BALL_RADIUS" != "0.0" ] && [ "$SOURCE_BALL_RADIUS" != "0" ]; then
    RUN_NAME="casea_${PRESET}_sheet${SHEET_TAN_DIV}${BAND_TAG}_glob${GLOBAL_DIV}_int${INTERIOR_DIV}_ff${FF_OFFSET_LAM}_src${SOURCE_BALL_RADIUS}_div${SOURCE_BALL_DIV}_${COMSOL_VERSION}_slurm${SLURM_JOB_ID}"
else
    RUN_NAME="casea_${PRESET}_sheet${SHEET_TAN_DIV}${BAND_TAG}_glob${GLOBAL_DIV}_int${INTERIOR_DIV}_ff${FF_OFFSET_LAM}_${COMSOL_VERSION}_slurm${SLURM_JOB_ID}"
fi
OUT_DIR="$CASE_DIR/comsol/${RUN_NAME}"
mkdir -p "$OUT_DIR/logs"

echo "================================================================================"
echo "COMSOL $COMSOL_VERSION Case A (Zero-Thickness Sheet) Validation: Dual Pol Rot (Job ${SLURM_JOB_ID})"
echo "Host             : $(hostname)"
echo "Date             : $(date --iso-8601=seconds)"
echo "Preset           : $PRESET"
echo "CPUs allocated   : $CORES"
echo "Linear Solver    : $LINEAR_SOLVER (single LU factorization across 2 RHS cases)"
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

# 1. Execute Bianisotropic Case A COMSOL Validation Pipeline (solves both cases via LU reuse)
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
    "${EXTRA_CASE_ARGS[@]}" \
    --air-extent "$AIR_EXTENT_M" \
    --radial-points "$RADIAL_POINTS" \
    --radial-max-radii "$RADIAL_MAX_RADII" \
    --radial-theta 45.0 \
    --radial-phi 0.0 \
    --zx-nx "$ZX_NX" \
    --zx-nz "$ZX_NZ" \
    --zx-span-radii "$ZX_SPAN_RADII"

echo ""
echo "--- Ensuring SIE Reference Far-Field Cuts Exist ---"
if [ -d "$CASE_DIR/cases" ]; then
    MISSING_FF=0
    for C_ID in "${TARGET_CASES[@]}"; do
        SIE_CASE_DIR="$CASE_DIR/cases/$C_ID"
        if [ -d "$SIE_CASE_DIR" ] && [ ! -f "$SIE_CASE_DIR/far_field/far_field_cuts.csv" ] && [ ! -f "$SIE_CASE_DIR/far_field_cuts.csv" ]; then
            MISSING_FF=1
        fi
    done
    if [ "$MISSING_FF" -eq 1 ]; then
        echo "Generating missing SIE reference far-field cuts..."
        MPLCONFIGDIR=/tmp/mpl python -u applications/anisotropic_gstc_sphere/plot_far_field.py \
            --run-dir "$CASE_DIR" \
            --cases "${TARGET_CASES[@]}" || true
    fi
fi

echo ""
echo "--- Running Benchmark Comparisons Against SIE Reference (Both Transformations) ---"
for C_ID in "${TARGET_CASES[@]}"; do
    SIE_CASE_DIR="$CASE_DIR/cases/$C_ID"
    COMSOL_CASE_DIR="$OUT_DIR/cases/$C_ID"
    if [ -d "$SIE_CASE_DIR" ] && [ -d "$COMSOL_CASE_DIR" ]; then
        echo ""
        echo ">>> Comparing case: $C_ID <<<"
        MPLCONFIGDIR=/tmp/mpl python -u applications/anisotropic_gstc_sphere/comsol/compare_sie_comsol.py \
            --sie-dir "$SIE_CASE_DIR" \
            --comsol-dir "$COMSOL_CASE_DIR" || true
    fi
done

echo ""
echo "--- Generating Physical Field Comparison Plots via plot_fields.py (Spherical Basis) ---"
declare -A REF_TRANSFORM=(
    ["electric_incidence"]="electric_to_magnetic"
    ["magnetic_incidence"]="magnetic_to_electric"
)

for C_ID in "${TARGET_CASES[@]}"; do
    REF_ID="${REF_TRANSFORM[$C_ID]}"
    echo ""
    echo ">>> Plotting spherical field suite for $C_ID (target: $REF_ID) <<<"
    # 1. Full spherical component suite (norms and all spherical vector components)
    MPLCONFIGDIR=/tmp/mpl python -u applications/anisotropic_gstc_sphere/plot_fields.py \
        --run-dir "$OUT_DIR" \
        --case "$C_ID" \
        --reference-transformation "$REF_ID" \
        --line-basis spherical \
        --map-basis spherical \
        --map-percentiles 20 98 \
        --magnitude-scale db || true

    # 2. Dedicated mode rotation comparison (E_theta TM mode vs E_phi TE mode)
    MPLCONFIGDIR=/tmp/mpl python -u applications/anisotropic_gstc_sphere/plot_fields.py \
        --run-dir "$OUT_DIR" \
        --case "$C_ID" \
        --reference-transformation "$REF_ID" \
        --line-basis spherical \
        --map-basis spherical \
        --line-components E_theta E_phi \
        --map-components E_theta E_phi \
        --map-percentiles 20 98 \
        --magnitude-scale db || true
done

echo ""
echo "================================================================================"
echo "SUMMARY METRICS (DUAL TRANSFORMATION CASE A):"
echo "================================================================================"
python - "$OUT_DIR" <<'PYEOF'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
comsol_metrics_path = out_dir / "comsol_metrics.json"

if comsol_metrics_path.is_file():
    with comsol_metrics_path.open() as f:
        cm = json.load(f)
    mesh = cm.get("mesh", {})
    solver = cm.get("solver", {})
    print(f"Total Mesh Elements : {mesh.get('total_elements', 'N/A'):,}")
    print(f"Sheet Triangles     : {mesh.get('sheet_triangles', 'N/A'):,}")
    print(f"Volume Tetrahedra   : {mesh.get('tetrahedra', 'N/A'):,}")
    print(
        f"LU Reused Across    : {solver.get('case_count', 1)} RHS cases "
        f"(lu_reused={solver.get('lu_factorization_reused', False)})"
    )
    print(f"Combined Solve Time : {solver.get('solve_duration_seconds', 0.0):.2f} s")
    print(f"Total Execution     : {cm.get('total_execution_seconds', 0.0):.2f} s")

for case_id in ("electric_incidence", "magnetic_incidence"):
    cmp_path = out_dir / "cases" / case_id / "comparison" / "comparison_metrics.json"
    if cmp_path.is_file():
        with cmp_path.open() as f:
            m = json.load(f)
        rad = m.get("radial_metrics", {})
        zx = m.get("zx_metrics", {})
        ff = m.get("far_field_metrics", {})
        print(f"\n--- Case: {case_id} ---")
        print(
            f"  Radial Rel L2 |E| / |H|  : "
            f"{rad.get('radial_l2_rel_err_E_percent', float('nan')):.2f}% / "
            f"{rad.get('radial_l2_rel_err_H_percent', float('nan')):.2f}%"
        )
        print(
            f"  Interior / Exterior |E|  : "
            f"{rad.get('interior_mean_rel_err_E_percent', float('nan')):.2f}% / "
            f"{rad.get('exterior_mean_rel_err_E_percent', float('nan')):.2f}%"
        )
        print(
            f"  ZX-Plane Mean / Med |E|  : "
            f"{zx.get('zx_mean_rel_diff_percent', float('nan')):.2f}% / "
            f"{zx.get('zx_median_rel_diff_percent', float('nan')):.2f}%"
        )
        e_mae = ff.get("E_plane_mae_db", float("nan"))
        h_mae = ff.get("H_plane_mae_db", float("nan"))
        xy_mae = ff.get("XY_plane_mae_db", float("nan"))
        print(f"  Far-Field MAE (E/H/XY)   : {e_mae:.2f} / {h_mae:.2f} / {xy_mae:.2f} dB")
PYEOF
echo "================================================================================"
echo "COMSOL 6.1 Dual Case A Job ${SLURM_JOB_ID} finished successfully at $(date --iso-8601=seconds)"