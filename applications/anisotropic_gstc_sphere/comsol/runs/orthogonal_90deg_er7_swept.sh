#!/usr/bin/env bash
# COMSOL validation: orthogonal 90deg dipole rotation, er=7, 1 m offset, swept shell mesh.
#
# Defaults reproduce the baseline. Override any parameter from the command line, e.g.:
#   TAG=layers7 SHELL_LAYERS=7 bash orthogonal_90deg_er7_swept.sh
# Each run writes to results/.../comsol/<NAME>_<TAG>/ and saves its own parameters
# (params.txt) and log (run.log) there, so every result records how it was made.
# After the COMSOL run, compare_sie_comsol.py writes plots to <output>/comparison.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"

BASE=applications/anisotropic_gstc_sphere
RESULT=$BASE/results/dipole_rotation
RUN=${RUN:-orthogonal_90deg_er7_offset_1m_phase_aligned_smooth002_norm_400iter}
NAME=${NAME:-orthogonal_90deg_er7_offset_1m_swept}
MESH=${MESH:-$BASE/meshes/reflectionless_100mhz_radius2m_ppw10/sphere.obj}
CONFIG=${CONFIG:-$BASE/configurations/moment_rot/rot90_eps7.json}
COMSOL_VERSION=${COMSOL_VERSION:-5.4}
SHELL_MESH=${SHELL_MESH:-swept}
TAG=${TAG:-}

SHELL_LAYERS=${SHELL_LAYERS:-5}
GRID_OVERSAMPLE=${GRID_OVERSAMPLE:-5.0}
GRID_METHOD=${GRID_METHOD:-cubic}
SHELL_TAN_DIV=${SHELL_TAN_DIV:-35.0}
GLOBAL_DIV=${GLOBAL_DIV:-6.0}
INTERIOR_DIV=${INTERIOR_DIV:-20.0}
SRC_RADIUS=${SRC_RADIUS:-0.15}
SRC_DIV=${SRC_DIV:-40.0}
INNER_WIDTH=${INNER_WIDTH:-0.05}
INNER_DIV=${INNER_DIV:-30.0}
OUTER_WIDTH=${OUTER_WIDTH:-0.05}
OUTER_DIV=${OUTER_DIV:-30.0}
CORES=${CORES:-20}

SIE_DIR=${SIE_DIR:-$RESULT/$RUN/simulation_iter400/cases/electric_incidence}
OUT=$RESULT/comsol/${NAME}${TAG:+_$TAG}
if [[ -e "$OUT/comsol_metrics.json" ]]; then
    echo "Refusing to overwrite existing result: $OUT (set TAG=<name>)" >&2
    exit 1
fi
mkdir -p "$OUT"

{
    date -Is
    echo "git: $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
    for v in RUN MESH CONFIG COMSOL_VERSION SHELL_MESH SHELL_LAYERS GRID_OVERSAMPLE GRID_METHOD SHELL_TAN_DIV GLOBAL_DIV INTERIOR_DIV \
             SRC_RADIUS SRC_DIV INNER_WIDTH INNER_DIV OUTER_WIDTH OUTER_DIV CORES; do
        echo "$v=${!v}"
    done
} > "$OUT/params.txt"

PYTHONUNBUFFERED=1 .venv/bin/python "$BASE/comsol/bianisotropic_shell_validation.py" \
    --thetas "$RESULT/$RUN/optimized_thetas.bin" \
    --mesh "$MESH" \
    --config "$CONFIG" \
    --version "$COMSOL_VERSION" \
    --shell-mesh "$SHELL_MESH" \
    --shell-layers "$SHELL_LAYERS" \
    --grid-oversample "$GRID_OVERSAMPLE" \
    --grid-method "$GRID_METHOD" \
    --shell-tangential-divisor "$SHELL_TAN_DIV" \
    --global-hmax-divisor "$GLOBAL_DIV" \
    --interior-hmax-divisor "$INTERIOR_DIV" \
    --source-ball-radius "$SRC_RADIUS" \
    --source-ball-hmax-divisor "$SRC_DIV" \
    --inner-shell-band-width "$INNER_WIDTH" \
    --inner-shell-band-hmax-divisor "$INNER_DIV" \
    --outer-shell-band-width "$OUTER_WIDTH" \
    --outer-shell-band-hmax-divisor "$OUTER_DIV" \
    --output-dir "$OUT" \
    --cores "$CORES" \
    2>&1 | tee "$OUT/run.log"

echo "=== Comparing SIE-BEM vs COMSOL ===" | tee -a "$OUT/run.log"
.venv/bin/python "$BASE/comsol/compare_sie_comsol.py" \
    --sie-dir "$SIE_DIR" \
    --comsol-dir "$OUT" \
    --output-dir "$OUT/comparison" \
    2>&1 | tee -a "$OUT/run.log"
