<#
.SYNOPSIS
    Runs COMSOL 6.1 validation for bianisotropic spherical metasurface (Case B) on Windows.
.DESCRIPTION
    Executes swept shell mesh validation and compares against SIE-BEM benchmark.
.EXAMPLE
    .\applications\anisotropic_gstc_sphere\comsol\runs\orthogonal_90deg_er7_swept.ps1
.EXAMPLE
    $env:TAG = "win_global20"; $env:GLOBAL_DIV = "20.0"; .\applications\anisotropic_gstc_sphere\comsol\runs\orthogonal_90deg_er7_swept.ps1
#>

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$BASE = "applications/anisotropic_gstc_sphere"
$RESULT = "$BASE/results/dipole_rotation"
$RUN = if ($env:RUN) { $env:RUN } else { "orthogonal_90deg_er7_offset_1m_phase_aligned_smooth002_norm_400iter" }
$NAME = if ($env:NAME) { $env:NAME } else { "orthogonal_90deg_er7_offset_1m_swept" }
$MESH = if ($env:MESH) { $env:MESH } else { "$BASE/meshes/reflectionless_100mhz_radius2m_ppw10/sphere.obj" }
$CONFIG = if ($env:CONFIG) { $env:CONFIG } else { "$BASE/configurations/moment_rot/rot90_eps7.json" }
$COMSOL_VERSION = if ($env:COMSOL_VERSION) { $env:COMSOL_VERSION } else { "6.1" }
$SHELL_MESH = if ($env:SHELL_MESH) { $env:SHELL_MESH } else { "swept" }
$TAG = if ($env:TAG) { "_$($env:TAG)" } else { "" }

$SHELL_LAYERS = if ($env:SHELL_LAYERS) { $env:SHELL_LAYERS } else { "5" }
$GRID_OVERSAMPLE = if ($env:GRID_OVERSAMPLE) { $env:GRID_OVERSAMPLE } else { "5.0" }
$GRID_METHOD = if ($env:GRID_METHOD) { $env:GRID_METHOD } else { "cubic" }
$SHELL_TAN_DIV = if ($env:SHELL_TAN_DIV) { $env:SHELL_TAN_DIV } else { "35.0" }
$GLOBAL_DIV = if ($env:GLOBAL_DIV) { $env:GLOBAL_DIV } else { "10.0" }
$INTERIOR_DIV = if ($env:INTERIOR_DIV) { $env:INTERIOR_DIV } else { "20.0" }
$SRC_RADIUS = if ($env:SRC_RADIUS) { $env:SRC_RADIUS } else { "0.15" }
$SRC_DIV = if ($env:SRC_DIV) { $env:SRC_DIV } else { "40.0" }
$INNER_WIDTH = if ($env:INNER_WIDTH) { $env:INNER_WIDTH } else { "0.05" }
$INNER_DIV = if ($env:INNER_DIV) { $env:INNER_DIV } else { "30.0" }
$OUTER_WIDTH = if ($env:OUTER_WIDTH) { $env:OUTER_WIDTH } else { "0.05" }
$OUTER_DIV = if ($env:OUTER_DIV) { $env:OUTER_DIV } else { "30.0" }
$CORES = if ($env:CORES) { $env:CORES } else { "128" }

$SIE_DIR = if ($env:SIE_DIR) { $env:SIE_DIR } else { "$RESULT/$RUN/simulation_iter400/cases/electric_incidence" }
$OUT = "$RESULT/comsol/${NAME}${TAG}"

if (Test-Path "$OUT/comsol_metrics.json") {
    Write-Error "Refusing to overwrite existing result: $OUT (set `$env:TAG = '<name>')"
    exit 1
}

New-Item -ItemType Directory -Force -Path $OUT | Out-Null

$paramsContent = @"
date=$(Get-Date -Format "yyyy-MM-ddTHH:mm:sszzz")
RUN=$RUN
MESH=$MESH
CONFIG=$CONFIG
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
CORES=$CORES
"@

Set-Content -Path "$OUT/params.txt" -Value $paramsContent

$env:PYTHONUNBUFFERED = "1"

Write-Host "Starting COMSOL $COMSOL_VERSION Validation..."
python -u "$BASE/comsol/bianisotropic_shell_validation.py" `
    --thetas "$RESULT/$RUN/optimized_thetas.bin" `
    --mesh "$MESH" `
    --config "$CONFIG" `
    --version "$COMSOL_VERSION" `
    --shell-mesh "$SHELL_MESH" `
    --shell-layers "$SHELL_LAYERS" `
    --grid-oversample "$GRID_OVERSAMPLE" `
    --grid-method "$GRID_METHOD" `
    --shell-tangential-divisor "$SHELL_TAN_DIV" `
    --global-hmax-divisor "$GLOBAL_DIV" `
    --interior-hmax-divisor "$INTERIOR_DIV" `
    --source-ball-radius "$SRC_RADIUS" `
    --source-ball-hmax-divisor "$SRC_DIV" `
    --inner-shell-band-width "$INNER_WIDTH" `
    --inner-shell-band-hmax-divisor "$INNER_DIV" `
    --outer-shell-band-width "$OUTER_WIDTH" `
    --outer-shell-band-hmax-divisor "$OUTER_DIV" `
    --output-dir "$OUT" `
    --cores "$CORES" 2>&1 | Tee-Object -FilePath "$OUT/run.log"

Write-Host "=== Comparing SIE-BEM vs COMSOL ==="
python -u "$BASE/comsol/compare_sie_comsol.py" `
    --sie-dir "$SIE_DIR" `
    --comsol-dir "$OUT" `
    --output-dir "$OUT/comparison" 2>&1 | Tee-Object -FilePath "$OUT/run.log" -Append

Write-Host "COMSOL Validation and SIE-BEM comparison completed successfully!"
