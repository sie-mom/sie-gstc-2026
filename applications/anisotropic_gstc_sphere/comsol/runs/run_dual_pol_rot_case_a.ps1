<#
.SYNOPSIS
    Windows PowerShell Driver: Case A (Zero-Thickness Sheet GSTC) Validation
    Dual Complementary Polarization Rotation (a = 0.7 * lambda0, er = 7)

.DESCRIPTION
    Solves BOTH complementary transformations on the same bianisotropic sheet
    (electric_incidence: p_z -> m_z; magnetic_incidence: m_z -> p_z) inside a
    single COMSOL 6.1 study by reusing a single MUMPS LU factorization across
    both RHS excitations.

    Enforces two-sided GSTC jump conditions directly on r = a via SurfaceCurrent
    (scu1) + SurfaceMagneticCurrentDensity (smcd1) with broken-H(curl) up()/down()
    field traces, using the Case 2 1:1 conformal band refinement around r = a.

.EXAMPLE
    # Run with default 64 cores:
    .\applications\anisotropic_gstc_sphere\comsol\runs\run_dual_pol_rot_case_a.ps1

.EXAMPLE
    # Run with explicit core count:
    .\applications\anisotropic_gstc_sphere\comsol\runs\run_dual_pol_rot_case_a.ps1 -Cores 64
    # or via environment variable:
    $env:CORES = "64"; .\applications\anisotropic_gstc_sphere\comsol\runs\run_dual_pol_rot_case_a.ps1
#>

param(
    [string]$Preset = $(if ($env:PRESET) { $env:PRESET } else { "case2" }),
    [int]$Cores = $(if ($env:CORES) { [int]$env:CORES } else { 64 }),
    [string]$CaseId = $(if ($env:CASE_ID) { $env:CASE_ID } else { "" }),
    [string]$ComsolVersion = $(if ($env:COMSOL_VERSION) { $env:COMSOL_VERSION } else { "6.1" }),
    [string]$LinearSolver = $(if ($env:LINEAR_SOLVER) { $env:LINEAR_SOLVER } else { "mumps" }),
    [string]$Discretization = "linear",
    [string]$PythonBin = "python",
    [switch]$PlotsOnly = $(if ($env:PLOTS_ONLY -eq "1" -or $env:PLOTS_ONLY -eq "true") { $true } else { $false })
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = "."

# Resolve python from active environment
$ResolvedPython = (Get-Command $PythonBin -ErrorAction SilentlyContinue).Source
if (-not $ResolvedPython) {
    $ResolvedPython = $PythonBin
}

# ------------------------------------------------------------------------------
# Controlled Mesh Convergence Presets (Case A Zero-Thickness Sheet)
# ------------------------------------------------------------------------------
switch ($Preset) {
    "case1" {
        $SheetTanDiv = 25.0
        $GlobalDiv = 20.0
        $InteriorDiv = 20.0
        $InnerBandWidth = 0.0
        $InnerBandDiv = 0.0
        $OuterBandWidth = 0.0
        $OuterBandDiv = 0.0
        $SourceBallRadius = 0.1
        $SourceBallDiv = 60.0
        $FfOffsetLam = 0.10
    }
    "case2" {
        # Conformal Band Refinement (Default): 1:1 aspect ratio volume tetrahedra at r = a
        $SheetTanDiv = 30.0
        $GlobalDiv = 20.0
        $InteriorDiv = 20.0
        $InnerBandWidth = 0.08
        $InnerBandDiv = 30.0
        $OuterBandWidth = 0.10
        $OuterBandDiv = 30.0
        $SourceBallRadius = 0.1
        $SourceBallDiv = 60.0
        $FfOffsetLam = 0.10
    }
    "case3" {
        $SheetTanDiv = 35.0
        $GlobalDiv = 20.0
        $InteriorDiv = 20.0
        $InnerBandWidth = 0.08
        $InnerBandDiv = 35.0
        $OuterBandWidth = 0.10
        $OuterBandDiv = 35.0
        $SourceBallRadius = 0.1
        $SourceBallDiv = 60.0
        $FfOffsetLam = 0.10
    }
    default {
        Write-Host "Unknown preset '$Preset', defaulting to case2."
        $SheetTanDiv = 30.0
        $GlobalDiv = 20.0
        $InteriorDiv = 20.0
        $InnerBandWidth = 0.08
        $InnerBandDiv = 30.0
        $OuterBandWidth = 0.10
        $OuterBandDiv = 30.0
        $SourceBallRadius = 0.1
        $SourceBallDiv = 60.0
        $FfOffsetLam = 0.10
    }
}

# At f0 = 10 GHz, lambda0 = c / 10 GHz = 0.0299792458 m
# a = 0.7 * lambda0 = 0.020985472 m, air-extent b = 0.95 * lambda0 = 0.02848028351 m
$AirExtentM = 0.02848028351
$RadialPoints = 300
$RadialMaxRadii = 2.0
$ZxNx = 200
$ZxNz = 200
$ZxSpanRadii = 2.0

$CaseDir = "applications/anisotropic_gstc_sphere/results/dipole_pol_rot/dual_pol_eps7"
if (-not (Test-Path $CaseDir)) {
    $CaseDir = "applications/anisotropic_gstc_sphere/results/dipole_pol_rot/dual_pol_rot_10ghz_eps7_reciprocal_lossless"
}
$Tensors = "$CaseDir/face_tensors.bin"
$Mesh = "applications/anisotropic_gstc_sphere/meshes/example_1_omega_matching_10ghz_eps7/sphere_ppw6.obj"
$Config = "$CaseDir/configuration.json"
if (-not (Test-Path $Config)) {
    $Config = "applications/anisotropic_gstc_sphere/configurations/dipole_pol_rot/dual_pol_eps7.json"
}

# Short output folder name on Windows to prevent MAX_PATH issues
$RunName = "casea_${Preset}_comsol_dual"
$OutDir = "$CaseDir/comsol/$RunName"
New-Item -ItemType Directory -Force -Path "$OutDir/logs" | Out-Null

$ParamsContent = @"
date=$(Get-Date -Format "yyyy-MM-ddTHH:mm:sszzz")
PRESET=$Preset
CORES=$Cores
PYTHON=$ResolvedPython
COMSOL_VERSION=$ComsolVersion
LINEAR_SOLVER=$LinearSolver
DISCRETIZATION=$Discretization
SHEET_TAN_DIV=$SheetTanDiv
GLOBAL_DIV=$GlobalDiv
INTERIOR_DIV=$InteriorDiv
INNER_BAND_WIDTH=$InnerBandWidth
INNER_BAND_DIV=$InnerBandDiv
OUTER_BAND_WIDTH=$OuterBandWidth
OUTER_BAND_DIV=$OuterBandDiv
SOURCE_BALL_RADIUS=$SourceBallRadius
SOURCE_BALL_DIV=$SourceBallDiv
FF_OFFSET_LAM=$FfOffsetLam
OUTPUT_DIR=$OutDir
"@
Set-Content -Path "$OutDir/params.txt" -Value $ParamsContent

Write-Host "================================================================================"
Write-Host "COMSOL $ComsolVersion Case A (Zero-Thickness Sheet) Validation: Dual Pol Rot (Windows)"
Write-Host "Date             : $(Get-Date -Format s)"
Write-Host "Python           : $ResolvedPython"
Write-Host "Preset           : $Preset"
Write-Host "CPUs allocated   : $Cores"
Write-Host "Linear Solver    : $LinearSolver (single LU factorization across 2 RHS cases)"
Write-Host "Tensors file     : $Tensors"
Write-Host "Mesh file        : $Mesh"
Write-Host "Configuration    : $Config"
Write-Host "Output Directory : $OutDir"
Write-Host "Sheet Tan Div    : $SheetTanDiv (Huygens FF offset = $FfOffsetLam * lambda0)"
Write-Host "================================================================================"

if (-not (Test-Path $Tensors)) {
    Write-Error "ERROR: Tensors file not found at $Tensors`nPlease ensure SIE results were copied."
    exit 1
}

$ExtraBandArgs = @()
if ($InnerBandWidth -gt 0.0) {
    $ExtraBandArgs += @("--inner-sheet-band-width", "$InnerBandWidth", "--inner-sheet-band-hmax-divisor", "$InnerBandDiv")
}
if ($OuterBandWidth -gt 0.0) {
    $ExtraBandArgs += @("--outer-sheet-band-width", "$OuterBandWidth", "--outer-sheet-band-hmax-divisor", "$OuterBandDiv")
}

$TargetCases = if ($CaseId) { @($CaseId) } else { @("electric_incidence", "magnetic_incidence") }
$ExtraCaseArgs = if ($CaseId) { @("--case-id", "$CaseId") } else { @() }

# 1. Execute Bianisotropic Case A COMSOL Validation Pipeline
if (-not $PlotsOnly) {
    $SolveHeaderMsg = if ($CaseId) { "Single Case ($CaseId)" } else { "All Cases (LU reuse)" }
    Write-Host "`n>>> [1/4] Running Bianisotropic Case A COMSOL Validation Pipeline ($SolveHeaderMsg) <<<"
    & $ResolvedPython -u applications/anisotropic_gstc_sphere/comsol/bianisotropic_sheet_validation.py `
        --tensors "$Tensors" `
        --mesh "$Mesh" `
        --config "$Config" `
        --output-dir "$OutDir" `
        --version "$ComsolVersion" `
        --cores $Cores `
        --linear-solver "$LinearSolver" `
        --discretization "$Discretization" `
        --sheet-tangential-divisor $SheetTanDiv `
        --global-mesh-divisor $GlobalDiv `
        --interior-hmax-divisor $InteriorDiv `
        --source-ball-radius $SourceBallRadius `
        --source-ball-hmax-divisor $SourceBallDiv `
        --far-field-offset-lambda $FfOffsetLam `
        $ExtraBandArgs `
        $ExtraCaseArgs `
        --air-extent $AirExtentM `
        --radial-points $RadialPoints `
        --radial-max-radii $RadialMaxRadii `
        --radial-theta 45.0 `
        --radial-phi 0.0 `
        --zx-nx $ZxNx `
        --zx-nz $ZxNz `
        --zx-span-radii $ZxSpanRadii 2>&1 | Tee-Object -FilePath "$OutDir/run.log"

    if ($LASTEXITCODE -ne 0) {
        Write-Error "COMSOL validation pipeline failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
} else {
    Write-Host "`n>>> [SKIPPED] Step 1: COMSOL Solve (-PlotsOnly specified, using existing results) <<<"
}

# 2. Ensure SIE Reference Far-Field Cuts Exist
Write-Host "`n>>> [2/4] Checking SIE Reference Far-Field Cuts <<<"
$MissingFf = $false
foreach ($CId in $TargetCases) {
    $SieFf1 = "$CaseDir/cases/$CId/far_field/far_field_cuts.csv"
    $SieFf2 = "$CaseDir/cases/$CId/far_field_cuts.csv"
    if (-not (Test-Path $SieFf1) -and -not (Test-Path $SieFf2)) {
        $MissingFf = $true
    }
}
if ($MissingFf) {
    Write-Host "Generating missing SIE reference far-field cuts..."
    & $ResolvedPython -u applications/anisotropic_gstc_sphere/plot_far_field.py `
        --run-dir "$CaseDir" `
        --cases $TargetCases 2>&1 | Tee-Object -FilePath "$OutDir/run.log" -Append
}

# 3. Running Benchmark Comparisons Against SIE Reference
Write-Host "`n>>> [3/4] Running Benchmark Comparisons Against SIE Reference <<<"
foreach ($CId in $TargetCases) {
    $SieCaseDir = "$CaseDir/cases/$CId"
    $ComsolCaseDir = "$OutDir/cases/$CId"
    if ((Test-Path $SieCaseDir) -and (Test-Path $ComsolCaseDir)) {
        Write-Host "`n--- Comparing Case: $CId ---"
        & $ResolvedPython -u applications/anisotropic_gstc_sphere/comsol/compare_sie_comsol.py `
            --sie-dir "$SieCaseDir" `
            --comsol-dir "$ComsolCaseDir" 2>&1 | Tee-Object -FilePath "$OutDir/run.log" -Append
    }
}

# 4. Generating Physical Field Comparison Plots via plot_fields.py (Spherical Basis)
Write-Host "`n>>> [4/4] Generating Physical Field Plots (Spherical Basis) <<<"
$RefTransforms = @{
    "electric_incidence" = "electric_to_magnetic"
    "magnetic_incidence" = "magnetic_to_electric"
}

foreach ($CId in $TargetCases) {
    $RefId = $RefTransforms[$CId]
    Write-Host "`n--- Plotting for $CId (target: $RefId) ---"

    # Full spherical component suite
    & $ResolvedPython -u applications/anisotropic_gstc_sphere/plot_fields.py `
        --run-dir "$OutDir" `
        --case "$CId" `
        --reference-transformation "$RefId" `
        --line-basis spherical `
        --map-basis spherical `
        --map-percentiles 20 98 `
        --magnitude-scale db 2>&1 | Tee-Object -FilePath "$OutDir/run.log" -Append

    # Dedicated mode rotation suite (E_theta TM vs E_phi TE)
    & $ResolvedPython -u applications/anisotropic_gstc_sphere/plot_fields.py `
        --run-dir "$OutDir" `
        --case "$CId" `
        --reference-transformation "$RefId" `
        --line-basis spherical `
        --map-basis spherical `
        --line-components E_theta E_phi `
        --map-components E_theta E_phi `
        --map-percentiles 20 98 `
        --magnitude-scale db 2>&1 | Tee-Object -FilePath "$OutDir/run.log" -Append
}

# 5. Print Final Summary Metrics
Write-Host "`n================================================================================"
Write-Host "SUMMARY METRICS (DUAL TRANSFORMATION CASE A):"
Write-Host "================================================================================"

$PySummaryCode = @'
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
'@

$PySummaryCode | & $ResolvedPython - "$OutDir" 2>&1 | Tee-Object -FilePath "$OutDir/run.log" -Append

Write-Host "================================================================================"
Write-Host "COMSOL Dual Case A Validation finished successfully at $(Get-Date -Format s)"
