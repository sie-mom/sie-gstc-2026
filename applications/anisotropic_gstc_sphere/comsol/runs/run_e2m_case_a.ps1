<#
.SYNOPSIS
    Windows PowerShell Driver: Case A (Zero-Thickness Sheet GSTC) Validation
    Electric to Magnetic Polarization Rotation (a = 0.7 * lambda0, er = 7)

.DESCRIPTION
    Replicates the benchmark TinkerCliffs Job 8378521 on Windows.
    Solves the electric-to-magnetic (p_z -> m_z) transformation on the bianisotropic
    sheet inside COMSOL 6.1 with MUMPS direct solver.

    Uses the Case 2 1:1 conformal band refinement around r = a:
    - Sheet Tan Div: 30.0
    - Inner/Outer Conformal Bands: 30.0
    - Source Ball: radius 0.1*a, div 60.0
    - Global/Interior: div 20.0

.EXAMPLE
    # Run with default 64 cores:
    .\applications\anisotropic_gstc_sphere\comsol\runs\run_e2m_case_a.ps1

.EXAMPLE
    # Run with explicit core count:
    .\applications\anisotropic_gstc_sphere\comsol\runs\run_e2m_case_a.ps1 -Cores 64
#>

param(
    [string]$Preset = $(if ($env:PRESET) { $env:PRESET } else { "case2" }),
    [int]$Cores = $(if ($env:CORES) { [int]$env:CORES } else { 64 }),
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
        # Conformal Band Refinement (Job 8378521 Benchmark): 1:1 aspect ratio volume tetrahedra at r = a
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

$CaseDir = "applications/anisotropic_gstc_sphere/results/dipole_pol_rot/e2m_eps7"
if (-not (Test-Path $CaseDir)) {
    $CaseDir = "applications/anisotropic_gstc_sphere/results/dipole_pol_rot/electric_to_magnetic_10ghz_eps7_reciprocal_lossless"
}
$Tensors = "$CaseDir/face_tensors.bin"
$Mesh = "applications/anisotropic_gstc_sphere/meshes/example_1_omega_matching_10ghz_eps7/sphere_ppw6.obj"
$Config = "$CaseDir/configuration.json"
if (-not (Test-Path $Config)) {
    $Config = "applications/anisotropic_gstc_sphere/configurations/dipole_pol_rot/e2m_eps7.json"
}
$SieDir = "$CaseDir/cases/electric_incidence"

# Short output folder name on Windows to prevent MAX_PATH issues
$RunName = "casea_${Preset}_comsol_e2m"
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
Write-Host "COMSOL $ComsolVersion Case A (Zero-Thickness Sheet) Validation: E2M Pol Rot (Windows)"
Write-Host "Replicating TinkerCliffs Job 8378521 Benchmark (7.45% radial L2 error)"
Write-Host "Date             : $(Get-Date -Format s)"
Write-Host "Python           : $ResolvedPython"
Write-Host "Preset           : $Preset"
Write-Host "CPUs allocated   : $Cores"
Write-Host "Linear Solver    : $LinearSolver"
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

# 1. Execute Bianisotropic Case A COMSOL Validation Pipeline (electric_incidence)
if (-not $PlotsOnly) {
    Write-Host "`n>>> [1/4] Running Bianisotropic Case A COMSOL Validation Pipeline <<<"
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
$SieFf1 = "$SieDir/far_field/far_field_cuts.csv"
$SieFf2 = "$SieDir/far_field_cuts.csv"
if (-not (Test-Path $SieFf1) -and -not (Test-Path $SieFf2)) {
    Write-Host "Generating missing SIE reference far-field cuts..."
    & $ResolvedPython -u applications/anisotropic_gstc_sphere/plot_far_field.py `
        --run-dir "$CaseDir" `
        --cases electric_incidence 2>&1 | Tee-Object -FilePath "$OutDir/run.log" -Append
}

# 3. Running Benchmark Comparison Against SIE Reference
Write-Host "`n>>> [3/4] Running Benchmark Comparison Against SIE Reference <<<"
if (Test-Path $SieDir) {
    Write-Host "`n--- Comparing against SIE Reference ($SieDir) ---"
    & $ResolvedPython -u applications/anisotropic_gstc_sphere/comsol/compare_sie_comsol.py `
        --sie-dir "$SieDir" `
        --comsol-dir "$OutDir" 2>&1 | Tee-Object -FilePath "$OutDir/run.log" -Append
}

# 4. Generating Physical Field Comparison Plots via plot_fields.py (Spherical Basis)
Write-Host "`n>>> [4/4] Generating Physical Field Comparison Plots via plot_fields.py (Spherical Basis) <<<"
# 1. Full spherical component suite (norms and all spherical vector components)
Write-Host "`n--- 1. Full spherical component suite (norms and all spherical vector components) ---"
& $ResolvedPython -u applications/anisotropic_gstc_sphere/plot_fields.py `
    --run-dir "$OutDir" `
    --case electric_incidence `
    --reference-transformation electric_to_magnetic `
    --line-basis spherical `
    --map-basis spherical `
    --map-percentiles 20 98 `
    --magnitude-scale db 2>&1 | Tee-Object -FilePath "$OutDir/run.log" -Append

# 2. Dedicated mode rotation comparison (E_theta TM mode vs E_phi TE mode)
Write-Host "`n--- 2. Dedicated mode rotation comparison (E_theta TM mode vs E_phi TE mode) ---"
& $ResolvedPython -u applications/anisotropic_gstc_sphere/plot_fields.py `
    --run-dir "$OutDir" `
    --case electric_incidence `
    --reference-transformation electric_to_magnetic `
    --line-basis spherical `
    --map-basis spherical `
    --line-components E_theta E_phi `
    --map-components E_theta E_phi `
    --map-percentiles 20 98 `
    --magnitude-scale db 2>&1 | Tee-Object -FilePath "$OutDir/run.log" -Append

Write-Host "`n================================================================================"
Write-Host "SUMMARY METRICS:"
Write-Host "================================================================================"
$ComsolMetricsPath = "$OutDir/comsol_metrics.json"
if (Test-Path $ComsolMetricsPath) {
    $cm = Get-Content $ComsolMetricsPath -Raw | ConvertFrom-Json
    if ($cm.mesh) {
        Write-Host ("Total Mesh Elements : {0:N0}" -f $cm.mesh.total_elements)
        Write-Host ("Sheet Triangles     : {0:N0}" -f $cm.mesh.sheet_triangles)
        Write-Host ("Volume Tetrahedra   : {0:N0}" -f $cm.mesh.tetrahedra)
    }
    if ($cm.solver) {
        Write-Host ("Solve Duration      : {0:N2} s" -f $cm.solver.solve_duration_seconds)
    }
    Write-Host ("Total Execution     : {0:N2} s" -f $cm.total_execution_seconds)
}
Write-Host "================================================================================"
Write-Host "Case A E2M Validation Completed."
Write-Host "Results saved to: $OutDir"
Write-Host "================================================================================"

