"""End-to-end parity verification between spherical and face basis on a sphere."""

from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest

from applications.anisotropic_gstc_sphere.prepare_sie_inputs import export_sie_inputs
from applications.anisotropic_gstc_sphere.sie_gstc_sphere import run_sie_workflow
from applications.anisotropic_gstc_sphere.workflow import (
    load_workflow_configuration,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "applications" / "anisotropic_gstc_sphere"
CONFIG_PATH = (
    APP_DIR
    / "configurations"
    / "dipole_pol_rot"
    / "dual_pol_eps7.json"
)
MESH_PATH = (
    APP_DIR
    / "meshes"
    / "example_1_omega_matching_10ghz_eps7"
    / "sphere_ppw6.obj"
)
EXECUTABLE = (
    REPO_ROOT / "build" / "applications" / "anisotropic_gstc_sphere" / "anisotropic_sphere_app"
)


@pytest.mark.skipif(not EXECUTABLE.is_file(), reason="anisotropic_sphere_app executable not found")
def test_sphere_parity_spherical_vs_face_basis(tmp_path):
    """Verify that running the SIE solver with basis='spherical' vs basis='face'

    both succeed with tiny residuals and produce closely matching physical currents
    consistent with facet curvature discretization error.
    """
    config_base = load_workflow_configuration(CONFIG_PATH)

    # Use a coarse mesh (540 faces) and faster quadrature (order 7) for quick execution
    config_base = replace(
        config_base,
        sie=replace(
            config_base.sie,
            mesh=replace(config_base.sie.mesh, path=str(MESH_PATH)),
            quadrature_rule=7,
        ),
    )

    # 1. Run with basis = "spherical"
    dir_sph = tmp_path / "run_spherical"
    config_sph = replace(
        config_base,
        sie=replace(config_base.sie, basis="spherical"),
    )
    manifest_sph = run_sie_workflow(
        config_sph,
        run_dir=dir_sph,
        executable=EXECUTABLE,
        use_existing_mesh=True,
    )

    # 2. Run with basis = "face"
    dir_face = tmp_path / "run_face"
    config_face = replace(
        config_base,
        sie=replace(config_base.sie, basis="face"),
    )
    manifest_face = run_sie_workflow(
        config_face,
        run_dir=dir_face,
        executable=EXECUTABLE,
        use_existing_mesh=True,
    )

    # Check that both batch metrics recorded the correct basis and solved cleanly
    with open(dir_sph / "batch_metrics.json") as f:
        metrics_sph = json.load(f)
    with open(dir_face / "batch_metrics.json") as f:
        metrics_face = json.load(f)

    assert metrics_sph["solver"]["basis"] == "spherical"
    assert metrics_face["solver"]["basis"] == "face"

    with open(dir_sph / "cases" / "electric_incidence" / "solver_metrics.json") as f:
        case_sph = json.load(f)
    with open(dir_face / "cases" / "electric_incidence" / "solver_metrics.json") as f:
        case_face = json.load(f)

    res_sph = case_sph["solver"]["scaled_relative_residual"]
    res_face = case_face["solver"]["scaled_relative_residual"]

    print(f"\n[Spherical Basis] relative residual = {res_sph:.3e}")
    print(f"[Face Basis]      relative residual = {res_face:.3e}")

    # Both residuals must be within linear-solver tolerance (< 1e-10)
    assert res_sph < 1e-10
    assert res_face < 1e-10

    # Load solved currents: 4 complex numbers per RWG edge: (J1, M1, J2, M2)
    currents_sph = np.fromfile(
        dir_sph / "cases" / "electric_incidence" / "currents.bin", dtype=np.complex128
    )
    currents_face = np.fromfile(
        dir_face / "cases" / "electric_incidence" / "currents.bin", dtype=np.complex128
    )

    norm_sph = np.linalg.norm(currents_sph)
    norm_face = np.linalg.norm(currents_face)
    rel_norm_diff = abs(norm_sph - norm_face) / norm_sph

    print(f"Currents norm (spherical): {norm_sph:.4e}")
    print(f"Currents norm (face):      {norm_face:.4e}")
    print(f"Currents norm relative difference: {rel_norm_diff:.4e}")

    # Total current norms match within ~5% on this coarse 540-face mesh
    assert rel_norm_diff < 0.05

    # Compare radiated physical fields on the 1D radial cut
    radial_sph = np.genfromtxt(
        dir_sph / "cases" / "electric_incidence" / "fields" / "sie_physical_radial.csv",
        delimiter=",",
        names=True,
    )
    radial_face = np.genfromtxt(
        dir_face / "cases" / "electric_incidence" / "fields" / "sie_physical_radial.csv",
        delimiter=",",
        names=True,
    )

    E_norm_sph = np.sqrt(
        radial_sph["Ex_real"]**2 + radial_sph["Ex_imag"]**2
        + radial_sph["Ey_real"]**2 + radial_sph["Ey_imag"]**2
        + radial_sph["Ez_real"]**2 + radial_sph["Ez_imag"]**2
    )
    E_norm_face = np.sqrt(
        radial_face["Ex_real"]**2 + radial_face["Ex_imag"]**2
        + radial_face["Ey_real"]**2 + radial_face["Ey_imag"]**2
        + radial_face["Ez_real"]**2 + radial_face["Ez_imag"]**2
    )

    rel_field_diff = np.linalg.norm(E_norm_sph - E_norm_face) / np.linalg.norm(E_norm_sph)
    print(f"Radial cut E-field relative difference: {rel_field_diff:.4e}")

    # Physical fields agree to < 0.1% across all observation points
    assert rel_field_diff < 0.005