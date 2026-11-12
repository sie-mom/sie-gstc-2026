"""Shared CLI helpers for application-local mesh wrappers.

Goal: keep application scripts very small by centralizing common argparse,
mesh-size resolution, and generator dispatch for each shape.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from sie_mom.mesh.mesh_generators import (
    BluntedConeSphereGenerator,
    BluntedConeSphereParams,
    CommonMeshOptions,
    ConeSphereGenerator,
    ConeSphereParams,
    SphereGenerator,
    SphereParams,
)


@dataclass
class AppContext:
    repo_root: Path
    app_name: str
    default_out: str
    preset_mesh_size: Dict[str, float]


def _resolve_mesh_size(
    mesh_size: float | None,
    lambda0: float | None,
    ppw: float,
    preset: str,
    preset_map: Dict[str, float],
) -> float:
    if mesh_size is not None:
        return mesh_size
    if lambda0 is not None:
        if ppw <= 0.0:
            raise ValueError("--ppw must be > 0.")
        return lambda0 / ppw
    return preset_map[preset]


def _add_common_mesh_args(p: argparse.ArgumentParser, preset_map: Dict[str, float], default_out: str) -> None:
    p.add_argument("--mesh-size", type=float, default=None, help="Target max edge size")
    p.add_argument("--lambda0", type=float, default=None, help="Wavelength (if used, mesh-size = lambda0 / ppw)")
    p.add_argument("--ppw", type=float, default=10.0, help="Points per wavelength")
    p.add_argument("--preset", choices=sorted(preset_map.keys()), default="medium")
    p.add_argument("--out", default=default_out)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--gui", action="store_true", help="Open Gmsh GUI")


def run_sphere_app(ctx: AppContext) -> int:
    p = argparse.ArgumentParser(description=f"Generate sphere mesh for {ctx.app_name}.")
    p.add_argument("--radius", type=float, default=1.0, help="Sphere radius")
    _add_common_mesh_args(p, ctx.preset_mesh_size, ctx.default_out)
    args = p.parse_args()

    h = _resolve_mesh_size(args.mesh_size, args.lambda0, args.ppw, args.preset, ctx.preset_mesh_size)
    out_path = (ctx.repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    opts = CommonMeshOptions(
        output_obj=str(out_path),
        mesh_size_min=0.8 * h,
        mesh_size_max=h,
        headless=not args.gui,
        model_name=ctx.app_name,
    )
    SphereGenerator(SphereParams(radius=args.radius), opts).generate()
    print(f"Generated: {out_path}")
    return 0


def run_cone_sphere_app(ctx: AppContext) -> int:
    p = argparse.ArgumentParser(description=f"Generate cone-sphere mesh for {ctx.app_name}.")
    p.add_argument("--sphere-radius", type=float, default=0.1)
    p.add_argument("--cone-length", type=float, default=0.6)
    _add_common_mesh_args(p, ctx.preset_mesh_size, ctx.default_out)
    args = p.parse_args()

    h = _resolve_mesh_size(args.mesh_size, args.lambda0, args.ppw, args.preset, ctx.preset_mesh_size)
    out_path = (ctx.repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    opts = CommonMeshOptions(
        output_obj=str(out_path),
        mesh_size_min=0.8 * h,
        mesh_size_max=h,
        headless=not args.gui,
        model_name=ctx.app_name,
    )
    ConeSphereGenerator(
        ConeSphereParams(sphere_radius=args.sphere_radius, cone_length=args.cone_length),
        opts,
    ).generate()
    print(f"Generated: {out_path}")
    return 0


def run_blunted_cone_sphere_app(ctx: AppContext) -> int:
    p = argparse.ArgumentParser(description=f"Generate blunted cone-sphere mesh for {ctx.app_name}.")
    p.add_argument("--sphere-radius", type=float, default=0.1)
    p.add_argument("--cone-length", type=float, default=0.6)
    p.add_argument("--nose-radius", type=float, default=0.02)
    _add_common_mesh_args(p, ctx.preset_mesh_size, ctx.default_out)
    args = p.parse_args()

    h = _resolve_mesh_size(args.mesh_size, args.lambda0, args.ppw, args.preset, ctx.preset_mesh_size)
    out_path = (ctx.repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    opts = CommonMeshOptions(
        output_obj=str(out_path),
        mesh_size_min=0.8 * h,
        mesh_size_max=h,
        headless=not args.gui,
        model_name=ctx.app_name,
    )
    BluntedConeSphereGenerator(
        BluntedConeSphereParams(
            sphere_radius=args.sphere_radius,
            cone_length=args.cone_length,
            nose_radius=args.nose_radius,
        ),
        opts,
    ).generate()
    print(f"Generated: {out_path}")
    return 0
