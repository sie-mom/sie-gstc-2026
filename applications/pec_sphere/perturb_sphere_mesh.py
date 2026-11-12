#!/usr/bin/env python3
"""Break an icosphere's vertex symmetry while preserving its faces and radius."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sie_mom.mesh.obj_mesh import load_triangle_obj, validate_triangle_mesh


def minimum_triangle_angle(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    angles = []
    for corner in range(3):
        first = triangles[:, (corner + 1) % 3] - triangles[:, corner]
        second = triangles[:, (corner + 2) % 3] - triangles[:, corner]
        cosine = np.sum(first * second, axis=1) / (
            np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        )
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return float(np.min(angles))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.05,
                        help="Maximum tangent displacement divided by median edge length")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 0.0 < args.fraction < 1.0:
        parser.error("--fraction must be between 0 and 1")
    if args.input.resolve() == args.output.resolve():
        parser.error("--input and --output must differ")

    original = load_triangle_obj(args.input)
    validate_triangle_mesh(original).require_valid(expected_euler_characteristic=2)
    radii = np.linalg.norm(original.vertices, axis=1)
    radius = float(np.median(radii))
    if np.max(np.abs(radii - radius)) > 1e-5 * radius:
        parser.error("input vertices must lie on one sphere centered at the origin")

    edges = np.vstack((original.faces[:, [0, 1]], original.faces[:, [1, 2]],
                       original.faces[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    median_edge = float(np.median(np.linalg.norm(
        original.vertices[edges[:, 0]] - original.vertices[edges[:, 1]], axis=1
    )))

    rng = np.random.default_rng(args.seed)
    normals = original.vertices / radii[:, None]
    random_vectors = rng.normal(size=original.vertices.shape)
    tangents = random_vectors - np.sum(random_vectors * normals, axis=1)[:, None] * normals
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    distances = rng.uniform(0.0, args.fraction * median_edge, len(normals))
    moved = original.vertices + distances[:, None] * tangents
    moved *= radius / np.linalg.norm(moved, axis=1)[:, None]

    lines = args.input.read_text().splitlines()
    vertex_index = 0
    for line_index, line in enumerate(lines):
        if line.startswith("v "):
            x, y, z = moved[vertex_index]
            lines[line_index] = f"v {x:.17g} {y:.17g} {z:.17g}"
            vertex_index += 1
    assert vertex_index == len(moved)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")

    result = load_triangle_obj(args.output)
    report = validate_triangle_mesh(result)
    report.require_valid(expected_euler_characteristic=2)
    assert np.array_equal(result.faces, original.faces)
    print(f"mesh={args.output} faces={report.face_count} edges={report.edge_count}")
    print(f"radius={radius:.9g} m, maximum displacement={args.fraction * median_edge:.9g} m")
    print(f"minimum triangle angle: {minimum_triangle_angle(original.vertices, original.faces):.2f}"
          f" -> {minimum_triangle_angle(result.vertices, result.faces):.2f} degrees")


if __name__ == "__main__":
    main()
