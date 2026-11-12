"""Shared analytical electromagnetic excitation fields."""

from .fields import (
    DipoleType,
    evaluate_dipole_fields,
    evaluate_dipole_fields_cartesian,
    evaluate_plane_wave_fields,
    evaluate_plane_wave_cartesian,
)

__all__ = [
    "DipoleType",
    "evaluate_dipole_fields",
    "evaluate_dipole_fields_cartesian",
    "evaluate_plane_wave_fields",
    "evaluate_plane_wave_cartesian",
]
