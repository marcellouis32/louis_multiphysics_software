"""Published correlations used as validation targets and as surrogate prior means.

These are not solver outputs. They serve three roles:
  1. acceptance bounds in the V&V regression suite,
  2. the mean function of the hybrid/ML surrogates, so a model is useful after
     ~20 CFD runs instead of ~2000,
  3. sanity bounds the AI setup agent checks its own results against.
"""

from __future__ import annotations

from enum import Enum


class ImpellerType(str, Enum):
    RUSHTON_6 = "rushton_6blade"
    PBT_4_45 = "pitched_blade_4x45"
    PBT_6_45 = "pitched_blade_6x45"
    HYDROFOIL_A310 = "hydrofoil_a310"
    HYDROFOIL_A320 = "hydrofoil_a320"
    ANCHOR = "anchor"
    HELICAL_RIBBON = "helical_ribbon"
    CUSTOM_STL = "custom_stl"


# Fully turbulent (Re > 1e4), fully baffled, single impeller, standard geometry.
# Ranges are literature spread, not uncertainty in any single measurement.
REFERENCE_POWER_NUMBER: dict[ImpellerType, tuple[float, float]] = {
    ImpellerType.RUSHTON_6: (4.8, 5.5),
    ImpellerType.PBT_4_45: (1.2, 1.6),
    ImpellerType.PBT_6_45: (1.6, 2.0),
    ImpellerType.HYDROFOIL_A310: (0.28, 0.35),
    ImpellerType.HYDROFOIL_A320: (0.55, 0.75),
    ImpellerType.ANCHOR: (0.3, 0.5),
    ImpellerType.HELICAL_RIBBON: (0.3, 0.6),
}

# Flow (pumping) numbers, same conditions.
REFERENCE_FLOW_NUMBER: dict[ImpellerType, float] = {
    ImpellerType.RUSHTON_6: 0.78,
    ImpellerType.PBT_4_45: 0.79,
    ImpellerType.PBT_6_45: 0.81,
    ImpellerType.HYDROFOIL_A310: 0.56,
    ImpellerType.HYDROFOIL_A320: 0.65,
}


def reference_power_number(impeller: ImpellerType) -> float | None:
    """Midpoint of the literature range, for use as a surrogate prior mean."""
    span = REFERENCE_POWER_NUMBER.get(impeller)
    return None if span is None else 0.5 * (span[0] + span[1])


def grenville_blend_time(
    speed_hz: float,
    power_number: float,
    impeller_diameter_m: float,
    tank_diameter_m: float,
) -> float:
    """95% blend time (s) from the Grenville correlation.

    Valid for turbulent (Re > ~5000), fully baffled tanks with liquid level ~= T.
    Outside that envelope this is a rough prior only, not an acceptance criterion.
    """
    if speed_hz <= 0 or power_number <= 0 or impeller_diameter_m <= 0:
        raise ValueError("speed, power number and impeller diameter must be positive")
    ratio = tank_diameter_m / impeller_diameter_m
    return 5.2 / (speed_hz * power_number ** (1.0 / 3.0)) * ratio**2


def reynolds_number(
    speed_hz: float, impeller_diameter_m: float, density: float, viscosity: float
) -> float:
    return density * speed_hz * impeller_diameter_m**2 / viscosity


def froude_number(speed_hz: float, impeller_diameter_m: float, gravity: float = 9.81) -> float:
    return speed_hz**2 * impeller_diameter_m / gravity


def tip_speed(speed_hz: float, impeller_diameter_m: float) -> float:
    import math

    return math.pi * speed_hz * impeller_diameter_m
