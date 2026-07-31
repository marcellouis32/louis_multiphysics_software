"""The case specification: the single contract shared by the solver, the report
generator, the ML pipeline and the AI setup agent.

Design constraints worth preserving as this grows:
  - Every field is human-readable and human-editable. The agent fills this schema;
    it never emits solver code.
  - Physics inputs are separated from numerics inputs so the design space can be
    sampled cleanly for surrogate training.
  - Dimensionless groups are first-class, because scale-up generalisation only
    works in Pi-space.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lms.schema.reference import (
    ImpellerType,
    froude_number,
    reynolds_number,
    tip_speed,
)

SCHEMA_VERSION = "0.1.0"

Positive = Annotated[float, Field(gt=0)]
NonNegative = Annotated[float, Field(ge=0)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


# --------------------------------------------------------------------------- geometry


class BottomType(str, Enum):
    FLAT = "flat"
    TORISPHERICAL = "torispherical"
    HEMISPHERICAL = "hemispherical"
    CONICAL = "conical"


class Vessel(Strict):
    diameter_m: Positive
    liquid_level_m: Positive
    bottom: BottomType = BottomType.FLAT
    stl_path: str | None = None


class Baffles(Strict):
    count: int = Field(default=4, ge=0, le=12)
    width_ratio: Positive = Field(default=0.1, description="baffle width / tank diameter")
    wall_offset_ratio: NonNegative = Field(default=0.02)
    thickness_m: Positive = 0.005


class Impeller(Strict):
    type: ImpellerType
    diameter_m: Positive
    clearance_m: Positive = Field(description="height of impeller centreline above vessel bottom")
    speed_rpm: Positive
    blade_width_m: Positive | None = None
    n_blades: int | None = Field(default=None, ge=2, le=12)
    pumping: Literal["down", "up"] = "down"
    stl_path: str | None = None

    @property
    def speed_hz(self) -> float:
        return self.speed_rpm / 60.0

    @model_validator(mode="after")
    def _custom_needs_stl(self) -> Impeller:
        if self.type is ImpellerType.CUSTOM_STL and not self.stl_path:
            raise ValueError("impeller type 'custom_stl' requires stl_path")
        return self


class Geometry(Strict):
    vessel: Vessel
    impellers: list[Impeller] = Field(min_length=1)
    baffles: Baffles = Baffles()

    @model_validator(mode="after")
    def _check_fits(self) -> Geometry:
        t = self.vessel.diameter_m
        for imp in self.impellers:
            if imp.diameter_m >= t:
                raise ValueError(
                    f"impeller diameter {imp.diameter_m} m exceeds vessel diameter {t} m"
                )
            if imp.clearance_m >= self.vessel.liquid_level_m:
                raise ValueError("impeller clearance is above the liquid level")
        return self


# --------------------------------------------------------------------------- fluid


class Fluid(Strict):
    model: Literal["newtonian", "power_law", "carreau"] = "newtonian"
    density_kg_m3: Positive
    viscosity_pa_s: Positive | None = Field(default=None, description="newtonian only")
    consistency_k: Positive | None = Field(default=None, description="power law / carreau")
    flow_index_n: Positive | None = None

    @model_validator(mode="after")
    def _required_params(self) -> Fluid:
        if self.model == "newtonian" and self.viscosity_pa_s is None:
            raise ValueError("newtonian fluid requires viscosity_pa_s")
        if self.model != "newtonian" and (self.consistency_k is None or self.flow_index_n is None):
            raise ValueError(f"{self.model} fluid requires consistency_k and flow_index_n")
        return self

    def apparent_viscosity(self, shear_rate: float) -> float:
        if self.model == "newtonian":
            return float(self.viscosity_pa_s)
        if self.model == "power_law":
            return float(self.consistency_k) * shear_rate ** (float(self.flow_index_n) - 1.0)
        raise NotImplementedError("carreau apparent viscosity not implemented yet")


# --------------------------------------------------------------------------- physics


class Turbulence(Strict):
    model: Literal["none", "les_smagorinsky"] = "les_smagorinsky"
    smagorinsky_constant: Positive = 0.1


class Momentum(Strict):
    engine: Literal["lbm"] = "lbm"
    collision: Literal["bgk", "mrt", "cumulant"] = "mrt"
    turbulence: Turbulence = Turbulence()


class ScalarField(Strict):
    name: str
    schmidt_number: Positive = 1000.0
    initial_condition: Literal["uniform", "top_patch", "bottom_patch"] = "top_patch"
    initial_value: float = 1.0


class Energy(Strict):
    enabled: bool = False
    prandtl_number: Positive = 7.0
    jacket_temperature_c: float | None = None
    initial_temperature_c: float = 20.0


class Physics(Strict):
    momentum: Momentum = Momentum()
    scalars: list[ScalarField] = Field(default_factory=list)
    energy: Energy = Energy()
    gravity_m_s2: Positive = 9.81


# --------------------------------------------------------------------------- numerics


class Numerics(Strict):
    cells_across_tank: int = Field(default=128, ge=32, le=1024)
    lattice_mach: Positive = Field(
        default=0.05, le=0.2, description="lattice velocity scaling; >0.1 breaks incompressibility"
    )
    duration_s: Positive
    precision: Literal["fp32", "fp64"] = "fp32"

    def dx(self, tank_diameter_m: float) -> float:
        return tank_diameter_m / self.cells_across_tank


# --------------------------------------------------------------------------- outputs


class FieldOutput(Strict):
    enabled: bool = True
    interval_s: Positive = 0.5
    variables: list[str] = Field(default_factory=lambda: ["velocity", "pressure"])
    canonical_resolution: int = Field(
        default=64,
        ge=16,
        description="fields are resampled to this cubic grid so runs at different mesh "
        "densities stay comparable as ML training tensors",
    )


class Outputs(Strict):
    quantities: list[str] = Field(
        default_factory=lambda: [
            "power_number",
            "power_per_volume",
            "blend_time_95",
            "tip_speed",
            "energy_dissipation_max",
            "shear_stress_max",
        ]
    )
    fields: FieldOutput = FieldOutput()


class RunConfig(Strict):
    device: str = "cuda:0"
    seed: int = 0


# --------------------------------------------------------------------------- top level


class DimensionlessGroups(Strict):
    """Scale-invariant feature vector. This is what surrogates learn in."""

    reynolds: float
    froude: float
    diameter_ratio: float
    clearance_ratio: float
    aspect_ratio: float
    baffle_width_ratio: float
    n_baffles: int
    impeller: ImpellerType

    def feature_vector(self) -> dict[str, float]:
        return {
            "log10_reynolds": math.log10(max(self.reynolds, 1e-12)),
            "log10_froude": math.log10(max(self.froude, 1e-12)),
            "diameter_ratio": self.diameter_ratio,
            "clearance_ratio": self.clearance_ratio,
            "aspect_ratio": self.aspect_ratio,
            "baffle_width_ratio": self.baffle_width_ratio,
            "n_baffles": float(self.n_baffles),
        }


class Case(Strict):
    schema_version: str = SCHEMA_VERSION
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)

    geometry: Geometry
    fluid: Fluid
    physics: Physics = Physics()
    numerics: Numerics
    outputs: Outputs = Outputs()
    run: RunConfig = RunConfig()

    @property
    def primary_impeller(self) -> Impeller:
        return self.geometry.impellers[0]

    @property
    def liquid_volume_m3(self) -> float:
        r = self.geometry.vessel.diameter_m / 2.0
        return math.pi * r**2 * self.geometry.vessel.liquid_level_m

    def dimensionless(self) -> DimensionlessGroups:
        imp = self.primary_impeller
        vessel = self.geometry.vessel
        mu = (
            self.fluid.viscosity_pa_s
            if self.fluid.model == "newtonian"
            # Metzner-Otto: shear rate ~ 11*N for shear-thinning fluids in stirred tanks
            else self.fluid.apparent_viscosity(11.0 * imp.speed_hz)
        )
        return DimensionlessGroups(
            reynolds=reynolds_number(
                imp.speed_hz, imp.diameter_m, self.fluid.density_kg_m3, float(mu)
            ),
            froude=froude_number(imp.speed_hz, imp.diameter_m, self.physics.gravity_m_s2),
            diameter_ratio=imp.diameter_m / vessel.diameter_m,
            clearance_ratio=imp.clearance_m / vessel.diameter_m,
            aspect_ratio=vessel.liquid_level_m / vessel.diameter_m,
            baffle_width_ratio=self.geometry.baffles.width_ratio,
            n_baffles=self.geometry.baffles.count,
            impeller=imp.type,
        )

    def tip_speed_m_s(self) -> float:
        imp = self.primary_impeller
        return tip_speed(imp.speed_hz, imp.diameter_m)

    def fingerprint(self) -> str:
        """Deterministic hash of physics + numerics. Becomes the run ID, so the
        results store is keyed by what was actually simulated."""
        payload = self.model_dump(mode="json", exclude={"name", "description", "tags", "run"})
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @model_validator(mode="after")
    def _physics_sanity(self) -> Case:
        groups = self.dimensionless()
        if groups.reynolds < 10.0:
            raise ValueError(
                f"impeller Reynolds number is {groups.reynolds:.2g}; below ~10 the flow is "
                "creeping and the LES closure is meaningless. Check viscosity and speed."
            )
        # Resolving the impeller needs cells across the blade, not just across the tank.
        cells_across_impeller = self.numerics.cells_across_tank * groups.diameter_ratio
        if cells_across_impeller < 20:
            raise ValueError(
                f"only {cells_across_impeller:.0f} cells span the impeller "
                f"(D/T = {groups.diameter_ratio:.2f}). Raise numerics.cells_across_tank to at "
                f"least {math.ceil(20 / groups.diameter_ratio)} to resolve the blades."
            )
        if self.geometry.baffles.count == 0 and groups.froude > 0.1:
            raise ValueError(
                f"unbaffled tank at Froude {groups.froude:.2f} will form a surface vortex, "
                "which this solver does not model (no free surface). Add baffles or reduce speed."
            )
        return self


def load_case(path: str | Path) -> Case:
    with open(path) as fh:
        return Case.model_validate(yaml.safe_load(fh))


def dump_case(case: Case, path: str | Path) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(case.model_dump(mode="json"), fh, sort_keys=False, default_flow_style=False)
