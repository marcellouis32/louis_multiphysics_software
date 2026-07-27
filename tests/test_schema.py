import copy

import pytest
import yaml
from pydantic import ValidationError

from lms.schema.case import Case, load_case
from lms.schema.reference import (
    REFERENCE_POWER_NUMBER,
    ImpellerType,
    grenville_blend_time,
)

EXAMPLE = "cases/examples/rushton_standard.yaml"


@pytest.fixture
def raw() -> dict:
    with open(EXAMPLE) as fh:
        return yaml.safe_load(fh)


def test_example_loads():
    case = load_case(EXAMPLE)
    assert case.name == "rushton_standard_lab"
    assert case.primary_impeller.type is ImpellerType.RUSHTON_6


def test_standard_geometry_ratios():
    groups = load_case(EXAMPLE).dimensionless()
    assert groups.diameter_ratio == pytest.approx(1 / 3, abs=1e-3)
    assert groups.clearance_ratio == pytest.approx(1 / 3, abs=1e-3)
    assert groups.aspect_ratio == pytest.approx(1.0, abs=1e-3)
    assert groups.reynolds == pytest.approx(4.99e4, rel=0.02)


def test_tip_speed_and_volume():
    case = load_case(EXAMPLE)
    assert case.tip_speed_m_s() == pytest.approx(1.571, rel=1e-3)
    assert case.liquid_volume_m3 == pytest.approx(0.0212, rel=1e-2)


def test_fingerprint_ignores_cosmetics(raw):
    a = Case.model_validate(raw)
    renamed = copy.deepcopy(raw)
    renamed["name"] = "something_else"
    renamed["description"] = "different text"
    assert Case.model_validate(renamed).fingerprint() == a.fingerprint()

    perturbed = copy.deepcopy(raw)
    perturbed["geometry"]["impellers"][0]["speed_rpm"] = 301
    assert Case.model_validate(perturbed).fingerprint() != a.fingerprint()


def test_feature_vector_is_scale_invariant(raw):
    """A 10x larger vessel at matched Re and geometry must produce identical features
    except Froude. This is the property the scale-up surrogate depends on."""
    big = copy.deepcopy(raw)
    big["geometry"]["vessel"]["diameter_m"] = 3.0
    big["geometry"]["vessel"]["liquid_level_m"] = 3.0
    big["geometry"]["impellers"][0]["diameter_m"] = 1.0
    big["geometry"]["impellers"][0]["clearance_m"] = 1.0
    big["geometry"]["impellers"][0]["speed_rpm"] = 3.0  # N D^2 held constant

    small_f = Case.model_validate(raw).dimensionless().feature_vector()
    big_f = Case.model_validate(big).dimensionless().feature_vector()
    assert small_f["log10_reynolds"] == pytest.approx(big_f["log10_reynolds"], abs=1e-6)
    assert small_f["diameter_ratio"] == pytest.approx(big_f["diameter_ratio"], abs=1e-9)
    assert small_f["aspect_ratio"] == pytest.approx(big_f["aspect_ratio"], abs=1e-9)


def test_rejects_underresolved_impeller(raw):
    raw["numerics"]["cells_across_tank"] = 32
    with pytest.raises(ValidationError, match="cells span the impeller"):
        Case.model_validate(raw)


def test_rejects_oversized_impeller(raw):
    raw["geometry"]["impellers"][0]["diameter_m"] = 0.4
    with pytest.raises(ValidationError, match="exceeds vessel diameter"):
        Case.model_validate(raw)


def test_rejects_unbaffled_vortexing(raw):
    raw["geometry"]["baffles"]["count"] = 0
    with pytest.raises(ValidationError, match="surface vortex"):
        Case.model_validate(raw)


def test_rejects_creeping_flow(raw):
    raw["fluid"]["viscosity_pa_s"] = 100.0
    with pytest.raises(ValidationError, match="creeping"):
        Case.model_validate(raw)


def test_rejects_unknown_field(raw):
    raw["geometry"]["vessel"]["diamter_m"] = 0.3
    with pytest.raises(ValidationError):
        Case.model_validate(raw)


def test_grenville_prior_is_plausible():
    """Sanity-check the correlation used as the surrogate prior: a 0.3 m standard
    Rushton tank at 5 rev/s should blend in a few seconds."""
    lo, hi = REFERENCE_POWER_NUMBER[ImpellerType.RUSHTON_6]
    theta = grenville_blend_time(5.0, 0.5 * (lo + hi), 0.10, 0.30)
    assert 4.0 < theta < 8.0
