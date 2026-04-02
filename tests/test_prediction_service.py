import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Dashboard" / "futureguard_dashboard"))

from core.services import PredictionService  # noqa: E402


def test_prediction_service_normalizes_locations_and_diseases():
    service = PredictionService()

    assert service.normalize_location("Mombasa, KE") == "mombasa"
    assert service.normalize_location("Garissa County") == "garissa"
    assert service.normalize_location("Rift Valley") == "nakuru"
    assert service.resolve_location("Central Highlands") == "nairobi"
    assert service.normalize_disease("Dengue Fever") == "dengue"
    assert service.normalize_disease("Flood") == "floods"


def test_prediction_service_returns_deterministic_region_metadata():
    service = PredictionService()

    result = service.predict_county("garissa", "drought")

    assert result["risk"] == "critical"
    assert result["confidence"] >= 85
    assert result["climate_region"] == "Arid and Semi-Arid North-East"
    assert result["county"] == "Garissa"
    assert len(result["drivers"]) == 3
