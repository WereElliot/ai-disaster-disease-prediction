import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "Dashboard" / "futureguard_dashboard"
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "futureguard_dashboard.settings")

import django  # noqa: E402

django.setup()

from django.test import Client  # noqa: E402

from core.services import PredictionService  # noqa: E402


def test_action_plan_api_supports_full_prediction_payload_with_climate_region():
    client = Client(HTTP_HOST="localhost")
    service = PredictionService()
    predictions = service.predict_all("nakuru")

    response = client.post(
        "/api/action-plan/",
        data=json.dumps(
            {
                "county": predictions["climate_region"],
                "predictions": predictions,
                "language": "en",
            }
        ),
        content_type="application/json",
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["climate_region"] == predictions["climate_region"]
    assert len(payload["hazard_sections"]) == 5
    assert {section["category"] for section in payload["hazard_sections"]} == {
        "disease",
        "disaster",
    }
    assert all(section["actions"] for section in payload["hazard_sections"])
