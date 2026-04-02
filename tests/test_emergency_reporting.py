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

from core.models import EmergencyReport  # noqa: E402


TEST_MARKER = "[TEST_EMERGENCY_REPORT]"


def _cleanup_reports() -> None:
    EmergencyReport.objects.filter(description__contains=TEST_MARKER).delete()


def test_public_emergency_report_submission_creates_admin_visible_report():
    _cleanup_reports()
    client = Client(HTTP_HOST="localhost")

    try:
        response = client.post(
            "/emergency-report/",
            {
                "reporter_name": "Test Reporter",
                "reporter_phone": "0700000000",
                "county": "Nairobi",
                "location_detail": "Kibra Estate",
                "emergency_type": "fire",
                "severity": "critical",
                "people_affected": "12",
                "description": f"{TEST_MARKER} Fire outbreak near market stalls.",
            },
        )

        assert response.status_code == 302

        report = EmergencyReport.objects.get(description__contains=TEST_MARKER)
        assert report.county == "Nairobi"
        assert report.emergency_type == "fire"
        assert report.severity == "critical"
        assert report.status == "new"

        admin_response = client.get("/admin-dashboard/?page=emergencies")
        page = admin_response.content.decode("utf-8")

        assert admin_response.status_code == 200
        assert "Community Emergencies" in page
        assert "Kibra Estate" in page
        assert "Fire" in page
    finally:
        _cleanup_reports()


def test_admin_can_update_emergency_report_status():
    _cleanup_reports()
    client = Client(HTTP_HOST="localhost")

    try:
        report = EmergencyReport.objects.create(
            reporter_name="Admin Workflow Test",
            county="Mombasa",
            location_detail="Likoni Ferry",
            emergency_type="flood",
            severity="high",
            description=f"{TEST_MARKER} Flooding on approach road.",
            people_affected=25,
        )

        response = client.post(
            f"/admin-dashboard/emergency-reports/{report.id}/status/",
            {
                "status": "acknowledged",
                "return_page": "emergencies",
            },
        )

        assert response.status_code == 302

        report.refresh_from_db()
        assert report.status == "acknowledged"
        assert report.reviewed_at is not None
    finally:
        _cleanup_reports()
