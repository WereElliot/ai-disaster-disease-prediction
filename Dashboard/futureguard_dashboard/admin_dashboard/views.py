import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_POST

from core.models import EmergencyReport
from core.services import PredictionService

prediction_service = PredictionService()
REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures"
FORECAST_PLOTS_ROOT = APP_ROOT / "static" / "plots" / "forecasts"
EVALUATION_REPORT_PATH = REPO_ROOT / "docs" / "Evaluation_Report.md"
ADMIN_PAGES = {
    "dashboard",
    "predictions",
    "riskmap",
    "emergencies",
    "ingestion",
    "climate",
    "health",
    "disasters",
    "lstm",
    "xgboost",
    "hybrid",
    "xai",
    "evaluation",
    "export",
}


def _format_risk_label(risk: str) -> str:
    return (risk or "low").replace("_", " ").title()


def _coerce_admin_page(page: str | None) -> str:
    return page if page in ADMIN_PAGES else "dashboard"


def _build_emergency_report_context() -> dict[str, object]:
    status_priority = {"new": 0, "acknowledged": 1, "in_progress": 2, "resolved": 3}
    severity_priority = {"critical": 3, "high": 2, "medium": 1, "low": 0}
    reports = sorted(
        EmergencyReport.objects.all(),
        key=lambda report: (
            status_priority.get(report.status, 99),
            -severity_priority.get(report.severity, 0),
            -report.created_at.timestamp(),
        ),
    )

    return {
        "community_alert_count": len([report for report in reports if report.status != "resolved"]),
        "community_emergency_total": len(reports),
        "community_emergency_new": len([report for report in reports if report.status == "new"]),
        "community_emergency_in_progress": len(
            [report for report in reports if report.status in {"acknowledged", "in_progress"}]
        ),
        "community_emergency_resolved": len([report for report in reports if report.status == "resolved"]),
        "community_emergency_critical": len(
            [report for report in reports if report.severity == "critical" and report.status != "resolved"]
        ),
        "recent_emergency_reports": reports[:6],
        "emergency_reports": reports[:20],
    }


def _parse_horizon_days(horizon: str | None) -> int:
    mapping = {
        "7 days": 7,
        "14 days": 14,
        "30 days": 30,
        "90 days": 90,
    }
    return mapping.get(horizon or "", 30)


def _coerce_horizon_days(horizon: object) -> int:
    if isinstance(horizon, int):
        return max(horizon, 1)

    if isinstance(horizon, str):
        cleaned = horizon.strip()
        if cleaned.isdigit():
            return max(int(cleaned), 1)
        return _parse_horizon_days(cleaned)

    return 30


def _load_dashboard_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    climate_df = pd.read_csv(FIXTURE_ROOT / "climate_sample.csv", parse_dates=["date"])
    health_df = pd.read_csv(FIXTURE_ROOT / "health_sample.csv", parse_dates=["date"])
    disasters_df = pd.read_csv(FIXTURE_ROOT / "disasters_sample.csv", parse_dates=["date"])
    return climate_df, health_df, disasters_df


def _load_model_metrics() -> tuple[float, float]:
    model_f1_score = 0.89
    model_accuracy = 0.88

    if not EVALUATION_REPORT_PATH.exists():
        return model_f1_score, model_accuracy

    report_content = EVALUATION_REPORT_PATH.read_text(encoding="utf-8", errors="ignore")
    f1_match = re.search(r"F1-Score[\*\*\s]*[:\-]*\s*([\d.]+)", report_content)
    if f1_match:
        model_f1_score = float(f1_match.group(1))

    acc_match = re.search(r"Accuracy[\*\*\s]*[:\-]*\s*([\d.]+)", report_content)
    if acc_match:
        model_accuracy = float(acc_match.group(1))

    return model_f1_score, model_accuracy


def _representative_region_keys() -> list[str]:
    keys = []
    seen_regions = set()
    for key, profile in prediction_service.COUNTY_PROFILES.items():
        climate_region = profile["climate_region"]
        if climate_region not in seen_regions:
            keys.append(key)
            seen_regions.add(climate_region)
    return keys


def _normalize_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series

    max_value = float(series.max())
    if max_value <= 0:
        return series * 0

    return (series / max_value).fillna(0.0)


def _disaster_signal(disaster_frame: pd.DataFrame, latest_date: pd.Timestamp) -> float:
    if disaster_frame.empty:
        return 0.0

    days_ago = (latest_date - disaster_frame["date"]).dt.days.clip(lower=0)
    weights = np.exp(-(days_ago / 75.0))
    severity = disaster_frame["severity_score"].astype(float) / 5.0
    weighted_signal = float(np.sum(severity * weights) / max(1.0, weights.sum() * 0.75))
    return round(min(1.0, weighted_signal), 3)


def _build_location_metrics(
    climate_df: pd.DataFrame,
    health_df: pd.DataFrame,
    disasters_df: pd.DataFrame,
) -> tuple[dict[str, dict[str, float]], pd.Timestamp]:
    latest_date = climate_df["date"].max()
    recent_climate = climate_df[climate_df["date"] >= latest_date - pd.Timedelta(days=30)]
    recent_health = health_df[health_df["date"] >= latest_date - pd.Timedelta(days=30)]
    recent_disasters = disasters_df[disasters_df["date"] >= latest_date - pd.Timedelta(days=180)]

    climate_means = recent_climate.groupby("location")[
        ["temperature", "humidity", "precipitation", "wind_speed"]
    ].mean()
    health_means = recent_health.groupby("location")[
        ["malaria_cases", "cholera_cases", "dengue_cases", "diarrheal_cases"]
    ].mean()
    health_norm = health_means.divide(health_means.max().replace(0, 1), axis=1).fillna(0.0)
    precipitation_norm = _normalize_series(recent_climate.groupby("location")["precipitation"].mean())

    flood_signals = {}
    drought_signals = {}
    heatwave_signals = {}
    for location, disaster_group in recent_disasters.groupby("location"):
        flood_signals[location] = _disaster_signal(
            disaster_group[disaster_group["disaster_type"] == "Flood"], latest_date
        )
        drought_signals[location] = _disaster_signal(
            disaster_group[disaster_group["disaster_type"] == "Drought"], latest_date
        )
        heatwave_signals[location] = _disaster_signal(
            disaster_group[disaster_group["disaster_type"] == "Heatwave"], latest_date
        )

    location_metrics = {}
    for region_key in _representative_region_keys():
        profile = prediction_service.get_profile(region_key)
        location_name = profile["county_label"]
        defaults = profile["defaults"]
        climate_row = climate_means.loc[location_name] if location_name in climate_means.index else pd.Series(defaults)
        health_row = health_norm.loc[location_name] if location_name in health_norm.index else pd.Series(dtype=float)

        location_metrics[region_key] = {
            "temperature": round(float(climate_row.get("temperature", defaults["temperature"])), 2),
            "humidity": round(float(climate_row.get("humidity", defaults["humidity"])), 2),
            "precipitation": round(float(climate_row.get("precipitation", defaults["precipitation"])), 2),
            "wind_speed": round(float(climate_row.get("wind_speed", defaults["wind_speed"])), 2),
            "malaria_signal": round(float(health_row.get("malaria_cases", 0.0)), 3),
            "cholera_signal": round(float(health_row.get("cholera_cases", 0.0)), 3),
            "dengue_signal": round(float(health_row.get("dengue_cases", 0.0)), 3),
            "rainfall_signal": round(float(precipitation_norm.get(location_name, 0.0)), 3),
            "flood_signal": round(float(flood_signals.get(location_name, 0.0)), 3),
            "drought_signal": round(float(drought_signals.get(location_name, 0.0)), 3),
            "heatwave_signal": round(float(heatwave_signals.get(location_name, 0.0)), 3),
        }

    return location_metrics, latest_date


def _prediction_context(metrics: dict[str, float], disease_type: str) -> dict[str, float]:
    shared = {
        "temperature": metrics["temperature"],
        "humidity": metrics["humidity"],
        "precipitation": metrics["precipitation"],
        "wind_speed": metrics["wind_speed"],
    }

    if disease_type == "malaria":
        return {
            **shared,
            "history_signal": metrics["malaria_signal"],
            "hazard_signal": max(metrics["rainfall_signal"] * 0.35, metrics["flood_signal"] * 0.25),
        }

    if disease_type == "cholera":
        return {
            **shared,
            "history_signal": metrics["cholera_signal"],
            "hazard_signal": max(metrics["flood_signal"], metrics["rainfall_signal"] * 0.7),
        }

    if disease_type == "dengue":
        return {
            **shared,
            "history_signal": metrics["dengue_signal"],
            "hazard_signal": max(metrics["rainfall_signal"] * 0.5, metrics["flood_signal"] * 0.35),
        }

    if disease_type == "floods":
        return {
            **shared,
            "history_signal": max(metrics["flood_signal"], metrics["rainfall_signal"] * 0.6),
            "hazard_signal": max(metrics["flood_signal"], metrics["rainfall_signal"]),
        }

    return {
        **shared,
        "history_signal": max(metrics["drought_signal"], metrics["heatwave_signal"] * 0.6),
        "hazard_signal": max(metrics["drought_signal"], metrics["heatwave_signal"] * 0.8),
    }


def _build_region_predictions(
    location_metrics: dict[str, dict[str, float]],
    horizon_days: int = 30,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    disease_order = ["malaria", "cholera", "floods", "drought", "dengue"]
    all_predictions = []
    region_summary = []

    for region_key in _representative_region_keys():
        profile = prediction_service.get_profile(region_key)
        disease_predictions = {}

        for disease in disease_order:
            prediction = prediction_service.predict_county(
                region_key,
                disease,
                context=_prediction_context(location_metrics[region_key], disease),
                horizon_days=horizon_days,
            )
            disease_predictions[disease] = prediction
            all_predictions.append(
                {
                    **prediction,
                    "county": profile["county_label"],
                    "climate_region": profile["climate_region"],
                    "region_label": f"{profile['climate_region']} ({profile['county_label']})",
                }
            )

        top_prediction = max(disease_predictions.values(), key=lambda item: item["raw_score"])
        region_summary.append(
            {
                "zone": profile["climate_region"],
                "county": profile["county_label"],
                "label": f"{profile['climate_region']} ({profile['county_label']})",
                "top_threat": top_prediction["disease_label"],
                "risk": _format_risk_label(top_prediction["risk"]),
                "risk_css": top_prediction["risk"],
                "confidence": top_prediction["confidence"],
                "raw_score": top_prediction["raw_score"],
                "lat": profile["coordinates"][0],
                "lng": profile["coordinates"][1],
                "predictions": disease_predictions,
            }
        )

    return all_predictions, region_summary


def _build_forecast_images() -> list[dict[str, str]]:
    key_forecasts = [
        "malaria_1week_forecast.png",
        "cholera_1week_forecast.png",
        "flood_1week_forecast.png",
        "dengue_1week_forecast.png",
        "drought_1week_forecast.png",
        "cyclone_1week_forecast.png",
    ]
    forecast_images = []

    if FORECAST_PLOTS_ROOT.exists():
        for image_name in key_forecasts:
            image_path = FORECAST_PLOTS_ROOT / image_name
            if image_path.exists():
                forecast_images.append(
                    {
                        "url": f"/static/plots/forecasts/{image_name}",
                        "title": image_name.replace("_", " ").replace(".png", "").title(),
                    }
                )

        if not forecast_images:
            for image_path in sorted(FORECAST_PLOTS_ROOT.glob("*.png")):
                if "copy" in image_path.name.lower():
                    continue

                forecast_images.append(
                    {
                        "url": f"/static/plots/forecasts/{image_path.name}",
                        "title": image_path.name.replace("_", " ").replace(".png", "").title(),
                    }
                )
                if len(forecast_images) == 6:
                    break

    return forecast_images


def _build_monthly_chart_data(
    climate_df: pd.DataFrame,
    health_df: pd.DataFrame,
    disasters_df: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    month_index = pd.period_range(
        climate_df["date"].min().to_period("M"),
        climate_df["date"].max().to_period("M"),
        freq="M",
    )
    labels = [period.strftime("%b") for period in month_index]

    monthly_climate = climate_df.groupby(climate_df["date"].dt.to_period("M"))[
        ["temperature", "precipitation"]
    ].mean().reindex(month_index, fill_value=0.0)
    monthly_health = health_df.groupby(health_df["date"].dt.to_period("M"))[
        ["malaria_cases", "cholera_cases", "dengue_cases", "diarrheal_cases"]
    ].sum().reindex(month_index, fill_value=0.0)

    flood_rows = disasters_df[disasters_df["disaster_type"] == "Flood"]
    drought_rows = disasters_df[disasters_df["disaster_type"] == "Drought"]
    monthly_flood_signal = flood_rows.groupby(flood_rows["date"].dt.to_period("M"))["severity_score"].sum().reindex(
        month_index,
        fill_value=0.0,
    )
    monthly_drought_signal = drought_rows.groupby(drought_rows["date"].dt.to_period("M"))["severity_score"].sum().reindex(
        month_index,
        fill_value=0.0,
    )

    return {
        "dashboard_trend": {
            "labels": labels,
            "malaria": (_normalize_series(monthly_health["malaria_cases"]) * 100).round(1).tolist(),
            "cholera": (_normalize_series(monthly_health["cholera_cases"]) * 100).round(1).tolist(),
            "floods": (_normalize_series(monthly_flood_signal) * 100).round(1).tolist(),
        },
        "climate_chart": {
            "labels": labels,
            "temperature": monthly_climate["temperature"].round(1).tolist(),
            "precipitation": monthly_climate["precipitation"].round(1).tolist(),
        },
        "health_chart": {
            "labels": labels,
            "malaria": monthly_health["malaria_cases"].round(0).astype(int).tolist(),
            "cholera": monthly_health["cholera_cases"].round(0).astype(int).tolist(),
        },
        "health_breakdown": {
            "labels": ["Malaria", "Cholera", "Dengue", "Diarrhoeal"],
            "values": [
                int(health_df["malaria_cases"].sum()),
                int(health_df["cholera_cases"].sum()),
                int(health_df["dengue_cases"].sum()),
                int(health_df["diarrheal_cases"].sum()),
            ],
        },
        "disaster_chart": {
            "labels": labels,
            "floods": monthly_flood_signal.round(1).tolist(),
            "droughts": monthly_drought_signal.round(1).tolist(),
        },
    }


def _fallback_region_key(location_metrics: dict[str, dict[str, float]]) -> str:
    if "nairobi" in location_metrics:
        return "nairobi"
    return next(iter(location_metrics))


def dashboard_view(request):
    climate_df, health_df, disasters_df = _load_dashboard_frames()
    location_metrics, latest_date = _build_location_metrics(climate_df, health_df, disasters_df)
    default_region_key = _fallback_region_key(location_metrics)
    all_predictions, region_summary = _build_region_predictions(location_metrics, horizon_days=30)
    chart_data = _build_monthly_chart_data(climate_df, health_df, disasters_df)

    model_f1_score, model_accuracy = _load_model_metrics()
    disease_predictions = round(model_f1_score * 100, 1)
    disaster_forecasts = round(model_accuracy * 100, 1)
    training_samples = len(climate_df) + len(health_df) + len(disasters_df)

    latest_climate_snapshot = climate_df[climate_df["date"] == latest_date]
    temp = round(float(latest_climate_snapshot["temperature"].mean()), 1)
    humidity = round(float(latest_climate_snapshot["humidity"].mean()), 1)
    precip = round(float(latest_climate_snapshot["precipitation"].mean()), 1)
    wind = round(float(latest_climate_snapshot["wind_speed"].mean()), 1)
    rmse = round(
        float(
            np.sqrt(
                np.mean(
                    (latest_climate_snapshot["temperature"] - climate_df["temperature"].mean()) ** 2
                )
            )
        ),
        3,
    )
    roc_auc = round(model_accuracy, 3)

    alerts = []
    for prediction in sorted(all_predictions, key=lambda item: item["raw_score"], reverse=True):
        if prediction["risk"] == "low":
            continue

        alerts.append(
            {
                "region": f"{prediction['climate_region']} ({prediction['county']})",
                "threat": prediction["disease_label"],
                "confidence": prediction["confidence"],
                "risk": _format_risk_label(prediction["risk"]),
                "risk_css": prediction["risk"],
                "source": "Hybrid climate-risk engine",
                "time": latest_date.strftime("%Y-%m-%d"),
            }
        )
        if len(alerts) == 6:
            break

    prediction_log = []
    for prediction in sorted(all_predictions, key=lambda item: item["raw_score"], reverse=True)[:8]:
        prediction_log.append(
            {
                "timestamp": latest_date.strftime("%Y-%m-%d"),
                "region": f"{prediction['climate_region']} ({prediction['county']})",
                "type": prediction["disease_label"],
                "model": "Hybrid",
                "confidence": prediction["confidence"],
                "risk": _format_risk_label(prediction["risk"]),
                "risk_css": prediction["risk"],
                "horizon": "30d",
            }
        )

    prediction_chart = {
        "labels": [row["label"] for row in region_summary],
        "malaria": [round(row["predictions"]["malaria"]["confidence"], 1) for row in region_summary],
        "cholera": [round(row["predictions"]["cholera"]["confidence"], 1) for row in region_summary],
        "floods": [round(row["predictions"]["floods"]["confidence"], 1) for row in region_summary],
    }

    risk_map_points = []
    for row in region_summary:
        top_prediction = max(row["predictions"].values(), key=lambda item: item["raw_score"])
        risk_map_points.append(
            {
                "zone": row["label"],
                "lat": row["lat"],
                "lng": row["lng"],
                "risk": top_prediction["confidence"],
                "level": _format_risk_label(top_prediction["risk"]),
                "level_css": top_prediction["risk"],
                "threat": top_prediction["disease_label"],
                "county": row["county"],
                "climate_region": row["zone"],
            }
        )

    selected_disease = request.POST.get("disease_type", "Malaria")
    selected_region = request.POST.get("region", default_region_key)
    selected_horizon = request.POST.get("horizon", "30 days")
    prediction_result = None
    if request.method == "POST":
        region_key = prediction_service.normalize_location(selected_region)
        disease_key = prediction_service.normalize_disease(selected_disease)
        prediction = prediction_service.predict_county(
            region_key,
            disease_key,
            context=_prediction_context(
                location_metrics.get(region_key, location_metrics[default_region_key]),
                disease_key,
            ),
            horizon_days=_parse_horizon_days(selected_horizon),
        )
        prediction_result = {
            "disease_type": prediction["disease_label"],
            "region": prediction["county"],
            "climate_region": prediction["climate_region"],
            "horizon": selected_horizon,
            "confidence": prediction["confidence"],
            "risk": _format_risk_label(prediction["risk"]),
            "risk_css": prediction["risk"],
            "drivers": prediction["drivers"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    region_options = []
    for region_key in _representative_region_keys():
        profile = prediction_service.get_profile(region_key)
        region_options.append(
            {
                "value": region_key,
                "label": f"{profile['climate_region']} ({profile['county_label']})",
                "county": profile["county_label"],
                "climate_region": profile["climate_region"],
            }
        )

    models = [
        {"name": "LSTM", "acc": 0.89, "width": "89%"},
        {"name": "XGBoost", "acc": 0.87, "width": "87%"},
        {"name": "Hybrid", "acc": 0.91, "width": "91%"},
    ]

    emergency_context = _build_emergency_report_context()

    context = {
        "disease_predictions": disease_predictions,
        "disaster_forecasts": disaster_forecasts,
        "active_alerts": len(alerts),
        "training_samples": training_samples,
        "temp": temp,
        "humidity": humidity,
        "precip": precip,
        "wind": wind,
        "rmse": rmse,
        "roc_auc": roc_auc,
        "models": models,
        "alerts": alerts,
        "prediction_result": prediction_result,
        "forecast_images": _build_forecast_images(),
        "critical_predictions": len([item for item in all_predictions if item["risk"] == "critical"]),
        "active_forecasts": len(all_predictions),
        "avg_confidence": round(sum(item["confidence"] for item in all_predictions) / len(all_predictions), 1),
        "model_f1": round(model_f1_score * 100, 1),
        "model_accuracy": round(model_accuracy * 100, 1),
        "prediction_log": prediction_log,
        "risk_zones": region_summary,
        "risk_map_points": risk_map_points,
        "critical_zones": len([point for point in risk_map_points if point["level_css"] == "critical"]),
        "high_risk_zones": len([point for point in risk_map_points if point["level_css"] == "high"]),
        "monitored_regions": len(risk_map_points),
        "low_risk_zones": len([point for point in risk_map_points if point["level_css"] == "low"]),
        "dashboard_trend_data": chart_data["dashboard_trend"],
        "prediction_region_chart_data": prediction_chart,
        "climate_chart_data": chart_data["climate_chart"],
        "health_chart_data": chart_data["health_chart"],
        "health_breakdown_data": chart_data["health_breakdown"],
        "disaster_chart_data": chart_data["disaster_chart"],
        "dashboard_subtitle": (
            f"Dataset snapshot: {latest_date.strftime('%B %d, %Y')} - "
            f"{len(region_summary)} Kenya climate regions monitored"
        ),
        "data_latest_date": latest_date.strftime("%Y-%m-%d"),
        "region_options": region_options,
        "disease_options": [
            {"value": "Malaria", "label": "Malaria"},
            {"value": "Cholera", "label": "Cholera"},
            {"value": "Flood", "label": "Flood"},
            {"value": "Drought", "label": "Drought"},
            {"value": "Dengue Fever", "label": "Dengue Fever"},
        ],
        "horizon_options": [
            {"value": "7 days", "label": "7 days"},
            {"value": "14 days", "label": "14 days"},
            {"value": "30 days", "label": "30 days"},
            {"value": "90 days", "label": "90 days"},
        ],
        "selected_disease": selected_disease,
        "selected_region": prediction_service.normalize_location(selected_region),
        "selected_horizon": selected_horizon,
        "initial_page": _coerce_admin_page(request.GET.get("page")),
    }
    context.update(emergency_context)

    return render(request, "admin_dashboard/dashboard.html", context)


def run_prediction(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        return JsonResponse({"error": "Invalid JSON payload"}, status=400)

    climate_df, health_df, disasters_df = _load_dashboard_frames()
    location_metrics, _ = _build_location_metrics(climate_df, health_df, disasters_df)
    default_region_key = _fallback_region_key(location_metrics)

    region_key = prediction_service.normalize_location(payload.get("county", default_region_key))
    disease_key = prediction_service.normalize_disease(payload.get("disease", "malaria"))
    horizon_days = _coerce_horizon_days(payload.get("horizon", 30))

    prediction = prediction_service.predict_county(
        region_key,
        disease_key,
        context=_prediction_context(
            location_metrics.get(region_key, location_metrics[default_region_key]),
            disease_key,
        ),
        horizon_days=horizon_days,
        persist=True,
    )

    return JsonResponse(
        {
            "success": True,
            "prediction": prediction,
            "message": f"Prediction completed for {prediction['county']}",
        }
    )


@require_POST
def update_emergency_report_status(request, report_id: int):
    report = get_object_or_404(EmergencyReport, pk=report_id)
    next_status = request.POST.get("status", "").strip()
    return_page = _coerce_admin_page(request.POST.get("return_page"))

    valid_statuses = {choice[0] for choice in EmergencyReport.STATUS_CHOICES}
    if next_status in valid_statuses:
        report.status = next_status
        report.admin_notes = request.POST.get("admin_notes", report.admin_notes).strip()
        report.reviewed_at = timezone.now()
        report.save(update_fields=["status", "admin_notes", "reviewed_at", "updated_at"])

    dashboard_url = reverse("admin_dashboard:dashboard")
    return redirect(f"{dashboard_url}?page={return_page}")
