from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

# Try to import models, but continue gracefully outside a configured Django app.
try:
    from .models import ClimateData, DiseaseIncidence, Prediction
except Exception:
    ClimateData = None
    DiseaseIncidence = None
    Prediction = None


class PredictionService:
    """Deterministic, climate-aware risk scoring for Kenya regions."""

    COUNTY_PROFILES = {
        "nairobi": {
            "county_label": "Nairobi",
            "climate_region": "Central Highlands",
            "coordinates": (-1.2866, 36.8172),
            "defaults": {
                "temperature": 22.6,
                "humidity": 68.0,
                "precipitation": 7.5,
                "wind_speed": 4.1,
            },
            "disease_bias": {
                "malaria": 0.46,
                "cholera": 0.34,
                "dengue": 0.29,
                "floods": 0.58,
                "drought": 0.22,
            },
        },
        "mombasa": {
            "county_label": "Mombasa",
            "climate_region": "Coastal Lowlands",
            "coordinates": (-4.0435, 39.6682),
            "defaults": {
                "temperature": 28.1,
                "humidity": 79.0,
                "precipitation": 9.8,
                "wind_speed": 5.6,
            },
            "disease_bias": {
                "malaria": 0.44,
                "cholera": 0.76,
                "dengue": 0.69,
                "floods": 0.57,
                "drought": 0.18,
            },
        },
        "kisumu": {
            "county_label": "Kisumu",
            "climate_region": "Lake Basin",
            "coordinates": (-0.1057, 34.7617),
            "defaults": {
                "temperature": 25.8,
                "humidity": 73.0,
                "precipitation": 10.5,
                "wind_speed": 4.7,
            },
            "disease_bias": {
                "malaria": 0.73,
                "cholera": 0.42,
                "dengue": 0.47,
                "floods": 0.68,
                "drought": 0.22,
            },
        },
        "nakuru": {
            "county_label": "Nakuru",
            "climate_region": "Rift Valley",
            "coordinates": (-0.2833, 36.0667),
            "defaults": {
                "temperature": 21.8,
                "humidity": 64.0,
                "precipitation": 6.0,
                "wind_speed": 4.4,
            },
            "disease_bias": {
                "malaria": 0.39,
                "cholera": 0.27,
                "dengue": 0.25,
                "floods": 0.43,
                "drought": 0.48,
            },
        },
        "kiambu": {
            "county_label": "Kiambu",
            "climate_region": "Central Highlands",
            "coordinates": (-1.1748, 36.8356),
            "defaults": {
                "temperature": 21.3,
                "humidity": 70.0,
                "precipitation": 8.0,
                "wind_speed": 4.0,
            },
            "disease_bias": {
                "malaria": 0.43,
                "cholera": 0.26,
                "dengue": 0.22,
                "floods": 0.40,
                "drought": 0.24,
            },
        },
        "machakos": {
            "county_label": "Machakos",
            "climate_region": "Eastern Transitional",
            "coordinates": (-1.5177, 37.2634),
            "defaults": {
                "temperature": 24.1,
                "humidity": 60.0,
                "precipitation": 4.1,
                "wind_speed": 4.6,
            },
            "disease_bias": {
                "malaria": 0.34,
                "cholera": 0.28,
                "dengue": 0.19,
                "floods": 0.25,
                "drought": 0.62,
            },
        },
        "eldoret": {
            "county_label": "Eldoret",
            "climate_region": "Western Highlands",
            "coordinates": (0.5143, 35.2698),
            "defaults": {
                "temperature": 20.6,
                "humidity": 69.0,
                "precipitation": 6.2,
                "wind_speed": 4.3,
            },
            "disease_bias": {
                "malaria": 0.31,
                "cholera": 0.24,
                "dengue": 0.20,
                "floods": 0.37,
                "drought": 0.36,
            },
        },
        "garissa": {
            "county_label": "Garissa",
            "climate_region": "Arid and Semi-Arid North-East",
            "coordinates": (-0.4569, 39.6583),
            "defaults": {
                "temperature": 30.2,
                "humidity": 49.0,
                "precipitation": 2.3,
                "wind_speed": 5.1,
            },
            "disease_bias": {
                "malaria": 0.17,
                "cholera": 0.24,
                "dengue": 0.12,
                "floods": 0.14,
                "drought": 0.82,
            },
        },
    }

    LOCATION_ALIASES = {
        "nairobi, ke": "nairobi",
        "mombasa, ke": "mombasa",
        "kisumu, ke": "kisumu",
        "nakuru, ke": "nakuru",
        "kiambu, ke": "kiambu",
        "machakos, ke": "machakos",
        "eldoret, ke": "eldoret",
        "garissa, ke": "garissa",
        "central highlands": "nairobi",
        "coastal lowlands": "mombasa",
        "lake basin": "kisumu",
        "rift valley": "nakuru",
        "eastern transitional": "machakos",
        "western highlands": "eldoret",
        "arid and semi-arid north-east": "garissa",
        "arid and semi arid north east": "garissa",
        "arid semi arid north east": "garissa",
        "north east asals": "garissa",
        "north-east asals": "garissa",
    }

    DISEASE_ALIASES = {
        "malaria": "malaria",
        "cholera": "cholera",
        "dengue": "dengue",
        "dengue fever": "dengue",
        "flood": "floods",
        "floods": "floods",
        "drought": "drought",
    }

    DISEASE_LABELS = {
        "malaria": "Malaria",
        "cholera": "Cholera",
        "dengue": "Dengue Fever",
        "floods": "Flood",
        "drought": "Drought",
    }

    def get_available_locations(self) -> list[dict[str, Any]]:
        locations = []
        for key, profile in self.COUNTY_PROFILES.items():
            locations.append(
                {
                    "value": key,
                    "county": profile["county_label"],
                    "label": f"{profile['climate_region']} ({profile['county_label']})",
                    "climate_region": profile["climate_region"],
                    "coordinates": profile["coordinates"],
                }
            )
        return locations

    def get_available_climate_regions(self) -> list[dict[str, Any]]:
        regions = []
        seen: set[str] = set()
        for key, profile in self.COUNTY_PROFILES.items():
            region = profile["climate_region"]
            normalized_region = region.lower()
            if normalized_region in seen:
                continue
            seen.add(normalized_region)
            regions.append(
                {
                    "value": region,
                    "label": region,
                    "county": profile["county_label"],
                    "canonical_location": key,
                }
            )
        return regions

    @staticmethod
    def _clean_location(location: str | None) -> str:
        cleaned = (location or "").strip().lower().replace("_", " ")
        cleaned = cleaned.replace("county", "").strip()
        return " ".join(cleaned.split())

    def resolve_location(self, location: str | None) -> str | None:
        cleaned = self._clean_location(location)
        if not cleaned:
            return None
        if cleaned in self.LOCATION_ALIASES:
            return self.LOCATION_ALIASES[cleaned]
        if cleaned in self.COUNTY_PROFILES:
            return cleaned
        for key, profile in self.COUNTY_PROFILES.items():
            if cleaned == profile["county_label"].lower():
                return key
        return None

    def normalize_location(self, location: str | None) -> str:
        return self.resolve_location(location) or "nairobi"

    def normalize_disease(self, disease_type: str | None) -> str:
        cleaned = (disease_type or "").strip().lower()
        cleaned = " ".join(cleaned.split())
        return self.DISEASE_ALIASES.get(cleaned, "malaria")

    def get_profile(self, county: str | None) -> dict[str, Any]:
        return self.COUNTY_PROFILES[self.normalize_location(county)]

    @staticmethod
    def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return float(max(lower, min(upper, value)))

    def _band_score(self, value: float, low: float, high: float, tolerance: float) -> float:
        if low <= value <= high:
            return 1.0
        if value < low:
            return self._clamp(1.0 - ((low - value) / tolerance))
        return self._clamp(1.0 - ((value - high) / tolerance))

    def _upper_score(self, value: float, threshold: float, tolerance: float) -> float:
        if value >= threshold:
            return 1.0
        return self._clamp(1.0 - ((threshold - value) / tolerance))

    def _lower_score(self, value: float, threshold: float, tolerance: float) -> float:
        if value <= threshold:
            return 1.0
        return self._clamp(1.0 - ((value - threshold) / tolerance))

    def _climate_signal(
        self,
        disease_type: str,
        temperature: float,
        humidity: float,
        precipitation: float,
        wind_speed: float,
    ) -> float:
        if disease_type == "malaria":
            return (
                0.40 * self._band_score(temperature, 22.0, 30.0, 8.0)
                + 0.30 * self._band_score(humidity, 60.0, 85.0, 25.0)
                + 0.30 * self._band_score(precipitation, 4.0, 18.0, 16.0)
            )
        if disease_type == "cholera":
            return (
                0.20 * self._band_score(temperature, 24.0, 32.0, 10.0)
                + 0.30 * self._upper_score(humidity, 65.0, 25.0)
                + 0.50 * self._upper_score(precipitation, 7.0, 14.0)
            )
        if disease_type == "dengue":
            return (
                0.35 * self._band_score(temperature, 24.0, 32.0, 10.0)
                + 0.35 * self._upper_score(humidity, 65.0, 25.0)
                + 0.30 * self._band_score(precipitation, 4.0, 16.0, 16.0)
            )
        if disease_type == "floods":
            return (
                0.55 * self._upper_score(precipitation, 8.0, 16.0)
                + 0.20 * self._upper_score(humidity, 65.0, 25.0)
                + 0.15 * self._upper_score(wind_speed, 4.5, 3.0)
                + 0.10 * self._band_score(temperature, 20.0, 28.0, 10.0)
            )
        return (
            0.50 * self._lower_score(precipitation, 5.0, 10.0)
            + 0.25 * self._upper_score(temperature, 26.0, 10.0)
            + 0.25 * self._lower_score(humidity, 60.0, 25.0)
        )

    def _weighted_score(
        self,
        disease_type: str,
        bias: float,
        climate_signal: float,
        history_signal: float,
        hazard_signal: float,
    ) -> float:
        if disease_type == "malaria":
            weights = (0.35, 0.35, 0.20, 0.10)
        elif disease_type == "cholera":
            weights = (0.25, 0.25, 0.25, 0.25)
        elif disease_type == "dengue":
            weights = (0.30, 0.35, 0.25, 0.10)
        elif disease_type == "floods":
            weights = (0.20, 0.35, 0.10, 0.35)
        else:
            weights = (0.25, 0.35, 0.10, 0.30)

        return (
            (weights[0] * bias)
            + (weights[1] * climate_signal)
            + (weights[2] * history_signal)
            + (weights[3] * hazard_signal)
        )

    def _risk_level(self, score: float) -> str:
        if score >= 0.80:
            return "critical"
        if score >= 0.65:
            return "high"
        if score >= 0.45:
            return "medium"
        return "low"

    def _confidence(
        self,
        score: float,
        bias: float,
        climate_signal: float,
        history_signal: float,
        hazard_signal: float,
        horizon_days: int,
    ) -> float:
        signals = np.array([bias, climate_signal, history_signal, hazard_signal], dtype=float)
        consistency = 1.0 - float(np.std(signals))
        raw_confidence = 58.0 + (score * 30.0) + (consistency * 8.0) - min(8.0, horizon_days / 20.0)
        return round(self._clamp(raw_confidence, 55.0, 97.0), 1)

    def predict_county(
        self,
        county: str,
        disease_type: str = "malaria",
        context: dict[str, Any] | None = None,
        horizon_days: int = 30,
        persist: bool = False,
    ) -> dict[str, Any]:
        """Generate a deterministic prediction for a county or Kenya climate region."""

        canonical_county = self.normalize_location(county)
        canonical_disease = self.normalize_disease(disease_type)
        profile = self.COUNTY_PROFILES[canonical_county]
        defaults = profile["defaults"]
        context = context or {}

        temperature = float(context.get("temperature", defaults["temperature"]))
        humidity = float(context.get("humidity", defaults["humidity"]))
        precipitation = float(context.get("precipitation", defaults["precipitation"]))
        wind_speed = float(context.get("wind_speed", defaults["wind_speed"]))
        history_signal = self._clamp(float(context.get("history_signal", profile["disease_bias"][canonical_disease])))
        hazard_signal = self._clamp(float(context.get("hazard_signal", history_signal)))
        bias = self._clamp(float(profile["disease_bias"][canonical_disease]))

        climate_signal = self._climate_signal(
            canonical_disease,
            temperature,
            humidity,
            precipitation,
            wind_speed,
        )
        raw_score = self._clamp(
            self._weighted_score(canonical_disease, bias, climate_signal, history_signal, hazard_signal)
        )
        risk_level = self._risk_level(raw_score)
        confidence = self._confidence(
            raw_score,
            bias,
            climate_signal,
            history_signal,
            hazard_signal,
            horizon_days,
        )

        component_scores = {
            "regional baseline": round(bias, 3),
            "climate suitability": round(climate_signal, 3),
            "historical burden": round(history_signal, 3),
            "hazard pressure": round(hazard_signal, 3),
        }
        drivers = [
            name
            for name, _ in sorted(component_scores.items(), key=lambda item: item[1], reverse=True)[:3]
        ]

        prediction_id = None
        if persist and Prediction is not None:
            try:
                prediction = Prediction.objects.create(
                    location=canonical_county,
                    disease_type=canonical_disease,
                    risk_level=risk_level,
                    confidence=confidence,
                    model_used="hybrid",
                    forecast_horizon_days=horizon_days,
                )
                prediction_id = prediction.id
            except Exception:
                prediction_id = None

        return {
            "risk": risk_level,
            "confidence": confidence,
            "raw_score": round(raw_score, 3),
            "prediction_id": prediction_id,
            "county": profile["county_label"],
            "climate_region": profile["climate_region"],
            "disease_type": canonical_disease,
            "disease_label": self.DISEASE_LABELS[canonical_disease],
            "coordinates": profile["coordinates"],
            "drivers": drivers,
            "component_scores": component_scores,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def predict_all(
        self,
        county: str,
        context_map: dict[str, dict[str, Any]] | None = None,
        horizon_days: int = 30,
        persist: bool = False,
    ) -> dict[str, Any]:
        """Get deterministic predictions for all supported diseases."""

        canonical_county = self.normalize_location(county)
        profile = self.COUNTY_PROFILES[canonical_county]
        diseases = ["malaria", "cholera", "dengue", "floods", "drought"]
        results: dict[str, Any] = {}

        for disease in diseases:
            disease_context = (context_map or {}).get(disease, {})
            results[disease] = self.predict_county(
                canonical_county,
                disease,
                context=disease_context,
                horizon_days=horizon_days,
                persist=persist,
            )

        results["timestamp"] = datetime.now().isoformat(timespec="seconds")
        results["county"] = profile["county_label"]
        results["climate_region"] = profile["climate_region"]
        return results
