
from django.db import models
from django.utils import timezone

class ClimateData(models.Model):
    """Climate data from NASA POWER and other sources"""
    
    location = models.CharField(max_length=100)
    date = models.DateField()
    temperature = models.FloatField(help_text="Temperature in Celsius")
    precipitation = models.FloatField(help_text="Precipitation in mm")
    humidity = models.FloatField(help_text="Relative humidity in %")
    wind_speed = models.FloatField(help_text="Wind speed in m/s")
    source = models.CharField(max_length=50, default='NASA POWER')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['location', 'date']
    
    def __str__(self):
        return f"{self.location} - {self.date}"


class DiseaseIncidence(models.Model):
    """Disease incidence data from WHO and local ministries"""
    
    DISEASE_CHOICES = [
        ('malaria', 'Malaria'),
        ('cholera', 'Cholera'),
        ('dengue', 'Dengue'),
        ('floods', 'Floods'),
        ('drought', 'Drought'),
    ]
    
    location = models.CharField(max_length=100)
    date = models.DateField()
    disease = models.CharField(max_length=50, choices=DISEASE_CHOICES)
    cases = models.PositiveIntegerField(default=0)
    deaths = models.PositiveIntegerField(default=0)
    source = models.CharField(max_length=50, default='WHO GHO')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['location', 'date', 'disease']
    
    def __str__(self):
        return f"{self.location} - {self.disease} - {self.date}"


class Prediction(models.Model):
    """Model predictions stored in database"""
    
    RISK_LEVELS = [
        ('critical', 'Critical'),
        ('high', 'High'),
        ('medium', 'Medium'),
        ('low', 'Low'),
    ]
    
    MODEL_CHOICES = [
        ('lstm', 'LSTM'),
        ('xgboost', 'XGBoost'),
        ('hybrid', 'Hybrid Ensemble'),
    ]
    
    location = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    disease_type = models.CharField(max_length=50, choices=DiseaseIncidence.DISEASE_CHOICES)
    risk_level = models.CharField(max_length=20, choices=RISK_LEVELS)
    confidence = models.FloatField(help_text="Confidence percentage")
    model_used = models.CharField(max_length=20, choices=MODEL_CHOICES, default='hybrid')
    forecast_horizon_days = models.PositiveIntegerField(default=30)
    
    def __str__(self):
        return f"{self.location} - {self.disease_type} - {self.risk_level} ({self.created_at.date()})"


class EmergencyReport(models.Model):
    """Emergency reports submitted by community users."""

    EMERGENCY_TYPES = [
        ("flood", "Flood"),
        ("fire", "Fire"),
        ("medical", "Medical Emergency"),
        ("disease", "Disease Outbreak"),
        ("drought", "Drought"),
        ("landslide", "Landslide"),
        ("accident", "Road Accident"),
        ("violence", "Security Incident"),
        ("infrastructure", "Infrastructure Failure"),
        ("other", "Other"),
    ]

    SEVERITY_LEVELS = [
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High"),
        ("critical", "Critical"),
    ]

    STATUS_CHOICES = [
        ("new", "New"),
        ("acknowledged", "Acknowledged"),
        ("in_progress", "In Progress"),
        ("resolved", "Resolved"),
    ]

    reporter_name = models.CharField(max_length=120, default="Anonymous Community Member")
    reporter_phone = models.CharField(max_length=30, blank=True)
    county = models.CharField(max_length=100)
    location_detail = models.CharField(max_length=180, blank=True)
    emergency_type = models.CharField(max_length=30, choices=EMERGENCY_TYPES)
    severity = models.CharField(max_length=20, choices=SEVERITY_LEVELS, default="medium")
    description = models.TextField()
    people_affected = models.PositiveIntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="new")
    admin_notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.county} - {self.get_emergency_type_display()} - {self.get_severity_display()}"

    @property
    def severity_css(self) -> str:
        return "moderate" if self.severity == "medium" else self.severity

    @property
    def status_badge_css(self) -> str:
        return {
            "new": "red",
            "acknowledged": "amber",
            "in_progress": "blue",
            "resolved": "green",
        }.get(self.status, "blue")

    @property
    def reporter_display(self) -> str:
        return self.reporter_name or "Anonymous Community Member"

    @property
    def created_at_display(self) -> str:
        local_time = timezone.localtime(self.created_at)
        return local_time.strftime("%Y-%m-%d %H:%M")
