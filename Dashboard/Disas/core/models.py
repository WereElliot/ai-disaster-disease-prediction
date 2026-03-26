
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