# Predictapp/urls.py

from django.urls import path
from . import views

app_name = 'predictapp'

urlpatterns = [
    # Main Dashboard - this will be at /dashboard/
    path('', views.dashboard_view, name='dashboard'),
    
    # Live Predictions
    path('predictions/', views.predictions, name='predictions'),
    path('run-prediction/', views.run_prediction, name='run_prediction'),
    
    # Risk Map
    path('risk-map/', views.risk_map, name='risk_map'),
    
    # Data Management
    path('data-ingestion/', views.data_ingestion, name='data_ingestion'),
    path('climate-data/', views.climate_data, name='climate_data'),
    path('health-records/', views.health_records, name='health_records'),
    path('disaster-events/', views.disaster_events, name='disaster_events'),
    
    # Models
    path('models/lstm/', views.lstm_model, name='lstm_model'),
    path('models/xgboost/', views.xgboost_model, name='xgboost_model'),
    path('models/hybrid/', views.hybrid_model, name='hybrid_model'),
    
    # XAI / Explainability
    path('xai/', views.xai_explainability, name='xai'),
    
    # Reports
    path('evaluation/', views.evaluation, name='evaluation'),
    path('export/', views.export_report, name='export'),
    
    # API endpoints for dashboard (AJAX calls)
    path('api/dashboard-stats/', views.dashboard_stats, name='dashboard_stats'),
    path('api/recent-alerts/', views.recent_alerts, name='recent_alerts'),
    path('api/model-performance/', views.model_performance, name='model_performance'),
]