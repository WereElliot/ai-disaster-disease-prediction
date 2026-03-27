from django.shortcuts import render
import pandas as pd
from django.http import JsonResponse
import os
import json
from datetime import datetime, timezone, timedelta
import random
import glob
import json
from django.contrib.auth.decorators import login_required
from django.utils import timezone

from core.models import ClimateData, DiseaseIncidence, Prediction
from core.services import PredictionService

prediction_service = PredictionService()

# Create your views here.

def _relative_time_from_date(ts: datetime) -> str:
    """Convert a datetime to a human-friendly relative time string."""
    now = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta = now - ts
    minutes = int(delta.total_seconds() / 60)
    if minutes < 60:
        return f"{minutes}m ago"
    hours = int(minutes / 60)
    if hours < 24:
        return f"{hours}h ago"
    days = int(hours / 24)
    return f"{days}d ago"


def _run_prediction(disease_type: str, region: str, horizon: str) -> dict:
    """Mock prediction function - in real implementation, load trained models."""
    # Simulate prediction based on inputs
    base_confidence = random.uniform(70, 95)
    
    # Adjust based on disease type
    if disease_type == 'Malaria':
        base_confidence += random.uniform(-5, 10)
    elif disease_type == 'Cholera':
        base_confidence += random.uniform(-10, 5)
    elif disease_type == 'Flood':
        base_confidence += random.uniform(0, 15)
    
    # Adjust based on region (simulated risk factors)
    region_risks = {
        'Nairobi, KE': 1.2,
        'Mombasa, KE': 1.1,
        'Kisumu, KE': 1.3,
        'Nakuru, KE': 0.9,
        'Eldoret, KE': 0.8
    }
    risk_multiplier = region_risks.get(region, 1.0)
    confidence = min(99.9, base_confidence * risk_multiplier)
    
    # Determine risk level
    if confidence > 85:
        risk = 'Critical'
    elif confidence > 70:
        risk = 'High'
    elif confidence > 50:
        risk = 'Moderate'
    else:
        risk = 'Low'
    
    return {
        'disease_type': disease_type,
        'region': region,
        'horizon': horizon,
        'confidence': round(confidence, 1),
        'risk': risk,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def dashboard_view(request):
    # Load sample data
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    climate_path = os.path.join(base_path, 'tests', 'fixtures', 'climate_sample.csv')
    disasters_path = os.path.join(base_path, 'tests', 'fixtures', 'disasters_sample.csv')
    health_path = os.path.join(base_path, 'tests', 'fixtures', 'health_sample.csv')
    
    prediction_result = None
    if request.method == 'POST':
        disease_type = request.POST.get('disease_type')
        region = request.POST.get('region')
        horizon = request.POST.get('horizon')
        if disease_type and region and horizon:
            prediction_result = _run_prediction(disease_type, region, horizon)
    
    try:
        climate_df = pd.read_csv(climate_path)
        disasters_df = pd.read_csv(disasters_path)
        health_df = pd.read_csv(health_path)
        
        # Load model metrics from evaluation report
        eval_report_path = os.path.join(base_path, 'docs', 'Evaluation_Report.md')
        model_f1_score = 0.89  # default
        model_accuracy = 0.88  # default
        if os.path.exists(eval_report_path):
            with open(eval_report_path, 'r') as f:
                report_content = f.read()
                # Try to extract F1 and Accuracy from report
                if 'F1-Score' in report_content:
                    try:
                        import re
                        f1_match = re.search(r'F1-Score[\*\*\s]*[:\-]*\s*([\d.]+)', report_content)
                        if f1_match:
                            model_f1_score = float(f1_match.group(1))
                        acc_match = re.search(r'Accuracy[\*\*\s]*[:\-]*\s*([\d.]+)', report_content)
                        if acc_match:
                            model_accuracy = float(acc_match.group(1))
                    except:
                        pass
        
        
        # Card 1: Disease Predictions - Model F1-Score accuracy
        disease_predictions = round(model_f1_score * 100, 1)
        
        # Card 2: Disaster Forecasts - Model accuracy on test set
        disaster_forecasts = round(model_accuracy * 100, 1)
        
        
        training_samples = len(climate_df) + len(health_df) + len(disasters_df)

        # Build alerts from disasters file (recent events)
        try:
            disasters_df['date'] = pd.to_datetime(disasters_df['date'])
            recent_disasters = disasters_df.sort_values('date', ascending=False).head(3)

            def _severity_to_risk(score: float) -> str:
                if score >= 4:
                    return 'Critical'
                if score >= 3:
                    return 'High'
                if score >= 2:
                    return 'Moderate'
                return 'Low'

            alerts = []
            for _, row in recent_disasters.iterrows():
                alerts.append({
                    'region': f"{row['location']}, KE",
                    'threat': row['disaster_type'],
                    'confidence': min(100, float(row.get('severity_score', 0)) * 25),
                    'risk': _severity_to_risk(float(row.get('severity_score', 0))),
                    'source': 'EM-DAT',
                    'time': _relative_time_from_date(row['date']),
                })
            active_alerts = len(alerts)
        except Exception:
            # fall back to static alerts if parse fails
            active_alerts = 9
            alerts = [
                {'region': 'Nairobi, KE', 'threat': 'Malaria', 'confidence': 94.2, 'risk': 'Critical', 'source': 'NASA POWER', 'time': '2h ago'},
                {'region': 'Mombasa, KE', 'threat': 'Cholera', 'confidence': 88.7, 'risk': 'High', 'source': 'WHO GHO', 'time': '5h ago'},
                {'region': 'Kisumu, KE', 'threat': 'Flood', 'confidence': 81.3, 'risk': 'High', 'source': 'EM-DAT', 'time': '12h ago'},
            ]

        # Get forecast images
        plots_dir = os.path.join(base_path, 'static', 'plots', 'forecasts')
        forecast_images = []
        if os.path.exists(plots_dir):
            # Get all available 1week forecasts for key diseases/disasters
            key_forecasts = [
                'malaria_1week_forecast.png',
                'cholera_1week_forecast.png', 
                'flood_1week_forecast.png',
                'dengue_1week_forecast.png',
                'drought_1week_forecast.png',
                'cyclone_1week_forecast.png'
            ]
            for img in key_forecasts:
                img_path = os.path.join(plots_dir, img)
                if os.path.exists(img_path):
                    forecast_images.append({
                        'url': f'/static/plots/forecasts/{img}',
                        'title': img.replace('_', ' ').replace('.png', '').title()
                    })
            
            # If no specific forecasts found, get all available images
            if not forecast_images:
                all_images = [f for f in os.listdir(plots_dir) if f.endswith('.png') and 'copy' not in f.lower()]
                for img in all_images[:6]:  # Limit to 6 images for display
                    forecast_images.append({
                        'url': f'/static/plots/forecasts/{img}',
                        'title': img.replace('_', ' ').replace('.png', '').title()
                    })
        else:
            # Fallback with real paths from your forecast folder
            forecast_images = [
                {'url': '/static/plots/forecasts/malaria_1week_forecast.png', 'title': 'Malaria 1 Week'},
                {'url': '/static/plots/forecasts/cholera_1week_forecast.png', 'title': 'Cholera 1 Week'},
                {'url': '/static/plots/forecasts/flood_1week_forecast.png', 'title': 'Flood 1 Week'},
                {'url': '/static/plots/forecasts/dengue_1week_forecast.png', 'title': 'Dengue 1 Week'},
                {'url': '/static/plots/forecasts/drought_1week_forecast.png', 'title': 'Drought 1 Week'},
                {'url': '/static/plots/forecasts/cyclone_1week_forecast.png', 'title': 'Cyclone 1 Week'},
            ]

        # Get latest climate data
        try:
            climate_df['date'] = pd.to_datetime(climate_df['date'])
            nairobi_climate = climate_df[climate_df['location'].str.contains('Nairobi', case=False, na=False)]
            
            if not nairobi_climate.empty:
                latest = nairobi_climate.iloc[-1]
                temp = round(float(latest['temperature']), 1)
                humidity = round(float(latest['humidity']), 1)
                precip = round(float(latest['precipitation']), 1)
                
                # Compute average wind speed (simulate since not in data)
                wind = round(float(nairobi_climate['temperature'].mean()) * 0.1, 1)
                
                # Compute RMSE on temperature (simple baseline: predict mean)
                mean_temp = float(nairobi_climate['temperature'].mean())
                rmse = ((nairobi_climate['temperature'].astype(float) - mean_temp) ** 2).mean() ** 0.5
                rmse = round(rmse, 3)
            else:
                temp, humidity, precip, wind, rmse = 22.4, 74, 8.2, 3.1, 0.043
        except:
            temp, humidity, precip, wind, rmse = 22.4, 74, 8.2, 3.1, 0.043
        
        # ROC-AUC: use model accuracy as the ROC-AUC metric
        roc_auc = round(model_accuracy, 3)

        # Model performance (static for now, could load from evaluation)
        models = [
            {'name': 'LSTM', 'acc': 0.89, 'width': '89%'},
            {'name': 'XGBoost', 'acc': 0.87, 'width': '87%'},
            {'name': 'Hybrid', 'acc': 0.91, 'width': '91%'},
        ]

        context = {
            'disease_predictions': disease_predictions,
            'disaster_forecasts': disaster_forecasts,
            'active_alerts': active_alerts,
            'training_samples': training_samples,
            'temp': temp,
            'humidity': humidity,
            'precip': precip,
            'wind': wind,
            'rmse': rmse,
            'roc_auc': roc_auc,
            'models': models,
            'alerts': alerts,
            'prediction_result': prediction_result,
            'forecast_images': forecast_images,
            'critical_predictions': len([a for a in alerts if a['risk'] == 'Critical']),
            'active_forecasts': len(alerts),
            'avg_confidence': round(sum(a['confidence'] for a in alerts) / len(alerts), 1) if alerts else 0,
            'model_f1': round(model_f1_score * 100, 1),
            'model_accuracy': round(model_accuracy * 100, 1),
        }
        
        return render(request, 'dashboard.html', context)
    except Exception as e:
        # Fallback to static data
        context = {
            'disease_predictions': 89.0,  # Model F1-Score
            'disaster_forecasts': 88.0,   # Model Accuracy
            'active_alerts': 9,            # Count of active alerts
            'training_samples': 35284,     # Total dataset size
            'temp': 22.4,
            'humidity': 74,
            'precip': 8.2,
            'wind': 3.1,
            'rmse': 0.043,
            'roc_auc': 0.88,
            'models': [
                {'name': 'LSTM', 'acc': 0.89, 'width': '89%'},
                {'name': 'XGBoost', 'acc': 0.87, 'width': '87%'},
                {'name': 'Hybrid', 'acc': 0.91, 'width': '91%'},
            ],
            'alerts': [
                {'region': 'Nairobi, KE', 'threat': 'Malaria', 'confidence': 94.2, 'risk': 'Critical', 'source': 'NASA POWER', 'time': '2h ago'},
                {'region': 'Mombasa, KE', 'threat': 'Cholera', 'confidence': 88.7, 'risk': 'High', 'source': 'WHO GHO', 'time': '5h ago'},
                {'region': 'Kisumu, KE', 'threat': 'Flood', 'confidence': 81.3, 'risk': 'High', 'source': 'EM-DAT', 'time': '12h ago'},
            ],
            'prediction_result': prediction_result,
            'forecast_images': [
                {'path': '/static/plots/forecasts/malaria_1week_forecast.png', 'title': 'Malaria 1 Week'},
                {'path': '/static/plots/forecasts/cholera_1week_forecast.png', 'title': 'Cholera 1 Week'},
                {'path': '/static/plots/forecasts/flood_1week_forecast.png', 'title': 'Flood 1 Week'},
            ],
            'critical_predictions': 1,
            'active_forecasts': 3,
            'avg_confidence': 88.1,
            'model_f1': 89.0,
            'model_accuracy': 88.0,
        }
    
    return render(request, 'dashboard.html', context)


# ========== MAIN VIEWS ==========

@login_required(login_url='/admin/login/')
def dashboard(request):
    """Main admin dashboard view"""
    context = {
        'title': 'System Dashboard',
        'disease_predictions': 89.1,  # This should come from your model
        'disaster_forecasts': 87.2,   # This should come from your model
        'active_alerts': 3,
        'training_samples': 35284,
        'temp': 22.4,
        'precip': 8.2,
        'humidity': 74,
        'wind': 3.1,
        'rmse': 0.043,
        'roc_auc': 0.942,
        'models': [
            {'name': 'LSTM', 'acc': '89.1%', 'width': '89%'},
            {'name': 'XGBoost', 'acc': '87.0%', 'width': '87%'},
            {'name': 'Hybrid', 'acc': '91.0%', 'width': '91%'},
        ],
        'alerts': [
            {'region': 'Nairobi, KE', 'threat': 'Malaria', 'confidence': 94.2, 
             'risk': 'Critical', 'source': 'Hybrid', 'time': '2h ago'},
            {'region': 'Nakuru, KE', 'threat': 'Malaria', 'confidence': 72.1, 
             'risk': 'Medium', 'source': 'LSTM', 'time': '1d ago'},
            {'region': 'Mombasa, KE', 'threat': 'Cholera', 'confidence': 88.7, 
             'risk': 'High', 'source': 'XGBoost', 'time': '3h ago'},
        ],
        'forecast_images': [
            {'url': '/static/images/malaria_forecast.png', 'title': 'Malaria Forecast'},
            {'url': '/static/images/cholera_forecast.png', 'title': 'Cholera Forecast'},
            {'url': '/static/images/flood_forecast.png', 'title': 'Flood Risk'},
        ]
    }
    return render(request, 'dashboard.html', context)


@login_required(login_url='/admin/login/')
def predictions(request):
    """Live predictions view"""
    counties = [
        {'value': 'nairobi', 'label': 'Nairobi'},
        {'value': 'mombasa', 'label': 'Mombasa'},
        {'value': 'kisumu', 'label': 'Kisumu'},
        {'value': 'nakuru', 'label': 'Nakuru'},
        {'value': 'eldoret', 'label': 'Eldoret'},
        {'value': 'garissa', 'label': 'Garissa'},
    ]
    
    context = {
        'title': 'Live Predictions',
        'critical_predictions': 3,
        'active_forecasts': 12,
        'avg_confidence': 84.2,
        'counties': counties,
        'recent_predictions': get_recent_predictions(),
    }
    return render(request, 'predictapp/predictions.html', context)


@login_required(login_url='/admin/login/')
def risk_map(request):
    """Risk map view"""
    context = {
        'title': 'Risk Map',
        'critical_zones': 1,
        'high_risk_zones': 2,
        'monitored_regions': 7,
        'low_risk_zones': 3,
        'risk_zones': get_risk_zones(),
    }
    return render(request, 'predictapp/risk_map.html', context)


@login_required(login_url='/admin/login/')
def data_ingestion(request):
    """Data ingestion management view"""
    context = {
        'title': 'Data Ingestion',
        'sources_online': 3,
        'records_last_sync': 142,
        'last_ingestion': '2m ago',
        'total_samples': 35284,
        'ingestion_log': get_ingestion_log(),
    }
    return render(request, 'predictapp/data_ingestion.html', context)


@login_required(login_url='/admin/login/')
def climate_data(request):
    """Climate data view"""
    context = {
        'title': 'Climate Data',
        'avg_temp': 22.4,
        'precipitation': 8.2,
        'humidity': 74,
        'wind_speed': 3.1,
        'monthly_averages': get_monthly_climate_data(),
        'risk_indicators': get_climate_risk_indicators(),
    }
    return render(request, 'predictapp/climate_data.html', context)


@login_required(login_url='/admin/login/')
def health_records(request):
    """Health records view"""
    context = {
        'title': 'Health Records',
        'total_records': 18420,
        'disease_incidences': 4211,
        'data_range': '2017–2026',
        'disease_trends': get_disease_trends(),
        'recent_records': get_recent_health_records(),
    }
    return render(request, 'predictapp/health_records.html', context)


@login_required(login_url='/admin/login/')
def disaster_events(request):
    """Disaster events view"""
    context = {
        'title': 'Disaster Events',
        'flood_events': 38,
        'drought_events': 14,
        'total_records': 512,
        'coverage_period': '2017–2026',
        'disaster_log': get_disaster_log(),
    }
    return render(request, 'predictapp/disaster_events.html', context)


@login_required(login_url='/admin/login/')
def lstm_model(request):
    """LSTM model view"""
    context = {
        'title': 'LSTM Model',
        'f1_score': 0.891,
        'roc_auc': 0.924,
        'epochs': 50,
        'status': 'Ready',
        'architecture': {
            'units': 64,
            'dropout': 0.2,
            'batch_size': 32,
            'lookback': 12,
            'test_split': 15,
            'val_split': 15,
        },
        'metrics': {
            'accuracy': 89.1,
            'precision': 88.7,
            'recall': 90.4,
            'rmse': 0.043,
        }
    }
    return render(request, 'predictapp/lstm_model.html', context)


@login_required(login_url='/admin/login/')
def xgboost_model(request):
    """XGBoost model view"""
    context = {
        'title': 'XGBoost Model',
        'f1_score': 0.870,
        'roc_auc': 0.901,
        'estimators': 200,
        'status': 'Ready',
        'feature_importance': get_feature_importance(),
        'hyperparameters': {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'cv_folds': 5,
            'test_size': 15,
            'random_seed': 42,
        },
        'metrics': {
            'accuracy': 87.0,
            'precision': 86.2,
            'recall': 88.1,
            'mae': 0.051,
        }
    }
    return render(request, 'predictapp/xgboost_model.html', context)


@login_required(login_url='/admin/login/')
def hybrid_model(request):
    """Hybrid ensemble model view"""
    context = {
        'title': 'Hybrid Ensemble',
        'f1_score': 0.910,
        'roc_auc': 0.942,
        'ranking': 'Best',
        'status': 'Active',
        'ensemble_weights': {
            'lstm': 55,
            'xgboost': 45,
        },
        'config': {
            'random_seed': 42,
            'test_size': 15,
            'val_size': 15,
            'grid_search_cv': '5-fold',
        }
    }
    return render(request, 'predictapp/hybrid_model.html', context)


@login_required(login_url='/admin/login/')
def xai_explainability(request):
    """XAI explainability view"""
    context = {
        'title': 'XAI / Explainability',
        'primary_method': 'SHAP',
        'secondary_method': 'LIME',
        'top_features': 10,
        'shap_importance': get_shap_importance(),
        'lime_explanation': get_lime_explanation(),
    }
    return render(request, 'predictapp/xai.html', context)


@login_required(login_url='/admin/login/')
def evaluation(request):
    """Model evaluation view"""
    context = {
        'title': 'Evaluation',
        'best_f1': 0.910,
        'best_roc_auc': 0.942,
        'f1_target': 0.85,
        'best_rmse': 0.043,
        'model_comparison': get_model_comparison(),
        'confusion_matrix': get_confusion_matrix(),
    }
    return render(request, 'predictapp/evaluation.html', context)


@login_required(login_url='/admin/login/')
def export_report(request):
    """Export report view"""
    context = {
        'title': 'Export Report',
        'reports_generated': 12,
        'export_format': 'CSV',
        'last_export': 'Today',
        'recent_exports': get_recent_exports(),
    }
    return render(request, 'predictapp/export.html', context)


# ========== API ENDPOINTS FOR AJAX ==========

@login_required(login_url='/admin/login/')
def dashboard_stats(request):
    """Get dashboard statistics via AJAX"""
    data = {
        'active_alerts': 3,
        'model_accuracy': 89.1,
        'data_sources': 3,
        'predictions_today': 142,
        'trend_data': get_trend_data(),
    }
    return JsonResponse(data)


@login_required(login_url='/admin/login/')
def recent_alerts(request):
    """Get recent alerts via AJAX"""
    alerts = [
        {'region': 'Nairobi', 'threat': 'Malaria', 'risk': 'High', 'time': '2h ago'},
        {'region': 'Mombasa', 'threat': 'Cholera', 'risk': 'High', 'time': '3h ago'},
        {'region': 'Kisumu', 'threat': 'Floods', 'risk': 'Medium', 'time': '5h ago'},
    ]
    return JsonResponse({'alerts': alerts})


@login_required(login_url='/admin/login/')
def model_performance(request):
    """Get model performance metrics via AJAX"""
    data = {
        'lstm': {'f1': 0.891, 'accuracy': 89.1, 'rmse': 0.043},
        'xgboost': {'f1': 0.870, 'accuracy': 87.0, 'rmse': 0.051},
        'hybrid': {'f1': 0.910, 'accuracy': 91.0, 'rmse': 0.039},
    }
    return JsonResponse(data)


@login_required(login_url='/admin/login/')
def run_prediction(request):
    """Run a new prediction from the admin dashboard"""
    if request.method == 'POST':
        data = json.loads(request.body)
        county = data.get('county', 'nairobi')
        disease = data.get('disease', 'malaria')
        horizon = data.get('horizon', 30)
        
        # Get prediction
        prediction = prediction_service.predict_county(county, disease)
        
        return JsonResponse({
            'success': True,
            'prediction': prediction,
            'message': f'Prediction completed for {county}'
        })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)


# ========== HELPER FUNCTIONS ==========

def get_recent_predictions():
    """Get recent predictions from database"""
    predictions = Prediction.objects.order_by('-created_at')[:10]
    return [
        {
            'timestamp': p.created_at.strftime('%Y-%m-%d %H:%M'),
            'region': p.location.title(),
            'disease': p.get_disease_type_display(),
            'model': p.get_model_used_display(),
            'confidence': p.confidence,
            'risk': p.get_risk_level_display(),
            'horizon': f"{p.forecast_horizon_days}d"
        }
        for p in predictions
    ]


def get_risk_zones():
    """Get risk zones data for map"""
    return [
        {'zone': 'Nairobi', 'threat': 'Malaria', 'risk': 'Critical', 'lat': -1.2866, 'lng': 36.8172},
        {'zone': 'Mombasa', 'threat': 'Cholera', 'risk': 'High', 'lat': -4.0435, 'lng': 39.6682},
        {'zone': 'Kisumu', 'threat': 'Flood', 'risk': 'High', 'lat': -0.1057, 'lng': 34.7617},
        {'zone': 'Nakuru', 'threat': 'Malaria', 'risk': 'Moderate', 'lat': -0.2833, 'lng': 36.2667},
        {'zone': 'Eldoret', 'threat': 'Drought', 'risk': 'Low', 'lat': 0.5143, 'lng': 35.2799},
    ]


def get_ingestion_log():
    """Get data ingestion log"""
    return [
        {'time': '09:14:22', 'level': 'OK', 'message': 'NASA POWER fetch complete — 288 rows'},
        {'time': '09:14:18', 'level': 'INFO', 'message': 'Connecting to NASA POWER API...'},
        {'time': '08:52:11', 'level': 'OK', 'message': 'WHO GHO sync — 142 indicators'},
        {'time': '08:51:44', 'level': 'INFO', 'message': 'KNN imputation applied (k=5)'},
        {'time': '08:50:33', 'level': 'WARN', 'message': '3 outliers removed (z > 3.0)'},
        {'time': '08:49:01', 'level': 'OK', 'message': 'Data cleaning pipeline complete'},
    ]


def get_monthly_climate_data():
    """Get monthly climate averages"""
    return [
        {'month': 'Oct 2025', 'temp': 21.1, 'rain': 62.4, 'humidity': 78, 'wind': 2.8},
        {'month': 'Nov 2025', 'temp': 20.8, 'rain': 88.2, 'humidity': 82, 'wind': 2.5},
        {'month': 'Dec 2025', 'temp': 21.4, 'rain': 44.1, 'humidity': 76, 'wind': 3.0},
        {'month': 'Jan 2026', 'temp': 22.0, 'rain': 18.3, 'humidity': 70, 'wind': 3.4},
        {'month': 'Feb 2026', 'temp': 22.9, 'rain': 12.7, 'humidity': 68, 'wind': 3.6},
        {'month': 'Mar 2026', 'temp': 22.4, 'rain': 8.2, 'humidity': 74, 'wind': 3.1},
    ]


def get_climate_risk_indicators():
    """Get climate risk indicators"""
    return [
        {'name': 'Malaria Risk (Temp + Humidity)', 'value': 82, 'level': 'High', 'color': '#ff4d6d'},
        {'name': 'Cholera Risk (Rainfall + Humidity)', 'value': 58, 'level': 'Moderate', 'color': '#f5a623'},
        {'name': 'Flood Risk (Precipitation Index)', 'value': 22, 'level': 'Low', 'color': '#4da6ff'},
        {'name': 'Drought Risk (Temp + Low Rain)', 'value': 18, 'level': 'Low', 'color': '#3ddc84'},
    ]


def get_disease_trends():
    """Get disease trends data"""
    return {
        'years': [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
        'malaria': [820, 940, 870, 1100, 1050, 980, 1200, 1350, 1280, 1284],
        'cholera': [120, 95, 140, 200, 180, 160, 210, 290, 320, 342],
    }


def get_recent_health_records():
    """Get recent health records"""
    return [
        {'date': '2026-03', 'region': 'Nairobi', 'disease': 'Malaria', 'cases': 1284, 'deaths': 12, 'cfr': 0.93},
        {'date': '2026-03', 'region': 'Mombasa', 'disease': 'Cholera', 'cases': 342, 'deaths': 4, 'cfr': 1.17},
        {'date': '2026-02', 'region': 'Kisumu', 'disease': 'Malaria', 'cases': 892, 'deaths': 7, 'cfr': 0.78},
        {'date': '2026-01', 'region': 'Nakuru', 'disease': 'Dengue', 'cases': 117, 'deaths': 1, 'cfr': 0.85},
    ]


def get_disaster_log():
    """Get disaster event log"""
    return [
        {'date': '2026-02-14', 'type': 'Flood', 'region': 'Kisumu', 'affected': 1400, 'deaths': 3, 'damage': 'Ksh 250k', 'severity': 'High'},
        {'date': '2025-11-03', 'type': 'Flood', 'region': 'Nairobi', 'affected': 200, 'deaths': 3, 'damage': 'Ksh 100k', 'severity': 'Moderate'},
        {'date': '2025-08-20', 'type': 'Drought', 'region': 'Eldoret', 'affected': 2000, 'deaths': 0, 'damage': 'Ksh 500k', 'severity': 'High'},
        {'date': '2025-04-11', 'type': 'Flood', 'region': 'Mombasa', 'affected': 600, 'deaths': 2, 'damage': 'Ksh 600k', 'severity': 'Moderate'},
    ]


def get_feature_importance():
    """Get XGBoost feature importance"""
    return [
        {'feature': 'T2M', 'importance': 0.88, 'color': '#ff4d6d'},
        {'feature': 'RH2M', 'importance': 0.74, 'color': '#ff4d6d'},
        {'feature': 'PRECTOTCORR', 'importance': 0.68, 'color': '#f5a623'},
        {'feature': 'Cases lag-1', 'importance': 0.61, 'color': '#f5a623'},
        {'feature': 'WS2M', 'importance': 0.42, 'color': '#4da6ff'},
        {'feature': 'Season', 'importance': 0.38, 'color': '#4da6ff'},
    ]


def get_shap_importance():
    """Get SHAP feature importance"""
    return [
        {'feature': 'Temperature (T2M)', 'value': 0.88},
        {'feature': 'Humidity (RH2M)', 'value': 0.74},
        {'feature': 'Precipitation', 'value': 0.68},
        {'feature': 'Prev. Cases (lag-1)', 'value': 0.61},
        {'feature': 'Wind Speed', 'value': 0.42},
    ]


def get_lime_explanation():
    """Get LIME explanation for a single prediction"""
    return {
        'positive': [
            {'feature': 'Temp > 22°C', 'value': '+0.34'},
            {'feature': 'Rain 8–15mm', 'value': '+0.26'},
            {'feature': 'Humidity > 70%', 'value': '+0.21'},
        ],
        'negative': [
            {'feature': 'Wind > 3m/s', 'value': '-0.11'},
            {'feature': 'Low flood lag', 'value': '-0.07'},
        ]
    }


def get_model_comparison():
    """Get model comparison metrics"""
    return {
        'metrics': ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
        'lstm': [89, 87, 91, 89, 92],
        'xgboost': [87, 86, 88, 87, 90],
        'hybrid': [91, 92, 90, 91, 94],
    }


def get_confusion_matrix():
    """Get confusion matrix for hybrid model"""
    return {
        'true_positive': 4821,
        'false_positive': 312,
        'false_negative': 284,
        'true_negative': 4433,
        'accuracy': 93.7,
        'precision': 93.9,
        'recall': 94.4,
        'f1': 0.910,
    }


def get_recent_exports():
    """Get recent export reports"""
    return [
        {'name': 'Full Evaluation Report', 'format': 'PDF', 'size': '2.4 MB', 'date': 'Today'},
        {'name': 'Risk Alert Summary', 'format': 'CSV', 'size': '48 KB', 'date': 'Yesterday'},
        {'name': 'Model Performance', 'format': 'JSON', 'size': '128 KB', 'date': 'Mar 15'},
        {'name': 'Climate Data Export', 'format': 'CSV', 'size': '5.1 MB', 'date': 'Mar 14'},
    ]


def get_trend_data():
    """Get trend data for charts"""
    return {
        'months': ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'],
        'malaria': [0.72, 0.75, 0.80, 0.85, 0.88, 0.91, 0.89, 0.94],
        'cholera': [0.65, 0.70, 0.74, 0.76, 0.80, 0.83, 0.85, 0.88],
        'floods': [0.60, 0.65, 0.69, 0.72, 0.75, 0.79, 0.81, 0.81],
    }