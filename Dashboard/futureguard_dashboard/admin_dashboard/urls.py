from django.urls import path
from . import views

app_name = 'admin_dashboard'

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('run-prediction/', views.run_prediction, name='run_prediction'),
    path(
        'emergency-reports/<int:report_id>/status/',
        views.update_emergency_report_status,
        name='update_emergency_report_status',
    ),
]
