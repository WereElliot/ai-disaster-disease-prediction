from django.urls import path
from . import views

app_name = 'community_dashboard'

urlpatterns = [
    path('', views.home, name='dashboard'),
    path('predictions/', views.predictions, name='predictions'),
    path('emergency-report/', views.emergency_report, name='emergency_report'),
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/action-plan/', views.api_action_plan, name='api_action_plan'),
]
