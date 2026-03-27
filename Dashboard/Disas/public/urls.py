# public/urls.py

from django.urls import path
from . import views

app_name = 'public'

urlpatterns = [
    # Home page at root
    path('', views.home, name='home'),
    
    # API endpoints
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/action-plan/', views.api_action_plan, name='api_action_plan'),
]