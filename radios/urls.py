from django.urls import path
from . import views

urlpatterns = [
    path('radios/', views.radios, name='radios'),
]