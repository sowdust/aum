from django.urls import path
from . import views

urlpatterns = [
    path('radios/', views.radios_list, name='radios_list'),
    path('radios/<slug:slug>', views.radio_detail, name='radio_detail'),
    path('radios/<slug:slug>/recordings', views.radio_recordings, name='radio_recordings'),
    path('', views.radios_list, name='index'),

]