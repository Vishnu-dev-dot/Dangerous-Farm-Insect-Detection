from django.urls import path
from . import views

app_name = 'insects'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_and_predict, name='upload'),
    path('calculate/', views.calculate_dosage, name='calculate'),
    path('gallery/',views.gallery,name='gallery')
]
