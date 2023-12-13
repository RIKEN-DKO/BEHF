from django.urls import path
from . import views

urlpatterns = [
    path('', views.search, name='search'),
    path('document/<int:doc_id>/', views.document, name='document'),
    path('document/<int:doc_id>/<str:resultTextAE>/', views.document, name='document'),

]
