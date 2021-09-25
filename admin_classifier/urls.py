from django.urls import path
from . import views

urlpatterns = [
    path('', views.Home.as_view(), name='admin_home'),
    path('classification', views.Classification.as_view(), name='admin_classification'),
    path('regression', views.Regression.as_view(), name='admin_regression'),
    path('clustering', views.Clustering.as_view(), name='admin_clustering')
]
