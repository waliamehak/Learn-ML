from django.urls import path
from . import views

urlpatterns = [
    path('', views.Home.as_view(), name='learner_home'),
    path('login', views.LoginPage.as_view(), name='login'),
    path('logout', views.LogoutPage.as_view(), name='logout'),
    path('register', views.RegisterPage.as_view(), name='register'),
    path('classification', views.Classification.as_view(), name='learner_classification'),
    path('regression', views.Regression.as_view(), name='learner_regression'),
    path('clustering', views.Clustering.as_view(), name='learner_clustering')
]
