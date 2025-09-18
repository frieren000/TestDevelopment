from django.urls import path
from . import views

urlpatterns = [
    path('hello/', views.hello),
    path('search_users_info/', views.search_users_info),
    path('user_register/', views.user_register),
]
