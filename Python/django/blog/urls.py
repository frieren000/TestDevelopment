from django.urls import path
from . import views

urlpatterns = [
    path('hello/', views.hello),
    path('search_user_info/', views.search_users_info)
]
