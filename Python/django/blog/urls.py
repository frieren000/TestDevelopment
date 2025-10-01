from django.urls import path
from . import views

urlpatterns = [
    path('hello/', views.hello),
    path('search_users_info/', views.search_users_info),
    path('user_register/', views.user_register),
    path('files_upload/', views.files_uploader),
    path('search_user_info_by_model/', views.search_user_info_by_model),
    
    # 以下为以LC题目解法为内容的API
    path('two_sum/', views.two_sum),  # 两数之和
    path('num_water_bottles/', views.num_water_bottles), # 换水问题
]
