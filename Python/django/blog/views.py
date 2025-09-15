import pymysql
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from tools import base_tools


@api_view(['GET'])
def hello(request):
    # 测试API
    hello_dict = {
        'message': 'Hello from Django',
        'status': 'success',
    }
    return JsonResponse(hello_dict)

@api_view(['GET'])
def search_users_info(request):
    # 查找用户信息
    connect = pymysql.connect(**base_tools.database_config)
    cursor = connect.cursor()
    
    search_user_sql = f'''
    SELECT * FROM users;
    '''
    cursor.execute(search_user_sql)
    rows = cursor.fetchall()
    data = list(rows)
    cursor.close()
    connect.close()
    
    data_content_dict = {
        'code': 200,
        'message': 'success',
        'data': data,
    }
    
    return JsonResponse(data_content_dict)