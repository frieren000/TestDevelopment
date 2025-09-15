import pymysql
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
    return Response(hello_dict)

@api_view(['GET'])
def search_users_info(request):
    # 查找用户信息
    data_list, columns = [], []
    
    
    try:
        search_user_sql = f'''
        select * form users where 1 = 1
        '''
        params = []
        user_id = request.GET.get('id')
        name = request.GET.get('name')
        email = request.GET.get('email')
        
        if user_id:
            search_user_sql += 'and id = %s'
            params.append(user_id)
        
        if name:
            search_user_sql += 'and name like %s'
            params.append(f'%{name}%')
        
        if email:
            search_user_sql += 'and email = %s'
            params.append(email)
        
        connect = pymysql.connect(**base_tools.database_config)
        cursor = connect.cursor()
        cursor.execute(search_user_sql)
        
        for col in cursor.description:
            columns.append(col[0])
            
        rows = cursor.fetchall()
        data_list = [dict(zip(columns, row)) for row in rows]
        
        data_content_dict = {
            'code': 200,
            'message': 'success',
            'data': data_list,
        }
        status = 200
        
    except Exception as e:
        data_content_dict = {
            'code':500,
            'message': 'failed',
            'data': str(e),
        }
        status = 500
        
    finally:
        if cursor:
            cursor.close()
        if connect:
            connect.close()
            
    return Response(data_content_dict,status=status)