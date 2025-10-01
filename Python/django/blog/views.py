import json
import pymysql
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import FileSystemStorage
from tools import base_tools
from .models import Users

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
        select * from users where 1 = 1
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
        cursor.execute(search_user_sql, params)
        
        for col in cursor.description:
            columns.append(col[0])
            
        rows = cursor.fetchall()
        data_list = [dict(zip(columns, row)) for row in rows]
        
        data_content_dict = {
            'status': 200,
            'message': 'success',
            'data': data_list,
        }
        status = 200
        
    except Exception as e:
        data_content_dict = {
            'status':500,
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

@api_view(['POST'])
def user_register(request):
    cursor = None
    connect = None
    try:
        data = request.data
        
        name = data.get('name')
        phone_num = data.get('phone_num')
        email = data.get('email')
        company = data.get('company')
        job = data.get('job')
        ipv4 = data.get('ipv4')

        if not name:
            return Response({'message': '用户名不能为空'}, status=400)

        if not all([phone_num, email, company, job, ipv4]):
            return Response({'message': '请填写完整信息！'}, status=400)

        connect = pymysql.connect(**base_tools.database_config)
        cursor = connect.cursor()

        # 检查用户是否已存在
        cursor.execute("SELECT 1 FROM users WHERE name = %s", (name,))
        if cursor.fetchone():
            return Response({'message': '用户已存在!'}, status=400)

        # 插入新用户
        cursor.execute('''
            INSERT INTO users(name, phone_num, email, company, job, ipv4) 
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', [name, phone_num, email, company, job, ipv4])
        
        connect.commit()

        return Response({
            'code': 200,
            'message': '注册成功!'
        }, status=201)

    except Exception as e:
        return Response({
            'message': f'服务器错误: {str(e)}'
        }, status=500)

    finally:
        if cursor:
            cursor.close()
        if connect:
            connect.close()
            
@api_view(['POST'])
def files_uploader(request):
    # 文件上传接口
    try:
        if 'file' not in request.FILES:
            message_dict = {
                'error_msg': '未选择文件!'
            }
            status = 400
            
            return Response(message_dict, status=status)
        
        else:
            upload_file = request.FILES['file']
            save_file = FileSystemStorage(location=base_tools.upload_file_path)
            file_name = save_file.save(upload_file.name, upload_file)
            message_dict = {
                'message': '文件上传成功!',
                'file_name': file_name,
            }
            status = 200
            
            return Response(message_dict, status=status)
        
    except Exception as e:
        err_msg = str(e)
        
        message_dict = {
            'message': '文件上传失败!',
            'error_msg': err_msg,
        }
        status = 400
        
        return Response(message_dict, status=status)

@api_view(['GET'])
def search_user_info_by_model(request):
    # 通过model文件查询数据库
    users = Users.objects.all()
    user_list = []
    for user in users:
            user_list.append({
                'name': user.name,
                'phone_num': user.phone_num,
                'email': user.email,
                'company': user.company,
                'job': user.job,
                'ipv4': user.ipv4,
            })
        
    return Response({
        'code': 200,
        'message': 'success',
        'data': user_list,
    })
    
@api_view(['POST'])
# 两数之和
def two_sum(request):
    try:
        data = json.loads(request.body)
        nums = data['nums']
        target = data['target']
        
        two_sum_list = []
        left_point = 0
        right_point = len(nums) - 1
        while left_point < right_point:
            sums = nums[left_point] + nums[right_point]
            if sums < target:
                left_point += 1
            elif sums == target:
                two_sum_list = [left_point + 1, right_point + 1]
                break
            else:
                right_point -= 1
                
        
        message_dict = {
            'code': 200,
            'message': 'success',
            'data': two_sum_list,
        }
        status = 200
    
    except Exception as e:
        message_dict = {
            'code': 400,
            'message': 'failed',
            'data': str(e),
        }
        status = 400
        
    return Response(message_dict, status=status)

@api_view(['POST'])
# 换水问题
def num_water_bottles(request):
    try:
        data = json.loads(request.body)
        num_bottles = data['numBottles']
        num_exchange = data['numExchange']
        
        ans = num_bottles + (num_bottles - 1) // (num_exchange - 1)
        
        message_dict = {
            'code': 200,
            'message': 'success',
            'data': ans,
        }
        status = 200
        
    except Exception as e:
        message_dict = {
            'code': 400,
            'message': 'failed',
            'data': str(e),
        }
        status = 400
        
    return Response(message_dict, status=status)