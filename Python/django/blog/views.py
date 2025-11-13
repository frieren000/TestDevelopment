import json
from .models import Users
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import FileSystemStorage


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
    try:
        queryset = Users.objects.all()

        user_id = request.GET.get('id')
        name = request.GET.get('name')
        email = request.GET.get('email')

        if user_id:
            queryset = queryset.filter(id=user_id)

        if name:
            queryset = queryset.filter(name__icontains=name)

        if email:
            queryset = queryset.filter(email=email)

        data_list = list(queryset.values())

        if data_list:
            return Response({
                'status': 200,
                'message': 'success',
                'data': data_list,
            }, status=200)
        else:
            return Response({
            'status': 500,
            'message': '数据不存在!',
        }, status=500)

    except Exception as e:
        return Response({
            'status': 500,
            'message': 'failed',
            'data': str(e),
        }, status=500)

@api_view(['POST'])
def user_register(request):
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

        if Users.objects.filter(name=name).exists():
            return Response({'message': '用户已存在!'}, status=400)

        Users.objects.create(
            name=name,
            phone_num=phone_num,
            email=email,
            company=company,
            job=job,
            ipv4=ipv4
        )

        return Response({
            'code': 200,
            'message': '注册成功!'
        }, status=201)

    except Exception as e:
        # 记录日志更佳，此处简化
        return Response({
            'message': f'服务器错误: {str(e)}'
        }, status=500)
            
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
            upload_file_path = '../django/uploader_files'
            upload_file = request.FILES['file']
            save_file = FileSystemStorage(location=upload_file_path)
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

        num_to_index = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_to_index:
                two_sum_list = [num_to_index[complement], i]
                break
            num_to_index[num] = i
        else:
            # 如果没找到
            two_sum_list = []

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
        if num_bottles > 0 and num_exchange > 0:
            ans = num_bottles + (num_bottles - 1) // (num_exchange - 1)
            message_dict = {
                'code': 200,
                'message': 'success',
                'data': ans,
            }
            status = 200
        else:
            message_dict = {
                'code': 400,
                'message': 'failed',
                'data': '参数小于零',
            }
            status = 400
        
    except Exception as e:
        message_dict = {
            'code': 400,
            'message': 'failed',
            'data': str(e),
        }
        status = 400
        
    return Response(message_dict, status=status)