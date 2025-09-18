import json
import requests

def test_api_search_user(params):
    test_api_url = 'http://127.0.0.1:8000/api/search_users_info/'
    response = requests.get(test_api_url, params, timeout=5)
    print(response.status_code)
    print(response.json())
    
params = {
    'email': 'leijie@example.com',
    }
test_api_search_user(params)