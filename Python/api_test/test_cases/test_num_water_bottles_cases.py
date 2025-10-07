import yaml
import pytest
from lib.api_client import APIClient
from lib.assertions import assert_response_code, assert_json_body

# 加载测试数据
with open('data/num_water_bottles_cases.yaml', encoding='utf-8') as f:
    test_cases = yaml.safe_load(f)

class TestNumWaterBottles:
    @pytest.fixture(scope="class")
    def client(self):
        return APIClient(env='test')  # 使用 test 环境

    @pytest.mark.parametrize("case", test_cases, ids=[c['name'] for c in test_cases])
    def test_two_sum(self, client, case):
        # 发送 POST 请求
        response = client.post("num_water_bottles/", json=case['payload'])

        # 断言状态码
        assert_response_code(response, case['expected_status'])

        # 断言响应体
        if 'expected_json' in case:
            assert_json_body(response, case['expected_json'])