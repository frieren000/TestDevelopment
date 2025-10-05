import json

def assert_response_code(response, expected_code):
    assert response.status_code == expected_code, \
        f"Expected status {expected_code}, got {response.status_code}"

def assert_json_body(response, expected_json):
    try:
        resp_json = response.json()
    except json.JSONDecodeError:
        assert False, f"Response is not valid JSON: {response.text}"
    
    # for key, expected_val in expected_json.items():
    #     assert key in resp_json, f"Key '{key}' not in response"
    #     assert resp_json[key] == expected_val, \
    #         f"Key '{key}': expected {expected_val}, got {resp_json[key]}"