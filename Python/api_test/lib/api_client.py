# API客户端
import os
import time
import yaml
import logging
import requests

from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from typing import Optional, Dict, Any, Union, Tuple

logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self, env: str = 'test'):
        config_path = f'config/env/{env}.yaml'
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {os.path.abspath(config_path)}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 解析失败 ({config_path}): {e}")

        if 'base_url' not in self.config:
            raise KeyError("配置文件中缺少 'base_url' 字段")

        self.base_url = self.config['base_url'].rstrip('/')
        self.session = requests.Session()

        # 重试 
        retry_strategy = Retry(
            total=self.config.get('retry', {}).get('total', 3),
            backoff_factor=self.config.get('retry', {}).get('backoff_factor', 1),
            status_forcelist=self.config.get('retry', {}).get('status_forcelist', [429, 500, 502, 503, 504]),
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST", "PATCH"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # 默认headers
        default_headers = {"Content-Type": "application/json"}
        config_headers = self.config.get('headers', {})
        default_headers.update(config_headers)
        self.session.headers.update(default_headers)

        # Token刷新
        self._token_refresh_callback = self.config.get('auth', {}).get('refresh_callback')
        self._token_header_name = self.config.get('auth', {}).get('header_name', 'Authorization')
        self._should_retry_on_401 = self.config.get('auth', {}).get('auto_refresh', False)

        logger.info(f"APIClient 初始化完成 | 环境: {env} | Base URL: {self.base_url}")

    def _build_url(self, endpoint: str) -> str:
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint
        return self.base_url + endpoint

    def _log_request(self, method: str, url: str, **kwargs):
        logger.info(f"→ {method.upper()} {url}")
        if kwargs.get('params'):
            logger.debug(f"   Query Params: {kwargs['params']}")
        if kwargs.get('json'):
            logger.debug(f"   JSON Body: {kwargs['json']}")
        elif kwargs.get('data') and not kwargs.get('files'):
            logger.debug(f"   Form/Data Body: {kwargs['data']}")
        if kwargs.get('files'):
            logger.debug(f"   Files: {list(kwargs['files'].keys())}")

    def _log_response(self, resp: requests.Response):
        logger.info(f"← {resp.status_code} {resp.reason} | URL: {resp.url}")
        if resp.text:
            preview = resp.text[:500] + ('...' if len(resp.text) > 500 else '')
            logger.debug(f"   Response Body: {preview}")

    def _refresh_token(self):
        """调用用户提供的回调函数刷新 token"""
        if not self._token_refresh_callback:
            raise RuntimeError("未配置 token 刷新回调函数，但收到了 401 响应")

        logger.info("触发 Token 刷新...")
        try:
            new_token = self._token_refresh_callback()
            if not new_token:
                raise ValueError("Token 刷新回调未返回有效 token")
            # 更新 session headers
            self.session.headers[self._token_header_name] = f"Bearer {new_token}"
            logger.info("Token 已更新")
        except Exception as e:
            logger.error(f"Token 刷新失败: {e}")
            raise

    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Union[Dict, str]] = None,
        json: Optional[Dict] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> requests.Response:
        """
        发送 HTTP 请求（支持文件上传）
        """
        url = self._build_url(endpoint)
        req_kwargs = {
            'params': params,
            'data': data,
            'json': json,
            'files': files,
            'headers': headers,
            'timeout': timeout or self.config.get('timeout', 10),
            **kwargs
        }

        # 如果上传文件，移除默认的 Content-Type(让 requests 自动设置 multipart)
        if files:
            req_headers = req_kwargs.get('headers', {})
            if 'Content-Type' in self.session.headers and 'Content-Type' not in req_headers:
                pass  

        self._log_request(method, url, **{k: v for k, v in req_kwargs.items() if v is not None})

        max_retries_on_401 = 1 
        attempt = 0

        while attempt <= max_retries_on_401:
            try:
                resp = self.session.request(method, url, **req_kwargs)
            except requests.exceptions.RequestException as e:
                logger.error(f"请求异常: {e}")
                raise

            # 检查是否需要刷新 token
            if (
                resp.status_code == 401
                and self._should_retry_on_401
                and attempt == 0  # 只重试一次
            ):
                logger.warning("收到 401,尝试刷新 Token 并重试请求...")
                self._refresh_token()
                attempt += 1
                continue  # 重试当前请求
            else:
                break  # 正常响应或已重试过

        self._log_response(resp)
        return resp

    # HTTP请求
    def get(self, endpoint: str, **kwargs) -> requests.Response:
        return self.request('GET', endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs) -> requests.Response:
        return self.request('POST', endpoint, **kwargs)

    def put(self, endpoint: str, **kwargs) -> requests.Response:
        return self.request('PUT', endpoint, **kwargs)

    def patch(self, endpoint: str, **kwargs) -> requests.Response:
        return self.request('PATCH', endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> requests.Response:
        return self.request('DELETE', endpoint, **kwargs)

    # 文件上传
    def upload_file(self, endpoint: str, file_path: str, field_name: str = 'file', **kwargs) -> requests.Response:
        """
        便捷文件上传方法
        :param endpoint: 接口路径
        :param file_path: 本地文件路径
        :param field_name: 表单字段名(默认 'file')
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        with open(file_path, 'rb') as f:
            files = {field_name: f}
            return self.post(endpoint, files=files, **kwargs)