# a = 'good'
# b = 'hello'
# c = a + ' ' + b
# print(a[::-1], 1)
()
# a = "22045612"
# b = "21304578"
# print(a.startswith('22', 0), 1)
# print(a.endswith('78', 0), 2)
# print(b.startswith('22', 0), 3)
# print(b.endswith('78', 0), 4)

# a = 'bigger'
# print(a.upper())
# b = 'BIGGER'
# print(b.lower())

# a = "str_1"
# b = "str_2"

# sub_a = a[0:len(a)]
# sub_b = b[0:len(b)]
# print(sub_a, sub_b)
# if sub_a == sub_b:
#     print("两个字串相同")
# else:
#     print("两个字串不同")

# a = 'str'
# b = []
# for i in range(1, 11):
#     b.append(str(i))
# c = ''.join(b)
# new_a = a + c
# print(new_a)
# 简单的多线程操作
# import time
# import random
# # 引入多线程的库
# from concurrent.futures import ThreadPoolExecutor

# def la(name):
#     print("%s is 正在拉" % name)
#     time.sleep(random.randint(1, 3))
#     res = random.randint(1, 10)
#     if res == 6:
        
#         raise EnvironmentError()
#     # 对下一个操作进行调用
#     return weight({'name':name, 'res':res})

# def weight(la):
#     name = la['name']
#     size = la['res']
#     return test({'name':name, 'size':size})

# def test(weight):
#     print(weight["name"])
#     print(weight["size"])
#     return 0
    
# if __name__ == '__main__':
#    with ThreadPoolExecutor(3) as thread:
#     futures = [
#     thread.submit(la, 'jack'),
#     thread.submit(la, 'mack'),
#     thread.submit(la, 'wuhu'),
#         ]

#     for future in futures:
#         try:
#             result = future.result()
#             print(result)
#         except Exception as e:
#             print(e)
# import time
# import random
# # 引入多线程的库
# from concurrent.futures import ProcessPoolExecutor

# def la(name):
#     try:
#         print("%s is 正在拉" % name)
#         time.sleep(random.randint(1, 3))
#         res = random.randint(1, 10)
#         # 对下一个操作进行调用
#         return weight({'name':name, 'res':res})
#     except Exception as e:
#         return e

# def weight(la):
#     try:
#         name = la['name']
#         size = la['res']
#         return test({'name':name, 'size':size})
#     except Exception as e:
#         return e

# def test(weight):
#     print(weight["name"], 1)
#     print(weight["size"], 2)
    
# if __name__ == '__main__':
#     name_list = ['jack', 'mike', 'peter']
#     process_num = len(name_list)
    
#     with ProcessPoolExecutor(process_num) as pool:
#         futures = [
#             pool.submit(la, name) for name in name_list
#         ]
    
#         for future in futures:
#             try:
#                 result = future.result()
#                 print(f"任务成功:{result}")
#             except:
#                 print(f"任务失败：{result}")
            
# import re
# import asyncio
# import aiohttp
# from urllib.parse import urljoin, urlparse
# TITLE_RE = re.compile(r'<title[^>]*>(.*?)</title>', re.IGNORECASE | re.DOTALL)
# async def fetch_title(session, url):
#     try:
#         # 添加超时防止卡住
#         async with session.get(url, timeout=10) as response:
#             if response.content_type != 'text/html':
#                 return url, "非HTML页面"
            
#             text = await response.text()
#             match = TITLE_RE.search(text)
#             title = match.group(1).strip() if match else "未找到标题"
#             # 去掉多余的空白和换行
#             title = ' '.join(title.split())
#             return url, title

#     except asyncio.TimeoutError:
#         return url, "请求超时"
#     except Exception as e:
#         return url, f"错误: {str(e)}"
# async def main():
#     # 要爬取的URL列表
#     urls = [
#         'https://httpbin.org/html',
#         'https://httpbin.org/html',
#         'https://example.com',
#         'https://httpbin.org/html',
#         'https://www.python.org',
#         'https://httpbin.org/delay/2',  # 测试延迟
#     ]

#     # 创建一个 aiohttp 的 ClientSession
#     async with aiohttp.ClientSession() as session:
#         # 并发地发起所有请求
#         tasks = [fetch_title(session, url) for url in urls]
#         results = await asyncio.gather(*tasks)

#         # 打印结果
#         print("爬取结果：")
#         for url, title in results:
#             print(f"{url} -> {title}")


# # 运行异步主函数
# if __name__ == "__main__":
#     import time
#     start = time.time()
#     asyncio.run(main())
#     print(f"总耗时: {time.time() - start:.2f} 秒")

# a = [[0, 1], [2, 3], [4, 5]]
# for i in range(len(a)):
#     for j in range(len(a[i])):
#         print(a[i][j])


# 简单的异步编程示例
# import asyncio
# import time

# # 定义异步任务
# async def wash_vegetables():
#     print("👩‍🍳 开始洗菜")
#     await asyncio.sleep(3)  # 模拟洗菜耗时 3 秒（比如等水）
#     print("✅ 洗菜完成")
#     return "干净的蔬菜"

# async def cut_vegetables():
#     print("🔪 开始切菜")
#     await asyncio.sleep(2)  # 模拟切菜耗时 2 秒
#     print("✅ 切菜完成")
#     return "切好的菜"

# async def turn_on_stove():
#     print("🔥 开始开火")
#     await asyncio.sleep(1)  # 模拟点火，等 1 秒
#     print("✅ 火开了")
#     return "燃烧的炉子"

# # 主函数：同时做这三件事
# async def main():
#     print(f"开始时间: {time.strftime('%X')}")

#     # 并发执行三个任务（不是串行！）
#     results = await asyncio.gather(
#         wash_vegetables(),
#         cut_vegetables(),
#         turn_on_stove()
#     )

#     print(f"全部完成时间: {time.strftime('%X')}")
#     print("任务结果:", results)

# # 运行异步程序
# if __name__ == '__main__':
#     asyncio.run(main())

# squares = [(x, y) for x in range(1, 11)  for y in range(1, 11)]
# print(squares)

# import requests
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import time
# from typing import List, Dict, Any

# # 配置项
# TEST_URL = "https://httpbin.org/get"  # 测试接口（GET）
# # TEST_URL = "https://httpbin.org/post"  # 如果测试 POST，取消注释这个
# REQUEST_COUNT = 10                    # 并发请求数量
# MAX_WORKERS = 5                       # 最大线程数（并发度）

# def test_get_api(url: str, request_id: int) -> Dict[str, Any]:
#     """测试 GET 接口"""
#     start_time = time.time()
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()  # 如果状态码不是 200 会抛异常
#         elapsed = time.time() - start_time
#         return {
#             "request_id": request_id,
#             "status": "success",
#             "status_code": response.status_code,
#             "elapsed": round(elapsed, 3),
#             "data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
#         }
#     except Exception as e:
#         elapsed = time.time() - start_time
#         return {
#             "request_id": request_id,
#             "status": "failed",
#             "error": str(e),
#             "elapsed": round(elapsed, 3)
#         }

# def test_post_api(url: str, request_id: int, payload: dict = None) -> Dict[str, Any]:
#     """测试 POST 接口"""
#     if payload is None:
#         payload = {"id": request_id, "message": "Hello from thread"}
#     start_time = time.time()
#     try:
#         response = requests.post(url, json=payload, timeout=10)
#         response.raise_for_status()
#         elapsed = time.time() - start_time
#         return {
#             "request_id": request_id,
#             "status": "success",
#             "status_code": response.status_code,
#             "elapsed": round(elapsed, 3),
#             "sent_data": payload,
#             "received_data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
#         }
#     except Exception as e:
#         elapsed = time.time() - start_time
#         return {
#             "request_id": request_id,
#             "status": "failed",
#             "error": str(e),
#             "elapsed": round(elapsed, 3)
#         }

# def run_interface_test():
#     """主测试函数"""
#     print(f"🚀 开始接口测试...")
#     print(f"   URL: {TEST_URL}")
#     print(f"   请求总数: {REQUEST_COUNT}")
#     print(f"   并发线程数: {MAX_WORKERS}")
#     print("-" * 60)

#     start_time = time.time()

#     # 使用线程池
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         # 提交所有任务
#         futures = []
#         for i in range(1, REQUEST_COUNT + 1):
#             # 根据 URL 选择 GET 或 POST
#             if "post" in TEST_URL:
#                 future = executor.submit(test_post_api, TEST_URL, i)
#             else:
#                 future = executor.submit(test_get_api, TEST_URL, i)
#             futures.append(future)

#         # 收集结果
#         results = []
#         for future in as_completed(futures):
#             result = future.result()
#             results.append(result)

#             # 实时打印结果（可选）
#             if result["status"] == "success":
#                 print(f"✅ 请求 {result['request_id']} 成功 | 耗时: {result['elapsed']}s | 状态码: {result['status_code']}")
#             else:
#                 print(f"❌ 请求 {result['request_id']} 失败 | 耗时: {result['elapsed']}s | 错误: {result['error']}")

#     # 统计结果
#     total_time = time.time() - start_time
#     success_count = sum(1 for r in results if r["status"] == "success")
#     failed_count = REQUEST_COUNT - success_count
#     avg_time = sum(r["elapsed"] for r in results) / len(results) if results else 0

#     print("-" * 60)
#     print("📊 测试结果汇总:")
#     print(f"   总请求数: {REQUEST_COUNT}")
#     print(f"   成功: {success_count} | 失败: {failed_count}")
#     print(f"   成功率: {success_count / REQUEST_COUNT * 100:.1f}%")
#     print(f"   平均响应时间: {avg_time:.3f}s")
#     print(f"   总耗时: {total_time:.3f}s")
#     print("✅ 测试完成！")

# if __name__ == "__main__":
#     run_interface_test()

# import requests
# from concurrent.futures import ThreadPoolExecutor, as_completed
# def test_post_api(api, times, message):
#     try:
#         response = requests.post(api, json=message, timeout=10)
#         response.raise_for_status()
#         if response.headers.get('content-type', '').startswith('application/json'):
#             content = response.json()
#         else:
#             content  = response.text
#         test_result = {
#                         'times':times,
#                         'status':'success',
#                         'data': "success",
#                         }
    
#         return test_result
    
#     except Exception as e:
#         test_result = {
#                         'times': times,
#                         'status': 'failed',
#                         'status_code': response.status_code,
#                         'reason': e,
#                         }
        
#         return test_result
    
# if __name__ == '__main__':
#     future_list = []
#     future_result_list = []
#     message_dict = {"message" : "Hello World!"}
#     test_api_list = ['http://temple.com/post'] * 100
    
#     with ThreadPoolExecutor(max_workers=len(test_api_list)) as executor:
#         for i in range(0, len(test_api_list)):
#             future = executor.submit(test_post_api, test_api_list[i], i + 1, message_dict)
#             future_list.append(future)
        
#         for future in as_completed(future_list):
#             result = future.result()
#             future_result_list.append(result['status_code'])
    
#     print(future_result_list)

# import pytest

# def add(a, b):
#     return a + b

# def subtract(a, b):
#     return a - b

# def test_add():
#     assert add(2, 3) == 5
#     assert add(-1, 1) == 0
#     assert add(0, 0) == 0
    
# def test_subtract():
#     assert subtract(5, 3) == 2
#     assert subtract(0, 5) == -5
#     assert subtract(-1, -1) == 0
    
# def test_add_with_floats():
#     assert abs(add(0.1, 0.2) - 0.3) < 1e-9
    
# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])

# from playwright.sync_api import sync_playwright
# import time

# # 启动 Playwright
# with sync_playwright() as p:
#     # 启动 Chromium 浏览器（无头模式设为 False 可见操作过程）
#     browser = p.chromium.launch(headless=False)
#     context = browser.new_context(
#         user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
#         viewport={"width": 1920, "height": 1080},
#         device_scale_factor=1,
#         is_mobile=False,
#         has_touch=False,
#         locale="zh-CN",
#         timezone_id="Asia/Shanghai",
#         # 权限
#         permissions=["geolocation", "notifications"],
#     )

#     page = context.new_page()

#     # 1️⃣ 访问百度首页
#     print("🌐 正在打开百度...")
#     page.goto("https://www.baidu.com")

#     # 2️⃣ 等待搜索框出现并输入关键词
#     print("正在清空搜索词...")
#     page.locator("#kw").clear()
#     print("🔍 正在输入搜索词...")
#     page.locator("#kw").fill("Playwright 自动化测试")
    
#     # 3️⃣ 点击“百度一下”按钮
#     print("🖱️  正在点击搜索按钮...")
#     page.locator("#su").click()

#     # 4️⃣ 等待搜索结果加载（等待某个结果链接出现）
#     print("⏳ 等待搜索结果加载...")
#     page.wait_for_selector("text=自动化测试", timeout=10000)  # 等待包含“自动化测试”的文本出现

#     # 5️⃣ （可选）截图保存结果页
#     screenshot_path = "baidu_search_result.png"
#     page.screenshot(path=screenshot_path)
#     print(f"📸 截图已保存：{screenshot_path}")

#     # 6️⃣ （可选）打印前几个搜索结果标题
#     print("\n📄 搜索结果标题：")
#     results = page.locator("h3 a")  # 百度搜索结果标题通常在 h3 > a 中
#     for i in range(min(5, results.count())):  # 只打印前5个
#         title = results.nth(i).text_content()
#         print(f"  {i+1}. {title.strip()}")

#     # 7️⃣ 等待几秒再关闭（方便观察）
#     time.sleep(3)

#     # 8️⃣ 关闭浏览器
#     browser.close()
#     print("✅ 操作完成，浏览器已关闭。")

# 装饰器示例
# 2.带参数装饰器模板
# from functools import wraps

# def decorator_with_args(param):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             # 使用param
#             return func(*args, **kwargs)
#         return wrapper
#     return decorator

nums_list_1 = [i for i in range(0, 101, 2)]
nums_list_2 = [i for i in range(0, 101) if i % 2 == 0]
if nums_list_1 == nums_list_2:
    print("1")