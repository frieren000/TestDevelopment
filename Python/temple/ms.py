# import os
# import time

# def times(func):
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         print(end_time - start_time)
#         return result
#     return wrapper
            
# @times
# def panduan_wenjian_cunzai(mubiao_file_path):
#     file_count = 0
#     for root, dirs, files in os.walk(mubiao_file_path):
#         for file in files:
#             file_path = os.path.join(root, file)
#             print(file_path)
#             file_count += 1
            
# if __name__ == "__main__":
#     mubiao_file_path = '/home/ubuntu/codes/Java'
#     if os.path.exists(mubiao_file_path):
#         all_files = panduan_wenjian_cunzai(mubiao_file_path)
#     else:
#         print(f"{mubiao_file_path}目标路径不存在!")
        
# def liang_shu_zhi_he(nums_list, target):
#     for i in range(0, len(nums_list)):
#         mubiao_num = target - nums_list[i]
#         for j in range(i, len(nums_list)):
#             if nums_list[j] == mubiao_num and i != j:
#                 print(i + 1, j + 1)
                # pass
    
        
# nums_list = [20, 70, 110, 150]
# target = 90
# liang_shu_zhi_he(nums_list, target)

#!/usr/bin/env python3
# from playwright.sync_api import sync_playwright
# import os

# print("🔍 正在检查 Playwright 环境...")

# # 1. 检查浏览器二进制
# print("\n📦 浏览器安装状态:")
# with sync_playwright() as p:
#     for name, browser in [("Chromium", p.chromium), ("Firefox", p.firefox), ("WebKit", p.webkit)]:
#         try:
#             path = browser.executable_path
#             if os.path.exists(path):
#                 print(f"✅ {name}: {path}")
#             else:
#                 print(f"❌ {name}: 路径不存在")
#         except Exception:
#             print(f"❌ {name}: 未安装或初始化失败")

# # 2. 检查系统依赖（仅 Linux）
# print("\n🧩 系统依赖库（Linux）:")
# if os.name == "posix" and os.path.exists("/usr/bin/dpkg"):
#     os.system("dpkg -l | grep -E '(libicu|gstreamer|flite|libavif)' 2>/dev/null || echo '未检测到关键依赖包（或非Debian系系统）'")

# print("\n🎉 检查完成！")