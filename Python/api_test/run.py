import os
import sys
import shutil
import subprocess

def run_tests():
    # 1. 确保报告目录存在
    results_dir = "reports/allure-results"
    report_dir = "reports/allure-report"
    os.makedirs(results_dir, exist_ok=True)

    # 2. 清理旧结果（更安全的方式）
    for file in os.listdir(results_dir):
        file_path = os.path.join(results_dir, file)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    # 3. 运行 pytest 并指定 --alluredir 生成结果
    print("开始运行测试...")
    result = subprocess.run([
        sys.executable, "-m", "pytest",  
        "--alluredir", results_dir,      
        "-v"    
    ])
    
    if result.returncode != 0:
        print("测试执行完成，但部分用例失败。")
    else:
        print("所有测试通过！")

    # 4. 检查是否有结果文件
    if not os.listdir(results_dir):
        print("未生成任何 Allure 结果文件，请检查 pytest 是否正常运行。")
        return

    # 5. 查找 allure 命令（安全方式）
    allure_path = shutil.which("allure")
    if not allure_path:
        print("未找到 Allure CLI,跳过报告生成。")
        print("请安装 Allure Commandline: https://docs.qameta.io/allure-report/docs/getting-started/")
        print(f"但原始结果仍保存在: {os.path.abspath(results_dir)}")
        return

    # 6. 生成 Allure 报告
    print("正在生成 Allure 报告...")
    report_result = subprocess.run([
        allure_path,
        "generate",
        results_dir,
        "-o", report_dir,
        "--clean"
    ])

    if report_result.returncode == 0:
        print(f"测试完成！报告路径: file://{os.path.abspath(report_dir)}/index.html")
    else:
        print("Allure 报告生成失败，请检查 Allure CLI 是否正常安装。")

if __name__ == '__main__':
    run_tests()