#!/usr/bin/env python3
"""
稳定性测试脚本
用于验证pose_stable.py的各项修复是否正常工作
"""

import os
import sys
import time
import subprocess
import signal

class Colors:
    """终端颜色"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """打印标题"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}\n")

def print_success(text):
    """打印成功信息"""
    print(f"{Colors.GREEN}✓{Colors.ENDC} {text}")

def print_error(text):
    """打印错误信息"""
    print(f"{Colors.RED}✗{Colors.ENDC} {text}")

def print_warning(text):
    """打印警告信息"""
    print(f"{Colors.YELLOW}⚠{Colors.ENDC} {text}")

def print_info(text):
    """打印信息"""
    print(f"{Colors.BLUE}ℹ{Colors.ENDC} {text}")

def test_file_exists():
    """测试1：检查文件是否存在"""
    print_header("测试1: 文件存在性检查")
    
    files_to_check = [
        "pose_stable.py",
        "config.py",
        "models/yolo11m-pose.engine",
        "icon/red.png",
        "icon/green.png"
    ]
    
    all_exist = True
    for file in files_to_check:
        if os.path.exists(file):
            print_success(f"文件存在: {file}")
        else:
            print_error(f"文件缺失: {file}")
            all_exist = False
    
    return all_exist

def test_imports():
    """测试2：检查依赖包"""
    print_header("测试2: 依赖包检查")
    
    packages = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("ultralytics", "ultralytics"),
        ("requests", "requests"),
        ("openai", "openai"),
        ("mediapipe", "mediapipe"),
    ]
    
    all_imported = True
    for module, package in packages:
        try:
            __import__(module)
            print_success(f"模块可用: {module} ({package})")
        except ImportError:
            print_error(f"模块缺失: {module} ({package})")
            all_imported = False
    
    return all_imported

def test_camera():
    """测试3：检查摄像头"""
    print_header("测试3: 摄像头检查")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print_error("无法打开摄像头")
            return False
        
        print_success("摄像头打开成功")
        
        ret, frame = cap.read()
        if not ret:
            print_error("无法读取摄像头画面")
            cap.release()
            return False
        
        print_success(f"摄像头读取成功 (分辨率: {frame.shape[1]}x{frame.shape[0]})")
        
        cap.release()
        return True
        
    except Exception as e:
        print_error(f"摄像头测试失败: {e}")
        return False

def test_config():
    """测试4：检查配置文件"""
    print_header("测试4: 配置文件检查")
    
    try:
        import config
        
        required_vars = [
            "MODEL_PATH",
            "camera_index",
            "SEGMENTATION_THRESHOLD",
            "PNG_COMPRESSION_LEVEL",
            "AI_IMAGE_SIZE",
            "punch_state",
            "pose_duration",
            "pose_stable_threshold"
        ]
        
        all_present = True
        for var in required_vars:
            if hasattr(config, var):
                value = getattr(config, var)
                print_success(f"配置项存在: {var} = {value}")
            else:
                print_error(f"配置项缺失: {var}")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print_error(f"配置文件检查失败: {e}")
        return False

def test_code_fixes():
    """测试5：检查代码修复"""
    print_header("测试5: 代码修复检查")
    
    try:
        with open("pose_stable.py", "r", encoding="utf-8") as f:
            code = f.read()
        
        fixes = [
            ("API_TIMEOUT", "API超时设置"),
            ("DOWNLOAD_TIMEOUT", "下载超时设置"),
            ("MAX_API_RETRIES", "API重试机制"),
            ("CAMERA_RECONNECT_DELAY", "摄像头重连配置"),
            ("init_camera", "摄像头重连函数"),
            ("signal_handler", "信号处理函数"),
            ("program_running", "程序运行标志"),
            ("cleanup_old_images", "磁盘清理函数"),
            ("segmentation_model.close()", "MediaPipe资源释放"),
            ("timeout=", "网络请求超时"),
        ]
        
        all_fixed = True
        for fix, description in fixes:
            if fix in code:
                print_success(f"修复已应用: {description} ({fix})")
            else:
                print_error(f"修复缺失: {description} ({fix})")
                all_fixed = False
        
        return all_fixed
        
    except Exception as e:
        print_error(f"代码检查失败: {e}")
        return False

def test_syntax():
    """测试6：检查Python语法"""
    print_header("测试6: Python语法检查")
    
    try:
        result = subprocess.run(
            ["python3", "-m", "py_compile", "pose_stable.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print_success("Python语法检查通过")
            return True
        else:
            print_error(f"Python语法错误:\n{result.stderr}")
            return False
            
    except Exception as e:
        print_error(f"语法检查失败: {e}")
        return False

def test_disk_space():
    """测试7：检查磁盘空间"""
    print_header("测试7: 磁盘空间检查")
    
    try:
        stat = os.statvfs('.')
        free_bytes = stat.f_bavail * stat.f_frsize
        free_gb = free_bytes / (1024**3)
        
        if free_gb > 1:
            print_success(f"磁盘空间充足: {free_gb:.2f} GB 可用")
            return True
        else:
            print_warning(f"磁盘空间较少: {free_gb:.2f} GB 可用")
            return True
            
    except Exception as e:
        print_error(f"磁盘空间检查失败: {e}")
        return False

def test_permissions():
    """测试8：检查文件权限"""
    print_header("测试8: 文件权限检查")
    
    files_to_check = [
        "pose_stable.py",
        "config.py",
        "punch_photos"  # 目录
    ]
    
    all_ok = True
    for file in files_to_check:
        if not os.path.exists(file):
            print_warning(f"文件/目录不存在: {file}")
            continue
            
        if os.access(file, os.R_OK):
            print_success(f"可读: {file}")
        else:
            print_error(f"不可读: {file}")
            all_ok = False
        
        if os.path.isfile(file) and file.endswith('.py'):
            if os.access(file, os.W_OK):
                print_success(f"可写: {file}")
            else:
                print_warning(f"不可写: {file}")
    
    return all_ok

def test_photo_dir():
    """测试9：检查照片目录"""
    print_header("测试9: 照片目录检查")
    
    photo_dir = "punch_photos"
    
    if not os.path.exists(photo_dir):
        try:
            os.makedirs(photo_dir)
            print_success(f"创建照片目录: {photo_dir}")
        except Exception as e:
            print_error(f"无法创建照片目录: {e}")
            return False
    else:
        print_success(f"照片目录存在: {photo_dir}")
    
    # 检查目录中的文件数量
    try:
        files = [f for f in os.listdir(photo_dir) if f.startswith("punch_")]
        print_info(f"已有照片数量: {len(files)}")
        
        if len(files) > 100:
            print_warning(f"照片数量较多，建议清理旧照片")
    except Exception as e:
        print_error(f"无法读取照片目录: {e}")
        return False
    
    return True

def test_network():
    """测试10：检查网络连接"""
    print_header("测试10: 网络连接检查")
    
    try:
        import requests
        
        # 测试火山引擎API连通性
        url = "https://ark.cn-beijing.volces.com"
        
        print_info(f"测试连接: {url}")
        response = requests.get(url, timeout=5)
        
        if response.status_code < 500:
            print_success(f"网络连接正常 (状态码: {response.status_code})")
            return True
        else:
            print_warning(f"API响应异常 (状态码: {response.status_code})")
            return True
            
    except requests.exceptions.Timeout:
        print_error("网络连接超时")
        return False
    except requests.exceptions.ConnectionError:
        print_error("无法连接到API服务器")
        return False
    except Exception as e:
        print_error(f"网络测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}稳定性测试脚本 - pose_stable.py{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}")
    
    tests = [
        ("文件存在性", test_file_exists),
        ("依赖包", test_imports),
        ("摄像头", test_camera),
        ("配置文件", test_config),
        ("代码修复", test_code_fixes),
        ("Python语法", test_syntax),
        ("磁盘空间", test_disk_space),
        ("文件权限", test_permissions),
        ("照片目录", test_photo_dir),
        ("网络连接", test_network),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"测试 '{name}' 发生异常: {e}")
            results.append((name, False))
    
    # 打印总结
    print_header("测试总结")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        if result:
            print_success(f"{name}: 通过")
        else:
            print_error(f"{name}: 失败")
    
    print(f"\n{Colors.BOLD}总计: {passed}/{total} 测试通过{Colors.ENDC}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ 所有测试通过！系统已准备好24小时运行{Colors.ENDC}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ 有 {total - passed} 个测试失败，请修复后再运行{Colors.ENDC}")
        return 1

def print_usage():
    """打印使用说明"""
    print(f"""
{Colors.BOLD}使用说明:{Colors.ENDC}

1. 运行完整测试:
   python3 test_stability.py

2. 如果所有测试通过，可以启动程序:
   python3 pose_stable.py

3. 后台运行（推荐用于24小时运行）:
   nohup python3 pose_stable.py > punch.log 2>&1 &

4. 查看运行日志:
   tail -f punch.log

5. 停止程序:
   pkill -f pose_stable.py
""")

if __name__ == "__main__":
    try:
        exit_code = run_all_tests()
        print_usage()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}测试已取消{Colors.ENDC}")
        sys.exit(130)

