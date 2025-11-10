"""
打卡系统服务器通信示例脚本
演示如何：
1. 从NFC卡片获取用户信息
2. 下载用户头像
3. 上传火柴人图片到OSS
4. 更新用户的avatar_url
"""

import requests
import oss2
import os
import time
from datetime import datetime

# ========== OSS配置 ==========
OSS_ACCESS_KEY_ID = "LTAI5tApDmGBq95haRq3EHpL"
OSS_ACCESS_KEY_SECRET = "xW9eRDTf0DH7JcbXEn2B6funlnTaUt"
OSS_BUCKET_NAME = "sensecap-statics"
OSS_REGION = "cn-shenzhen"
OSS_ENDPOINT = f"https://oss-{OSS_REGION}.aliyuncs.com"
OSS_UPLOAD_PATH = "nfc-trace/makerfaire-2025/"  # OSS上传路径前缀

# ========== API配置 ==========
API_BASE_URL = "https://makerfaire-nfc.seeed.cn/api/v1"

# ========== 本地路径配置 ==========
PHOTOS_DIR = "./punch_photos"
AVATARS_DIR = "./avatars"


def get_user_info(user_uuid):
    """
    从服务器获取用户信息
    
    Args:
        user_uuid: 用户的UUID (从NFC卡片获取)
    
    Returns:
        dict: 用户信息，如果失败返回None
    """
    url = f"{API_BASE_URL}/cards/{user_uuid}"
    
    try:
        print(f"正在获取用户信息: {url}")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 0:
                user_data = data.get('data')
                print(f"✓ 用户信息获取成功: {user_data.get('nick_name')}")
                return user_data
            else:
                print(f"✗ API返回错误: {data.get('msg')}")
                return None
        else:
            print(f"✗ 请求失败，状态码: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"✗ 获取用户信息时发生错误: {e}")
        return None


def download_user_avatar(logo_url, user_uuid):
    """
    下载用户头像到本地
    
    Args:
        logo_url: 头像URL
        user_uuid: 用户UUID
    
    Returns:
        str: 本地头像路径，如果失败返回None
    """
    if not logo_url:
        print("⚠ 用户没有上传头像")
        return None
    
    try:
        print(f"正在下载用户头像: {logo_url}")
        
        # 创建头像保存目录
        if not os.path.exists(AVATARS_DIR):
            os.makedirs(AVATARS_DIR)
        
        # 下载头像
        response = requests.get(logo_url, timeout=10)
        
        if response.status_code == 200:
            # 获取文件扩展名
            ext = logo_url.split('.')[-1].split('?')[0]  # 处理URL参数
            if ext not in ['jpg', 'jpeg', 'png', 'gif']:
                ext = 'jpg'
            
            # 保存到本地
            avatar_path = os.path.join(AVATARS_DIR, f"{user_uuid}.{ext}")
            with open(avatar_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ 头像下载成功: {avatar_path}")
            return avatar_path
        else:
            print(f"✗ 头像下载失败，状态码: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"✗ 下载头像时发生错误: {e}")
        return None


def upload_to_oss(local_file_path, user_uuid):
    """
    上传火柴人图片到阿里云OSS
    
    Args:
        local_file_path: 本地文件路径
        user_uuid: 用户UUID
    
    Returns:
        str: OSS上的文件访问URL，如果失败返回None
    """
    try:
        print(f"正在上传文件到OSS: {local_file_path}")
        
        # 初始化OSS认证和Bucket对象
        auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
        bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)
        
        # 生成OSS上的文件名：{用户UUID}_{时间戳}.png
        timestamp = int(time.time())
        file_ext = os.path.splitext(local_file_path)[1]  # 获取文件扩展名
        oss_filename = f"{user_uuid}_{timestamp}{file_ext}"
        oss_key = OSS_UPLOAD_PATH + oss_filename
        
        # 上传文件
        result = bucket.put_object_from_file(oss_key, local_file_path)
        
        if result.status == 200:
            # 生成访问URL
            file_url = f"https://sensecap-statics.seeed.cn/{oss_key}"
            print(f"✓ 文件上传成功: {file_url}")
            return file_url
        else:
            print(f"✗ 文件上传失败，状态码: {result.status}")
            return None
            
    except Exception as e:
        print(f"✗ 上传到OSS时发生错误: {e}")
        return None


def update_user_avatar(user_uuid, avatar_url):
    """
    更新用户的打卡照片URL到服务器
    
    Args:
        user_uuid: 用户UUID
        avatar_url: 打卡照片的URL
    
    Returns:
        bool: 是否更新成功
    """
    url = f"{API_BASE_URL}/cards/{user_uuid}/avatar"
    
    try:
        print(f"正在更新用户avatar_url: {url}")
        
        data = {
            "avatar_url": avatar_url
        }
        
        response = requests.put(url, json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('code') == 0:
                print(f"✓ avatar_url更新成功")
                return True
            else:
                print(f"✗ API返回错误: {result.get('msg')}")
                return False
        else:
            print(f"✗ 请求失败，状态码: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ 更新avatar_url时发生错误: {e}")
        return False


def complete_punch_flow(user_uuid, stickman_image_path):
    """
    完整的打卡流程示例
    
    Args:
        user_uuid: 用户UUID（从NFC卡片读取）
        stickman_image_path: 本地生成的火柴人图片路径
    
    Returns:
        bool: 流程是否成功完成
    """
    print("\n" + "="*60)
    print(f"开始打卡流程 - 用户UUID: {user_uuid}")
    print("="*60 + "\n")
    
    # 步骤1: 获取用户信息
    print("【步骤1】获取用户信息")
    user_info = get_user_info(user_uuid)
    if not user_info:
        print("✗ 打卡流程失败：无法获取用户信息\n")
        return False
    
    print(f"  - 昵称: {user_info.get('nick_name')}")
    print(f"  - 位置: {user_info.get('location')}")
    print(f"  - 公司: {user_info.get('official_name')}")
    print()
    
    # 步骤2: 下载用户头像（可选）
    print("【步骤2】下载用户头像")
    logo_url = user_info.get('logo_url')
    avatar_path = download_user_avatar(logo_url, user_uuid)
    if avatar_path:
        print(f"  - 头像已保存到: {avatar_path}")
    print()
    
    # 步骤3: 上传火柴人图片到OSS
    print("【步骤3】上传火柴人图片到OSS")
    oss_url = upload_to_oss(stickman_image_path, user_uuid)
    if not oss_url:
        print("✗ 打卡流程失败：无法上传图片到OSS\n")
        return False
    print(f"  - OSS URL: {oss_url}")
    print()
    
    # 步骤4: 更新用户的avatar_url
    print("【步骤4】更新用户的avatar_url")
    success = update_user_avatar(user_uuid, oss_url)
    if not success:
        print("✗ 打卡流程失败：无法更新avatar_url\n")
        return False
    print()
    
    print("="*60)
    print("✓ 打卡流程全部完成！")
    print("="*60 + "\n")
    return True


# ========== 示例使用 ==========
if __name__ == "__main__":
    # 测试用例
    test_uuid = "5397d68f-6b54-4ec9-856f-560b07961fd9"
    test_stickman_image = "./punch_photos/stickman_20251107_043428.png"
   
    print("\n" + "="*60)
    print("打卡系统服务器通信示例")
    print("="*60 + "\n")
    
    # 如果你想测试完整流程，需要先准备一个测试图片
    if os.path.exists(test_stickman_image):
        # 执行完整打卡流程
        complete_punch_flow(test_uuid, test_stickman_image)
    else:
        print("⚠ 提示：测试图片不存在，仅演示单个API调用\n")
        
        # 演示单个API调用
        print("【演示1】获取用户信息")
        user_info = get_user_info(test_uuid)
        print()
        
        if user_info:
            print("【演示2】下载用户头像")
            logo_url = user_info.get('logo_url')
            avatar_path = download_user_avatar(logo_url, test_uuid)
            print()
        
        print("\n提示：要测试完整流程，请先创建测试图片：")
        print(f"  mkdir -p {PHOTOS_DIR}")
        print(f"  # 创建一个测试图片到: {test_stickman_image}")

