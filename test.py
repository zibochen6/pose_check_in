import os
import requests
from datetime import datetime
from openai import OpenAI
import base64

# 读取本地图片文件并转换为base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 初始化OpenAI客户端（兼容方舟API）
client = OpenAI( 
    base_url="https://ark.cn-beijing.volces.com/api/v3", 
    api_key="0b02eee6-5201-46c3-95a8-59594aa6dc38",  # 直接使用API密钥或从环境变量读取
) 

# 生成图片
imagesResponse = client.images.generate( 
    model="doubao-seedream-4-0-250828", 
    prompt="保留画面主体的姿态，将图片风格转化为动漫风格，但不要改变人物的五官神态，要能在动漫风格的图片看的出来主角真人的特点和神态！",
    size="2K",
    response_format="url",
    extra_body = {
        "image": f"data:image/jpeg;base64,{encode_image('./test.jpg')}",
        "watermark": True
    }
) 

# 创建保存图片的目录
save_dir = "check_in_perpole"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 下载并保存图片
print(f"图片URL: {imagesResponse.data[0].url}")

try:
    # 下载图片
    response = requests.get(imagesResponse.data[0].url)
    response.raise_for_status()
    
    # 生成文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_image_{timestamp}.jpg"
    filepath = os.path.join(save_dir, filename)
    
    # 保存图片到本地
    with open(filepath, 'wb') as f:
        f.write(response.content)
    
    print(f"✓ 图片已保存到: {filepath}")
    
except requests.exceptions.RequestException as e:
    print(f"✗ 下载图片失败: {e}")
except Exception as e:
    print(f"✗ 保存图片时出错: {e}")