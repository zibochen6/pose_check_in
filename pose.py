import os
# OpenCV 使用 Qt5 后端（构建时只支持 Qt，不支持 GTK）
# 需要正确配置 Qt 插件路径，确保 Qt 能找到 xcb 插件
# 必须在导入 cv2 之前设置环境变量

# 恢复 Qt 插件目录（如果之前被禁用了）
qt_plugin_path = os.path.expanduser('~/.local/lib/python3.10/site-packages/cv2/qt/plugins')
qt_plugin_backup = qt_plugin_path + '.disabled'
if os.path.exists(qt_plugin_backup) and not os.path.exists(qt_plugin_path):
    try:
        os.rename(qt_plugin_backup, qt_plugin_path)
        print(f"已恢复 Qt 插件目录")
    except Exception as e:
        print(f"警告：无法恢复 Qt 插件目录: {e}")

# 设置 Qt 插件路径，确保能找到 xcb 插件
# OpenCV 的 Qt 插件目录
if os.path.exists(qt_plugin_path):
    # 设置 QT_PLUGIN_PATH 包含 OpenCV 的插件目录和系统 Qt 插件目录
    system_qt_plugins = '/usr/lib/aarch64-linux-gnu/qt5/plugins'
    qt_plugin_paths = [qt_plugin_path, system_qt_plugins]
    os.environ['QT_PLUGIN_PATH'] = ':'.join(qt_plugin_paths)
    # print(f"已设置 Qt 插件路径: {os.environ['QT_PLUGIN_PATH']}")

# 设置优先使用 V4L2（Linux 摄像头后端）
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_DSHOW'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_V4L2'] = '1'

# 确保 DISPLAY 环境变量正确设置
if 'DISPLAY' not in os.environ or not os.environ.get('DISPLAY'):
    os.environ['DISPLAY'] = ':0'

import cv2
import time
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from config import *
import requests
import base64
from openai import OpenAI
import threading
from queue import Queue
import mediapipe as mp

# GUI 支持
HAVE_GUI = True

# OpenCV 使用 Qt5 后端（已正确配置 Qt 插件路径）
print("OpenCV 使用 Qt5 后端运行")

# 加载YOLO姿态估计模型
model = YOLO(MODEL_PATH,task="pose")  # 使用TensorRT引擎文件

# 初始化 MediaPipe Selfie Segmentation（实时背景去除）
print(f"\n{'='*50}")
print(f"正在加载背景去除模型: MediaPipe Selfie Segmentation...")
model_load_start = time.time()
mp_selfie_segmentation = mp.solutions.selfie_segmentation
# model_selection: 0=general (快速), 1=landscape (精确但稍慢)
segmentation_model = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
model_load_time = time.time() - model_load_start
print(f"✓ 背景去除模型加载完成，耗时: {model_load_time:.2f}秒")
print(f"✓ 模型类型: MediaPipe (实时优化)")
print(f"{'='*50}\n")

# 初始化OpenAI客户端（用于图片风格化）
openai_client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="0b02eee6-5201-46c3-95a8-59594aa6dc38",
)

def encode_image_to_base64(image):
    """将OpenCV图像转换为base64字符串"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# 加载状态图标
red_icon = cv2.imread("icon/red.png", cv2.IMREAD_UNCHANGED)
green_icon = cv2.imread("icon/green.png", cv2.IMREAD_UNCHANGED)

# 确保图标加载成功
if red_icon is None or green_icon is None:
    print("警告：无法加载图标文件！")
    red_icon = np.zeros((100, 100, 4), dtype=np.uint8)
    green_icon = np.zeros((100, 100, 4), dtype=np.uint8)

# 初始化摄像头
cap = cv2.VideoCapture(camera_index)

# 设置MJPG格式（Motion-JPEG, compressed）
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# 可选分辨率：
# 1920x1080 (推荐，高分辨率)
# 1280x960
# 1280x720
# 设置摄像头分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 获取实际摄像头分辨率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"摄像头分辨率: {frame_width}x{frame_height}")

# 创建照片保存目录
photos_dir = "punch_photos"
if not os.path.exists(photos_dir):
    os.makedirs(photos_dir)

# 创建全屏窗口
window_name = "AI Punch Clock System"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 获取屏幕分辨率（通过创建一个临时全屏窗口来获取）
screen_width = 1920  # 默认值
screen_height = 1080  # 默认值
try:
    # 尝试获取实际屏幕分辨率
    import subprocess
    result = subprocess.run(['xrandr'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if '*' in line:  # 当前使用的分辨率会有*标记
            parts = line.split()
            for part in parts:
                if 'x' in part and part[0].isdigit():
                    screen_width, screen_height = map(int, part.split('x'))
                    break
            break
except:
    print("无法自动获取屏幕分辨率，使用默认值1920x1080")

print(f"屏幕分辨率: {screen_width}x{screen_height}")

# FPS计算变量
p_time = time.time()

# 异步推理队列
inference_queue = Queue(maxsize=1)
result_queue = Queue(maxsize=1)

# 最后推理结果
last_results = None
results_lock = threading.Lock()

# UI显示相关的全局变量
display_state = "camera"  # "camera" 或 "result"
generated_image = None  # 生成的动漫图片
is_generating = False  # 是否正在生成
generation_angle = 0  # 旋转角度
generation_lock = threading.Lock()

def create_blur_with_spinner(frame, angle, text="Generating your avatar, please wait..."):
    """创建模糊背景+旋转加载动画"""
    # 高斯模糊
    blurred = cv2.GaussianBlur(frame, (51, 51), 0)
    
    # 添加半透明遮罩
    overlay = blurred.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    blurred = cv2.addWeighted(blurred, 0.7, overlay, 0.3, 0)
    
    # 绘制旋转圆圈
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
    radius = 60
    thickness = 8
    
    # 绘制圆环
    cv2.circle(blurred, (center_x, center_y), radius, (100, 100, 100), thickness)
    
    # 绘制旋转的弧
    arc_length = 90  # 弧长（度）
    start_angle = int(angle) % 360
    cv2.ellipse(blurred, (center_x, center_y), (radius, radius), 
                0, start_angle, start_angle + arc_length, (0, 255, 255), thickness)
    
    # 添加文字提示
    lines = text.split('\n')
    y_offset = center_y + radius + 60
    for line in lines:
        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = center_x - text_size[0] // 2
        cv2.putText(blurred, line, (text_x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 40
    
    return blurred

def create_side_by_side_view(left_img, right_img, gap=40):
    """创建左右拼接视图（美化版）"""
    # 调整右图大小以匹配左图高度
    h1, w1 = left_img.shape[:2]
    h2, w2 = right_img.shape[:2]
    
    # 调整右图高度
    scale = h1 / h2
    new_w2 = int(w2 * scale)
    right_resized = cv2.resize(right_img, (new_w2, h1))
    
    # 创建拼接画布（深色背景）
    total_width = w1 + gap + new_w2
    canvas = np.zeros((h1, total_width, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)  # 深灰色背景
    
    # 添加边框和阴影效果到左图
    border = 5
    shadow_offset = 8
    # 阴影
    cv2.rectangle(canvas, (shadow_offset, shadow_offset), 
                 (w1 + shadow_offset, h1 + shadow_offset), (0, 0, 0), -1)
    # 白色边框
    cv2.rectangle(canvas, (0, 0), (w1, h1), (255, 255, 255), border)
    # 放置左图
    canvas[border:h1-border, border:w1-border] = left_img[border:h1-border, border:w1-border]
    
    # 绘制渐变分隔线
    sep_x = w1 + gap // 2
    for i in range(h1):
        alpha = i / h1  # 渐变效果
        color = (int(50 + 100 * alpha), int(50 + 100 * alpha), int(50 + 100 * alpha))
        cv2.line(canvas, (sep_x, i), (sep_x, i), color, 3)
    
    # 添加边框和阴影到右图
    right_x = w1 + gap
    # 阴影
    cv2.rectangle(canvas, (right_x + shadow_offset, shadow_offset), 
                 (right_x + new_w2 + shadow_offset, h1 + shadow_offset), (0, 0, 0), -1)
    # 白色边框
    cv2.rectangle(canvas, (right_x, 0), (right_x + new_w2, h1), (255, 255, 255), border)
    # 放置右图
    canvas[border:h1-border, right_x+border:right_x+new_w2-border] = right_resized[border:h1-border, border:new_w2-border]
    
    # 添加标题栏（带渐变背景）
    header_h = 60
    header = np.zeros((header_h, total_width, 3), dtype=np.uint8)
    for i in range(header_h):
        alpha = 1 - (i / header_h)
        color = int(80 * alpha)
        header[i, :] = (color, color, color)
    
    # 合并标题栏
    result = np.vstack([header, canvas])
    
    # 添加标题文字
    cv2.putText(result, "Original (No Background)", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, "AI Generated Style", (right_x + 20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 255), 2, cv2.LINE_AA)
    
    return result

def scale_to_fullscreen(image, target_width, target_height):
    """将图像缩放以填充整个屏幕，不留黑边
    
    方案：直接将图像拉伸到目标尺寸，显示完整画面，无裁剪
    
    Args:
        image: 输入图像
        target_width: 目标宽度（屏幕宽度）
        target_height: 目标高度（屏幕高度）
    
    Returns:
        缩放后的图像
    """
    # 直接将图像拉伸到目标尺寸，不裁剪任何内容
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return resized

def inference_worker():
    """后台推理线程"""
    global last_results
    while True:
        frame = inference_queue.get()
        if frame is None:  # 停止信号
            break
        
        try:
            # 进行姿态估计推理（降低推理分辨率以提高速度）
            results = model(frame, verbose=False, imgsz=640)
            
            # 更新结果（线程安全）
            with results_lock:
                last_results = results
        except Exception as e:
            print(f"推理错误: {e}")

# 启动推理线程
inference_thread = threading.Thread(target=inference_worker, daemon=True)
inference_thread.start()


def overlay_icon_with_alpha(background, icon, x, y, alpha=1.0):
    """在背景图像上叠加带透明度的图标"""
    h, w = icon.shape[:2]
    
    # 确保坐标在范围内
    if y + h > background.shape[0] or x + w > background.shape[1]:
        return background
    if x < 0 or y < 0:
        return background
    
    # 提取RGB和Alpha通道
    if icon.shape[2] == 4:
        icon_rgb = icon[:, :, :3]
        icon_alpha = icon[:, :, 3] / 255.0 * alpha  # 应用额外的透明度
    else:
        icon_rgb = icon
        icon_alpha = np.ones((h, w)) * alpha
    
    # 获取背景区域
    bg_region = background[y:y+h, x:x+w]
    
    # Alpha混合
    for c in range(3):
        bg_region[:, :, c] = (icon_alpha * icon_rgb[:, :, c] + 
                              (1 - icon_alpha) * bg_region[:, :, c])
    
    background[y:y+h, x:x+w] = bg_region
    return background

def blend_icons(red_icon, green_icon, progress):
    """根据进度混合红绿图标
    progress: 0.0 (完全红色) -> 1.0 (完全绿色)
    """
    # 创建一个空白画布
    blended = np.zeros_like(red_icon)
    
    # 红色透明度：从1.0到0.0
    red_alpha = 1.0 - progress
    # 绿色透明度：从0.0到1.0
    green_alpha = progress
    
    # 混合两个图标
    if red_icon.shape[2] == 4:
        # 处理RGBA
        for c in range(3):
            blended[:, :, c] = (red_alpha * red_icon[:, :, c] + 
                               green_alpha * green_icon[:, :, c])
        # Alpha通道也混合
        blended[:, :, 3] = np.maximum(red_icon[:, :, 3], green_icon[:, :, 3])
    else:
        # 处理RGB
        blended = (red_alpha * red_icon + green_alpha * green_icon).astype(np.uint8)
    
    return blended.astype(np.uint8)

def is_person_in_roi(keypoints, roi_x, roi_y, roi_width, roi_height):
    """检查人是否在检测区域内"""
    if len(keypoints) == 0:
        return False
    
    # 检查关键点是否在ROI内
    valid_keypoints = 0
    for x, y, conf in keypoints:
        if conf > 0.6:
            if roi_x <= x <= roi_x + roi_width and roi_y <= y <= roi_y + roi_height:
                valid_keypoints += 1
    
    # 如果超过一半的关键点在ROI内，认为人在区域内
    return valid_keypoints > len([k for k in keypoints if k[2] > 0.5]) * 0.5

def calculate_pose_distance(keypoints1, keypoints2):
    """计算两个姿态之间的距离（仅使用身体关键点，提高检测流畅性）"""
    if keypoints1 is None or keypoints2 is None:
        return float('inf')
    
    total_distance = 0
    valid_points = 0
    
    # 身体关键点距离计算
    important_keypoints = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # 鼻子、左肩、右肩、左肘、右肘、左腕、右腕、左髋、右髋、左膝、右膝、左踝、右踝
    
    for i in important_keypoints:
        if (i < len(keypoints1) and i < len(keypoints2) and 
            keypoints1[i][2] > 0.5 and keypoints2[i][2] > 0.5):
            distance = np.sqrt((keypoints1[i][0] - keypoints2[i][0])**2 + 
                             (keypoints1[i][1] - keypoints2[i][1])**2)
            total_distance += distance
            valid_points += 1
    
    return total_distance / valid_points if valid_points > 0 else float('inf')


def get_person_bounding_box_from_detection(results):
    """从YOLO检测结果中直接获取人的边界框"""
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return None
    
    # 获取第一个检测到的人的边界框
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 格式: [x1, y1, x2, y2]
    
    if len(boxes) == 0:
        return None
    
    # 取第一个人的边界框
    x1, y1, x2, y2 = boxes[0]
    
    # 添加一些边距
    margin = 20
    x_min = max(0, int(x1 - margin))
    y_min = max(0, int(y1 - margin))
    x_max = min(frame_width, int(x2 + margin))
    y_max = min(frame_height, int(y2 + margin))
    
    return (x_min, y_min, x_max, y_max)

# detect_hands 函数已移除，不再使用手部检测以提高系统流畅性

def generate_anime_style_image(person_image_base64):
    """调用AI生成动漫风格图片（无水印）"""
    try:
        # 优化5: 简化prompt以加快API响应速度
        images_response = openai_client.images.generate(
            model="doubao-seedream-4-0-250828",
            prompt="Transform to Disney animation style. Preserve exact pose, body position, and proportions. Only change art style, colors, and lighting.",
            size=AI_IMAGE_SIZE,  # 使用配置文件中的设置
            response_format="url",
            extra_body={
                "image": f"data:image/png;base64,{person_image_base64}",
                "watermark": False,
                "negative_prompt": "different pose, changed body position, altered proportions"
            }
        )
        return images_response.data[0].url
    except Exception as e:
        print(f"生成风格图片失败: {e}")
        return None

def generate_anime_image_async(frame, person_bbox):
    """后台线程生成动漫风格图片"""
    global generated_image, bg_removed_image, is_generating, display_state
    
    # 记录开始时间
    start_time = time.time()
    
    # 初始化性能监控变量
    bg_removal_time = 0
    api_time = 0
    
    if person_bbox is None:
        print("无法获取人的检测框")
        return
    
    x_min, y_min, x_max, y_max = person_bbox
    
    # 裁剪人的区域（裁切目标框）
    cropped_person = frame[y_min:y_max, x_min:x_max]
    
    if cropped_person.size == 0:
        print("检测框区域无效")
        return
    
    # 生成时间戳用于文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: MediaPipe 背景去除（实时高效）
    print("\n" + "="*50)
    print(f"📸 背景去除开始 (模型: MediaPipe Selfie Segmentation)")
    bg_removal_start = time.time()
    
    # MediaPipe 处理
    h_orig, w_orig = cropped_person.shape[:2]
    print(f"  ├─ 图像尺寸: {w_orig}x{h_orig}")
    
    # 转换为 RGB (MediaPipe 需要 RGB)
    print(f"  ├─ 正在抠图...")
    segmentation_start = time.time()
    rgb_frame = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
    
    # 执行分割
    results = segmentation_model.process(rgb_frame)
    mask = results.segmentation_mask
    
    segmentation_time = time.time() - segmentation_start
    print(f"  ├─ 抠图完成: {segmentation_time:.2f}秒")
    
    # 创建 BGRA 图像（原始颜色 + 透明通道）
    # 提高阈值以获得更彻底的背景去除
    # mask 是 0.0-1.0 的浮点数，使用阈值处理
    threshold = SEGMENTATION_THRESHOLD  # 使用配置文件中的阈值
    mask_threshold = (mask > threshold).astype(np.uint8) * 255
    
    # 可选：应用形态学操作以平滑边缘
    kernel = np.ones((3, 3), np.uint8)
    mask_threshold = cv2.morphologyEx(mask_threshold, cv2.MORPH_CLOSE, kernel)
    mask_threshold = cv2.morphologyEx(mask_threshold, cv2.MORPH_OPEN, kernel)
    
    # 创建 BGRA 图像
    bg_removed = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2BGRA)
    bg_removed[:, :, 3] = mask_threshold  # 设置 alpha 通道
    
    bg_removal_time = time.time() - bg_removal_start
    print(f"  └─ 背景去除总耗时: {bg_removal_time:.2f}秒")
    print("="*50)
    
    # 统计透明度
    non_zero = np.count_nonzero(mask_threshold < 255)
    total = mask_threshold.size
    transparent_percent = (non_zero / total) * 100
    print(f"  ├─ 背景透明度: {transparent_percent:.1f}% 像素已透明化")
    print(f"  ├─ 抠图阈值: {threshold} (越高背景去除越彻底)")
    
    # 保存调试图片（如果启用）
    if SAVE_DEBUG_IMAGES:
        debug_path = os.path.join(photos_dir, f"debug_nobg_{timestamp}.png")
        cv2.imwrite(debug_path, bg_removed)
        print(f"  ├─ 调试图片已保存: {debug_path}")
    
    # PNG 编码
    encode_param = [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION_LEVEL]
    _, buffer = cv2.imencode('.png', bg_removed, encode_param)
    person_image_base64 = base64.b64encode(buffer).decode('utf-8')
    print(f"  └─ 图像编码完成，大小: {len(person_image_base64)/1024:.1f}KB")
    
    # 调用AI生成动漫风格图片
    print("正在生成风格图片...")
    api_start = time.time()
    anime_image_url = generate_anime_style_image(person_image_base64)
    api_time = time.time() - api_start
    print(f"API调用耗时: {api_time:.2f}秒")
    
    if anime_image_url:
        print(f"风格图片URL: {anime_image_url}")
        
        # 下载并保存生成的动漫风格图片
        try:
            response = requests.get(anime_image_url)
            response.raise_for_status()
            
            anime_filename = f"punch_{timestamp}.jpg"
            anime_filepath = os.path.join(photos_dir, anime_filename)
            
            with open(anime_filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"风格图片已保存: {anime_filepath}")
            
            # 读取生成的图片并更新全局变量
            anime_image = cv2.imread(anime_filepath)
            if anime_image is not None:
                with generation_lock:
                    generated_image = anime_image.copy()
                    is_generating = False
                    display_state = "result"
                print(f"✓ 图片生成完成！")
            
        except requests.exceptions.RequestException as e:
            print(f"下载风格图片失败: {e}")
        except Exception as e:
            print(f"保存风格图片时出错: {e}")
    
    # 计算并打印总耗时
    end_time = time.time()
    total_time = end_time - start_time
    
    # 性能汇总
    print("\n" + "="*50)
    print("📊 性能汇总:")
    print(f"  ├─ 背景去除: {bg_removal_time:.2f}秒")
    print(f"  ├─ API调用:  {api_time:.2f}秒")
    print(f"  └─ 总耗时:   {total_time:.2f}秒")
    print("="*50)
    print(f"✓ 打卡完成！\n")

print("姿态打卡系统初始化...")
print("请进入检测区域并保持pose 3秒进行打卡")
print("按 'q' 键退出程序")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("无法读取摄像头画面")
        break
    
    # 计算FPS
    c_time = time.time()
    fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
    p_time = c_time
    
    
    
    # 异步推理（非阻塞）- 直接使用原始帧
    if not inference_queue.full():
        try:
            inference_queue.put_nowait(frame)
        except:
            pass
    
    # 获取最新的推理结果（线程安全）
    with results_lock:
        results = last_results
    
    # 创建显示帧（带可视化）- 确保每帧都显示流畅的画面
    display_frame = frame.copy()
    
    # 绘制ROI区域
    cv2.rectangle(display_frame, (roi_x, roi_y), 
                  (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 1)
    
    # 绘制YOLO姿态检测结果（根据配置决定是否显示）
    if results is not None and show_detection_results and results[0].keypoints is not None:
        pose_keypoints = results[0].keypoints.data.cpu().numpy()
        
        for person_kp in pose_keypoints:
            # 绘制骨架连接线
            for connection in skeleton:
                pt1_idx, pt2_idx = connection
                if (pt1_idx < len(person_kp) and pt2_idx < len(person_kp) and
                    person_kp[pt1_idx][2] > 0.5 and person_kp[pt2_idx][2] > 0.5):
                    pt1 = (int(person_kp[pt1_idx][0]), int(person_kp[pt1_idx][1]))
                    pt2 = (int(person_kp[pt2_idx][0]), int(person_kp[pt2_idx][1]))
                    cv2.line(display_frame, pt1, pt2, (255, 0, 255), 2)  # 紫色骨架
            
            # 绘制关键点
            for i, (x, y, conf) in enumerate(person_kp):
                if conf > 0.5:
                    cv2.circle(display_frame, (int(x), int(y)), 5, (0, 0, 255), -1)  # 红色关键点
    
    # 获取关键点数据
    current_keypoints = None
    if results is not None and results[0].keypoints is not None:
        keypoints = results[0].keypoints.data.cpu().numpy()
        num_people = len(keypoints)
        
        # 检查是否只有一个人
        if num_people == 1:
            person_keypoints = keypoints[0]
            current_keypoints = person_keypoints
            
            # 检查人是否在ROI内
            if is_person_in_roi(person_keypoints, roi_x, roi_y, roi_width, roi_height):
                
                #还没人进入打卡区域
                if punch_state == "waiting":
                    punch_state = "detecting"
                    # print("检测到人员进入打卡区域")
                #已经有人在打卡区域
                elif punch_state == "detecting":
                    punch_state = "posing"
                    pose_start_time = time.time()
                    last_pose_keypoints = person_keypoints.copy()
                    # print("开始检测pose，请保持姿态3秒...")

                #已经摆pose了，开始检测pose是否稳定
                elif punch_state == "posing":
                    # 检查姿态是否稳定（仅使用身体关键点）
                    pose_distance = calculate_pose_distance(person_keypoints, last_pose_keypoints)
                    current_time = time.time()
                    elapsed_time = current_time - pose_start_time
                    
                    if pose_distance < pose_stable_threshold:
                        # 姿态稳定，更新计时
                        remaining_time = pose_duration - elapsed_time
                        
                        if remaining_time <= 0:
                            # 时间到，拍照并启动后台生成
                            punch_state = "generating"
                            person_bbox = get_person_bounding_box_from_detection(results)
                            
                            # 标记开始生成
                            with generation_lock:
                                is_generating = True
                                display_state = "camera"
                                generated_image = None
                            
                            # 在后台线程生成图片
                            generation_thread = threading.Thread(
                                target=generate_anime_image_async, 
                                args=(frame.copy(), person_bbox),
                                daemon=True
                            )
                            generation_thread.start()
                            print("开始生成风格图片...")
                        else:
                            # 显示倒计时
                            cv2.putText(display_frame, f"Hold pose: {remaining_time:.1f}s", 
                                      (200, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            # 更新参考姿态（允许轻微移动）
                            last_pose_keypoints = person_keypoints.copy()
                    else:
                        # 姿态不稳定，重新开始
                        punch_state = "detecting"
                        pose_start_time = None
                        print(f"姿态不稳定，距离: {pose_distance:.1f}，阈值: {pose_stable_threshold}")
                
                elif punch_state == "generating":
                    # 正在生成，保持状态直到生成完成
                    with generation_lock:
                        if display_state == "result":
                            # 生成完成，切换到success状态
                            punch_state = "success"
                
                elif punch_state == "success":
                    # 成功状态，等待人离开重置
                    pass  # 显示由display_state控制
            else:
                # 人不在ROI内，重置状态
                if punch_state == "success" or punch_state == "generating":
                    # 从成功状态重置
                    punch_state = "waiting"
                    pose_start_time = None
                    with generation_lock:
                        display_state = "camera"
                        generated_image = None
                        is_generating = False
                    print("System reset, ready for next punch-in")
                    print(f"Debug: punch_state={punch_state}, display_state={display_state}, generated_image={'None' if generated_image is None else 'Exists'}")
                elif punch_state != "waiting":
                    punch_state = "waiting"
                    pose_start_time = None
                    # 确保也重置显示状态
                    with generation_lock:
                        display_state = "camera"
                        generated_image = None
                        is_generating = False
                    print("Please enter detection area")
        
        elif num_people > 1:
            # 多个人的情况
            if punch_state not in ["success", "generating"]:
                # 其他状态下多个人，重置状态
                punch_state = "waiting"
                pose_start_time = None
                # 确保也重置显示状态
                with generation_lock:
                    display_state = "camera"
                    generated_image = None
                    is_generating = False
                print("Too many people detected! Please ensure only one person in the area.")
        
        else:
            # 没有人，重置状态
            if punch_state != "waiting":
                punch_state = "waiting"
                pose_start_time = None
                # 确保也重置显示状态
                with generation_lock:
                    display_state = "camera"
                    generated_image = None
                    is_generating = False
                print("区域内无人，系统已重置")
                print(f"Debug: punch_state={punch_state}, display_state={display_state}, generated_image={'None' if generated_image is None else 'Exists'}")
    
    # 根据状态决定显示什么
    with generation_lock:
        current_display_state = display_state
        current_is_generating = is_generating
        current_generated = generated_image
    
    # 最终显示的画面
    final_display = None
    
    if punch_state == "generating" and current_is_generating:
        # 生成中：显示模糊+旋转动画
        generation_angle = (generation_angle + 10) % 360  # 旋转角度
        final_display = create_blur_with_spinner(
            display_frame, 
            generation_angle,
            "Generating your avatar\nPlease wait..."
        )
    elif current_display_state == "result" and current_generated is not None and punch_state == "success":
        # 显示结果：只显示生成的动漫图片（单窗口）
        # 使用屏幕分辨率
        target_width = screen_width
        target_height = screen_height
        
        # 标题和底部的高度
        header_h = 60
        footer_h = 80
        
        # 可用于显示图片的高度
        image_height = target_height - header_h - footer_h
        
        # 调整生成图片的大小以适应窗口
        img = current_generated.copy()
        h, w = img.shape[:2]
        
        # 按比例缩放，使高度刚好fit
        scale_h = image_height / h
        scale_w = target_width / w
        scale = min(scale_h, scale_w)  # 使用较小的比例，确保不超出
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # 创建居中的图片区域（黑色背景）
        img_canvas = np.zeros((image_height, target_width, 3), dtype=np.uint8)
        # 居中放置图片
        x_offset = (target_width - new_w) // 2
        y_offset = (image_height - new_h) // 2
        img_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
        
        # 添加标题栏
        header = np.zeros((header_h, target_width, 3), dtype=np.uint8)
        # 渐变背景
        for i in range(header_h):
            alpha = 1 - (i / header_h)
            color = int(80 * alpha)
            header[i, :] = (color, color + 40, color + 60)
        
        # 添加标题文字（缩小字体以适应窗口）
        title = "Success!"
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (target_width - text_size[0]) // 2
        cv2.putText(header, title, (text_x, 38), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        # 添加底部提示框
        footer = np.zeros((footer_h, target_width, 3), dtype=np.uint8)
        # 渐变背景
        for i in range(footer_h):
            alpha = i / footer_h
            color = int(50 * alpha)
            footer[i, :] = (color, color, color)
        
        # 添加提示文字（放大并居中）
        text1 = "Complete! Leave area to reset"
        
        font_scale = 1.2  # 放大字体
        thickness = 3     # 加粗
        text_size1 = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x1 = (target_width - text_size1[0]) // 2  # 居中
        text_y1 = footer_h // 2 + text_size1[1] // 2   # 垂直居中
        cv2.putText(footer, text1, (text_x1, text_y1), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        
        # 合并标题、图片和底部提示 (确保总大小为640x480)
        final_display = np.vstack([header, img_canvas, footer])
    else:
        # 正常状态：显示摄像头画面
        final_display = display_frame.copy()
        
        # 显示状态信息
        status_text = {
            "waiting": "Waiting for person...",
            "detecting": "Person detected, get ready!",
            "posing": "Hold your pose!",
            "capturing": "Capturing...",
            "generating": "Generating...",
            "success": "Success!"
        }
        
        # 显示状态
        cv2.putText(final_display, f"Status: {status_text.get(punch_state, 'Unknown')}", 
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        # # 显示FPS（左上角，在状态信息下方）
        # cv2.putText(final_display, f"FPS: {fps:.1f}", 
        #           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
        # 在摄像头画面上显示状态图标
        # 只在 posing 状态（用户保持姿势）时显示，避免其他状态的干扰
        icon_size = 80
        
        # 仅在 posing 状态且有倒计时时显示图标
        if punch_state == "posing" and pose_start_time is not None:
            # 图标显示在屏幕中央，"Hold pose"文字上方
            icon_x = (final_display.shape[1] - icon_size) // 2  # 水平居中
            icon_y = 320  # "Hold pose"文字在430，图标显示在上方
            
            # 计算进度和图标颜色（从红到绿的渐变）
            elapsed = time.time() - pose_start_time
            progress = min(elapsed / pose_duration, 1.0)
            current_icon = blend_icons(red_icon, green_icon, progress)
            
            # 叠加图标
            resized_icon = cv2.resize(current_icon, (icon_size, icon_size))
            final_display = overlay_icon_with_alpha(final_display, resized_icon, icon_x, icon_y, alpha=0.9)
    
    # 显示最终画面（缩放到全屏）
    if final_display is not None:
        # 将画面缩放并裁剪以填充整个屏幕
        fullscreen_display = scale_to_fullscreen(final_display, screen_width, screen_height)
        cv2.imshow(window_name, fullscreen_display)
    
    # 按'q'键退出
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    # 确保适当的帧率
    time.sleep(0.033)  # 约30 FPS

# 释放资源
# 通知推理线程停止
inference_queue.put(None)
cap.release()
cv2.destroyAllWindows()

# Qt 插件目录已正确配置，无需恢复

print("程序结束")