import cv2
import time
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from config import *

# 加载YOLO姿态估计模型
model = YOLO(MODEL_PATH,task="pose")  # 使用TensorRT引擎文件

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

# 设置摄像头分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 创建照片保存目录
photos_dir = "punch_photos"
if not os.path.exists(photos_dir):
    os.makedirs(photos_dir)

# FPS计算变量
p_time = time.time()


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

def calculate_pose_distance(keypoints1, keypoints2):
    """计算两个姿态之间的距离"""
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

def enhance_frame_for_dark_lighting(frame):
    """优化暗光环境下的画面质量"""
    # 使用CLAHE (Contrast Limited Adaptive Histogram Equalization) 增强对比度
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 对L通道应用CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # 合并通道并转换回BGR
    enhanced_lab = cv2.merge([l, a, b])
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 轻微降噪
    enhanced_frame = cv2.bilateralFilter(enhanced_frame, 9, 75, 75)
    
    return enhanced_frame

def draw_stickman(keypoints, canvas_width=500, canvas_height=700):
    """根据关键点在透明画布上绘制美观的火柴人"""
    # 先创建白色画布（RGB格式）用于绘制
    canvas_rgb = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # 找到关键点的边界来进行缩放和居中
    valid_points = []
    for x, y, conf in keypoints:
        if conf > 0.3:  # 降低阈值以获取更多关键点
            valid_points.append((x, y))
    
    if len(valid_points) == 0:
        # 返回透明画布
        canvas_rgba = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
        return canvas_rgba
    
    # 计算关键点的边界
    x_coords = [p[0] for p in valid_points]
    y_coords = [p[1] for p in valid_points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # 计算缩放比例，保持纵横比
    person_width = x_max - x_min
    person_height = y_max - y_min
    
    scale_x = (canvas_width * 0.7) / person_width if person_width > 0 else 1
    scale_y = (canvas_height * 0.7) / person_height if person_height > 0 else 1
    scale = min(scale_x, scale_y)
    
    # 计算偏移，使火柴人居中
    offset_x = (canvas_width - person_width * scale) / 2 - x_min * scale
    offset_y = (canvas_height - person_height * scale) / 2 - y_min * scale
    
    # 转换关键点坐标到画布坐标
    def transform_point(x, y):
        new_x = int(x * scale + offset_x)
        new_y = int(y * scale + offset_y)
        return (new_x, new_y)
    
    # 定义身体各部分的线条粗细和颜色（RGB格式）
    body_color = (50, 50, 50)  # 深灰色
    joint_color = (100, 150, 255)  # 浅蓝色关节
    head_color = (255, 200, 150)  # 肤色头部
    
    # 1. 首先计算头部位置和大小
    head_center = None
    head_radius = 20  # 默认头部半径
    neck_bottom = None  # 脖子底部位置（肩膀中点）
    
    # 计算肩膀位置
    if keypoints[5][2] > 0.2 and keypoints[6][2] > 0.2:
        shoulder_dist = np.sqrt((keypoints[5][0] - keypoints[6][0])**2 + 
                               (keypoints[5][1] - keypoints[6][1])**2)
        head_radius = int(shoulder_dist * scale * 0.4)  # 头部大小
        
        left_shoulder = transform_point(keypoints[5][0], keypoints[5][1])
        right_shoulder = transform_point(keypoints[6][0], keypoints[6][1])
        neck_bottom = ((left_shoulder[0] + right_shoulder[0]) // 2,
                       (left_shoulder[1] + right_shoulder[1]) // 2)
        
        # 如果有鼻子关键点，直接使用它作为头部中心
        if keypoints[0][2] > 0.1:
            head_center = transform_point(keypoints[0][0], keypoints[0][1])
        else:
            # 否则从肩膀中点向上推算：脖子长度约为肩宽的0.4倍，头部半径再向上
            neck_length = int(shoulder_dist * scale * 0.4)
            head_center = (neck_bottom[0], neck_bottom[1] - neck_length - head_radius)
    
    # 2. 绘制脖子（从肩膀中点到头部底部）
    if head_center is not None and neck_bottom is not None:
        neck_top = (head_center[0], head_center[1] + head_radius)
        cv2.line(canvas_rgb, neck_bottom, neck_top, body_color, 5)
    
    # 3. 绘制头部（简单的圆球，在脖子上方）
    if head_center is not None:
        # 绘制头部轮廓 - 只是一个简单的圆球
        cv2.circle(canvas_rgb, head_center, head_radius, head_color, -1)
        cv2.circle(canvas_rgb, head_center, head_radius, body_color, 2)
    
    # 3. 绘制躯干和四肢骨架（粗线条）
    thickness_map = {
        'torso': 8,      # 躯干
        'arm_upper': 6,   # 大臂
        'arm_lower': 5,   # 小臂
        'leg_upper': 7,   # 大腿
        'leg_lower': 6    # 小腿
    }
    
    # 定义不同部位的连接和粗细
    body_parts = [
        # 躯干
        ([5, 6], thickness_map['torso']),           # 肩膀
        ([5, 11], thickness_map['torso']),          # 左侧躯干
        ([6, 12], thickness_map['torso']),          # 右侧躯干
        ([11, 12], thickness_map['torso']),         # 髋部
        
        # 左臂
        ([5, 7], thickness_map['arm_upper']),       # 左大臂
        ([7, 9], thickness_map['arm_lower']),       # 左小臂
        
        # 右臂
        ([6, 8], thickness_map['arm_upper']),       # 右大臂
        ([8, 10], thickness_map['arm_lower']),      # 右小臂
        
        # 左腿
        ([11, 13], thickness_map['leg_upper']),     # 左大腿
        ([13, 15], thickness_map['leg_lower']),     # 左小腿
        
        # 右腿
        ([12, 14], thickness_map['leg_upper']),     # 右大腿
        ([14, 16], thickness_map['leg_lower']),     # 右小腿
    ]
    
    # 绘制身体各部分
    for connection, thickness in body_parts:
        pt1_idx, pt2_idx = connection
        if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and
            keypoints[pt1_idx][2] > 0.3 and keypoints[pt2_idx][2] > 0.3):
            pt1 = transform_point(keypoints[pt1_idx][0], keypoints[pt1_idx][1])
            pt2 = transform_point(keypoints[pt2_idx][0], keypoints[pt2_idx][1])
            cv2.line(canvas_rgb, pt1, pt2, body_color, thickness)
    
    # 4. 绘制关节点（较大的圆圈）
    important_joints = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    for i in important_joints:
        if i < len(keypoints) and keypoints[i][2] > 0.3:
            pt = transform_point(keypoints[i][0], keypoints[i][1])
            cv2.circle(canvas_rgb, pt, 8, joint_color, -1)
            cv2.circle(canvas_rgb, pt, 8, body_color, 2)
    
    # 将白色背景转换为透明背景
    # 创建RGBA画布
    canvas_rgba = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    
    # 复制RGB通道
    canvas_rgba[:, :, :3] = canvas_rgb
    
    # 设置Alpha通道：白色背景(255,255,255)变为透明，其他区域不透明
    # 计算每个像素与白色的差异
    white = np.array([255, 255, 255], dtype=np.uint8)
    diff = np.sum(np.abs(canvas_rgb.astype(np.int16) - white), axis=2)
    
    # 如果像素接近白色（差异小于阈值），设为透明；否则设为不透明
    threshold = 10  # 阈值，允许一些误差
    canvas_rgba[:, :, 3] = np.where(diff < threshold, 0, 255).astype(np.uint8)
    
    return canvas_rgba

def save_stickman_image(stickman_canvas, save_dir="punch_photos"):
    """保存火柴人图片到本地"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stickman_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    
    # 保存图片
    cv2.imwrite(filepath, stickman_canvas)
    print(f"火柴人图片已保存: {filepath}")
    return filepath

def create_split_screen(left_frame, right_stickman, canvas_width=1280, canvas_height=720):
    """创建左右分屏画布"""
    # 左边画面宽度，右边火柴人宽度
    left_width = int(canvas_width * 0.6)  # 左边占60%
    right_width = canvas_width - left_width  # 右边占40%
    
    # 创建画布
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # 调整左边摄像头画面大小
    left_resized = cv2.resize(left_frame, (left_width, canvas_height))
    
    # 调整右边火柴人大小（保持纵横比）
    stickman_h, stickman_w = right_stickman.shape[:2]
    scale = min(right_width / stickman_w, canvas_height / stickman_h) * 0.9  # 留一些边距
    new_w = int(stickman_w * scale)
    new_h = int(stickman_h * scale)
    right_resized = cv2.resize(right_stickman, (new_w, new_h))
    
    # 将火柴人居中放置在右边区域
    y_offset = (canvas_height - new_h) // 2
    x_offset = left_width + (right_width - new_w) // 2
    
    # 将左边画面放到画布
    canvas[:, :left_width] = left_resized
    
    # 将右边火柴人放到画布（处理透明背景）
    if right_resized.shape[2] == 4:  # RGBA格式
        # 提取RGB和Alpha通道
        stickman_rgb = right_resized[:, :, :3]
        stickman_alpha = right_resized[:, :, 3] / 255.0
        
        # 获取画布中对应的区域
        bg_region = canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
        
        # Alpha混合
        for c in range(3):
            bg_region[:, :, c] = (stickman_alpha * stickman_rgb[:, :, c] + 
                                  (1 - stickman_alpha) * bg_region[:, :, c]).astype(np.uint8)
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = bg_region
    else:  # RGB格式（向后兼容）
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = right_resized
    
    # 绘制中间分隔线
    cv2.line(canvas, (left_width, 0), (left_width, canvas_height), (200, 200, 200), 2)
    
    return canvas

print("姿态打卡系统初始化...")
print("请进入摄像头画面并保持pose 3秒进行打卡")
print("按 'q' 键退出程序")

# 初始化变量
frozen_stickman = None  # 用于存储打卡成功后固定的火柴人图像

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("无法读取摄像头画面")
        break
    
    # 计算FPS    
    # 进行姿态估计推理（使用增强后的画面）
    results = model(frame, verbose=False)
    
    # 创建显示帧（带可视化）
    display_frame = frame.copy()
    
    # 绘制YOLO姿态检测结果（根据配置决定是否显示）
    if show_detection_results and results[0].keypoints is not None:
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
                    # 可选：显示关键点编号
                    # cv2.putText(display_frame, f"{i}", (int(x), int(y-10)), 
                    #           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # 获取关键点数据
    current_keypoints = None
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.data.cpu().numpy()
        num_people = len(keypoints)
        
        # 检查是否只有一个人
        if num_people == 1:
            person_keypoints = keypoints[0]
            current_keypoints = person_keypoints
            
            #还没人进入打卡区域
            if punch_state == "waiting":
                punch_state = "detecting"
                # print("检测到人员进入画面")
            #已经有人在打卡区域
            elif punch_state == "detecting":
                punch_state = "posing"
                pose_start_time = time.time()
                last_pose_keypoints = person_keypoints.copy()
                # print("开始检测pose，请保持姿态3秒...")

            #已经摆pose了，开始检测pose是否稳定
            elif punch_state == "posing":
                # 检查姿态是否稳定
                pose_distance = calculate_pose_distance(person_keypoints, last_pose_keypoints)
                current_time = time.time()
                elapsed_time = current_time - pose_start_time
                
                # 显示当前姿态距离用于调试
                # cv2.putText(display_frame, f"Pose Distance: {pose_distance:.1f}", 
                #           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if pose_distance < pose_stable_threshold:
                    # 姿态稳定，更新计时
                    remaining_time = pose_duration - elapsed_time
                    
                    if remaining_time <= 0:
                        # 时间到，打卡成功
                        punch_state = "success"
                        # 生成并固定火柴人图像
                        frozen_stickman = draw_stickman(person_keypoints)
                        # 保存火柴人图片
                        save_stickman_image(frozen_stickman)
                        print("打卡完成！")
                    else:
                        # 显示倒计时
                        cv2.putText(display_frame, f"Hold pose: {remaining_time:.1f}s", 
                                  (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # 更新参考姿态（允许轻微移动）
                        last_pose_keypoints = person_keypoints.copy()
                else:
                    # 姿态不稳定，重新开始
                    punch_state = "detecting"
                    pose_start_time = None
                    print(f"姿态不稳定，距离: {pose_distance:.1f}，阈值: {pose_stable_threshold}")
            
            elif punch_state == "success":
                # 成功状态，等待重置
                cv2.putText(display_frame, "Punch Success!", (10, 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "Move away to reset", (10, 140), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        elif num_people > 1:
            # 多个人的情况
            if punch_state == "success":
                # 打卡成功后，允许多人存在，不重置
                cv2.putText(display_frame, "Punch Success!", (10, 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "Move away to reset", (10, 140), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                # 其他状态下多个人，重置状态
                punch_state = "waiting"
                pose_start_time = None
                cv2.putText(display_frame, "Too many people! Only one person allowed", 
                          (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        else:
            # 没有人，重置状态
            if punch_state == "success":
                # 打卡成功后，人离开则重置整个系统
                punch_state = "waiting"
                pose_start_time = None
                frozen_stickman = None  # 重置固定的火柴人
                print("系统已重置，等待下一位用户")
            elif punch_state != "waiting":
                punch_state = "waiting"
                pose_start_time = None
                print("区域内无人，系统已重置")
    
    # 显示状态信息
    status_text = {
        "waiting": "Waiting for person...",
        "detecting": "Person detected, preparing...",
        "posing": "Hold your pose!",
        "capturing": "Capturing...",
        "success": "Success!"
    }
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    # 显示FPS（左上角）
    cv2.putText(display_frame, f"FPS: {fps:.1f}", 
              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 显示状态（FPS下方）
    cv2.putText(display_frame, f"Status: {status_text.get(punch_state, 'Unknown')}", 
              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 显示检测到的人数
    # num_people = len(keypoints) if results[0].keypoints is not None else 0
    # cv2.putText(display_frame, f"People: {num_people}", (10, 60), 
    #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 在右上角显示状态图标
    icon_size = 80  # 图标大小
    icon_x = display_frame.shape[1] - icon_size - 10  # 右上角，距离边缘10像素
    icon_y = 10
    
    # 根据状态计算进度和显示图标
    if punch_state == "waiting":
        # 等待状态：显示红色图标
        current_icon = red_icon
        progress = 0.0
    elif punch_state == "detecting":
        # 检测到人：红色开始淡化
        current_icon = blend_icons(red_icon, green_icon, 0.2)
        progress = 0.2
    elif punch_state == "posing":
        # 正在打卡：根据倒计时显示进度
        if pose_start_time is not None:
            elapsed = time.time() - pose_start_time
            progress = min(elapsed / pose_duration, 1.0)  # 0.0 -> 1.0
            current_icon = blend_icons(red_icon, green_icon, progress)
        else:
            current_icon = red_icon
            progress = 0.0
    elif punch_state == "success":
        # 成功：显示绿色图标
        current_icon = green_icon
        progress = 1.0
    else:
        current_icon = red_icon
        progress = 0.0
    
    # 缩放图标到指定大小
    resized_icon = cv2.resize(current_icon, (icon_size, icon_size))
    
    # 叠加图标到画面
    display_frame = overlay_icon_with_alpha(display_frame, resized_icon, icon_x, icon_y, alpha=0.9)
    
    # 生成右边的火柴人画面
    if punch_state == "success" and frozen_stickman is not None:
        # 打卡成功后，显示固定的火柴人（保持透明背景）
        right_stickman = frozen_stickman.copy()
        # 在火柴人上添加"打卡成功"文字
        cv2.putText(right_stickman, "Punch Success!", 
                   (right_stickman.shape[1]//2 - 150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
    elif current_keypoints is not None and len(current_keypoints) > 0:
        # 实时显示火柴人（透明背景）
        right_stickman = draw_stickman(current_keypoints)
    else:
        # 没有检测到人，显示空白画布和提示
        right_stickman = np.ones((700, 500, 3), dtype=np.uint8) * 255
        cv2.putText(right_stickman, "Waiting for pose...", 
                   (right_stickman.shape[1]//2 - 140, right_stickman.shape[0]//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
    
    # 创建左右分屏画面
    split_screen = create_split_screen(display_frame, right_stickman)
    
    # 显示画面
    cv2.imshow("Punch Clock System", split_screen)
    
    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
print("程序结束")