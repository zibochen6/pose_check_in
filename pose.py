import cv2
import time
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from config import *
import mediapipe as mp

# 加载YOLO姿态估计模型
model = YOLO(MODEL_PATH,task="pose")  # 使用TensorRT引擎文件

# 初始化MediaPipe手部检测
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

def calculate_pose_distance(keypoints1, keypoints2, hands_data1=None, hands_data2=None):
    """计算两个姿态之间的距离（包括身体和手部）"""
    if keypoints1 is None or keypoints2 is None:
        return float('inf')
    
    total_distance = 0
    valid_points = 0
    
    # 1. 身体关键点距离计算
    important_keypoints = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # 鼻子、左肩、右肩、左肘、右肘、左腕、右腕、左髋、右髋、左膝、右膝、左踝、右踝
    
    for i in important_keypoints:
        if (i < len(keypoints1) and i < len(keypoints2) and 
            keypoints1[i][2] > 0.5 and keypoints2[i][2] > 0.5):
            distance = np.sqrt((keypoints1[i][0] - keypoints2[i][0])**2 + 
                             (keypoints1[i][1] - keypoints2[i][1])**2)
            total_distance += distance
            valid_points += 1
    
    # 2. 手部关键点距离计算
    if hands_data1 and hands_data2 and len(hands_data1) > 0 and len(hands_data2) > 0:
        # 尝试匹配相同的手（左手对左手，右手对右手）
        for hand1 in hands_data1:
            for hand2 in hands_data2:
                if hand1['label'] == hand2['label']:  # 相同的手
                    landmarks1 = hand1['landmarks']
                    landmarks2 = hand2['landmarks']
                    
                    # 只计算关键手部点（手腕、指尖、掌心关键点）
                    key_hand_points = [0, 4, 8, 12, 16, 20]  # 手腕和5个指尖
                    
                    for idx in key_hand_points:
                        if idx < len(landmarks1) and idx < len(landmarks2):
                            # 手部坐标是归一化的，需要乘以画面尺寸
                            dist = np.sqrt(
                                ((landmarks1[idx][0] - landmarks2[idx][0]) * frame_width)**2 + 
                                ((landmarks1[idx][1] - landmarks2[idx][1]) * frame_height)**2
                            )
                            total_distance += dist
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

def detect_hands(image):
    """使用MediaPipe检测手部关键点"""
    # 转换为RGB（MediaPipe需要RGB格式）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)
    
    hands_data = []
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # 获取手部类型（左手或右手）
            hand_label = handedness.classification[0].label  # "Left" or "Right"
            
            # 提取21个手部关键点
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            
            hands_data.append({
                'label': hand_label,
                'landmarks': np.array(landmarks)
            })
    
    return hands_data

def draw_stickman_with_hands(keypoints, hands_data, canvas_width=500, canvas_height=700):
    """根据关键点和手部数据在白色画布上绘制完整的火柴人"""
    # 先绘制基础火柴人
    canvas = draw_stickman(keypoints, canvas_width, canvas_height)
    
    if not hands_data:
        return canvas
    
    # 找到关键点的边界来进行缩放和居中（与draw_stickman保持一致）
    valid_points = []
    for x, y, conf in keypoints:
        if conf > 0.3:
            valid_points.append((x, y))
    
    if len(valid_points) == 0:
        return canvas
    
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
    
    # 转换手部关键点坐标到画布坐标
    def transform_hand_point(x, y):
        # 手部坐标是归一化的(0-1)，需要转换到画布坐标系统
        new_x = int(x * frame_width * scale + offset_x)
        new_y = int(y * frame_height * scale + offset_y)
        return (new_x, new_y)
    
    # MediaPipe手部关键点连接关系
    hand_connections = [
        # 拇指
        [0, 1], [1, 2], [2, 3], [3, 4],
        # 食指
        [0, 5], [5, 6], [6, 7], [7, 8],
        # 中指
        [0, 9], [9, 10], [10, 11], [11, 12],
        # 无名指
        [0, 13], [13, 14], [14, 15], [15, 16],
        # 小指
        [0, 17], [17, 18], [18, 19], [19, 20],
        # 手掌
        [5, 9], [9, 13], [13, 17]
    ]
    
    # 定义颜色方案（与身体一致）
    body_color = (50, 50, 50)  # 深灰色（与身体骨架一致）
    joint_color = (100, 150, 255)  # 浅蓝色关节（与身体关节一致）
    
    # 定义手指线条粗细
    finger_thickness = {
        'palm': 4,        # 手掌主线
        'thumb': 3,       # 拇指
        'finger': 3,      # 其他手指
        'connection': 4   # 手掌连接线
    }
    
    for hand in hands_data:
        landmarks = hand['landmarks']
        
        # 首先绘制手掌连接线（更粗）
        palm_connections = [[5, 9], [9, 13], [13, 17], [0, 5], [0, 17]]
        for connection in palm_connections:
            pt1_idx, pt2_idx = connection
            if pt1_idx < len(landmarks) and pt2_idx < len(landmarks):
                pt1 = transform_hand_point(landmarks[pt1_idx][0], landmarks[pt1_idx][1])
                pt2 = transform_hand_point(landmarks[pt2_idx][0], landmarks[pt2_idx][1])
                
                if (0 <= pt1[0] < canvas_width and 0 <= pt1[1] < canvas_height and
                    0 <= pt2[0] < canvas_width and 0 <= pt2[1] < canvas_height):
                    cv2.line(canvas, pt1, pt2, body_color, finger_thickness['palm'])
        
        # 绘制五根手指（分别设置粗细）
        finger_groups = [
            ([0, 1, 2, 3, 4], finger_thickness['thumb']),     # 拇指
            ([5, 6, 7, 8], finger_thickness['finger']),        # 食指
            ([9, 10, 11, 12], finger_thickness['finger']),     # 中指
            ([13, 14, 15, 16], finger_thickness['finger']),    # 无名指
            ([17, 18, 19, 20], finger_thickness['finger'])     # 小指
        ]
        
        for finger, thickness in finger_groups:
            for i in range(len(finger) - 1):
                pt1_idx = finger[i]
                pt2_idx = finger[i + 1]
                if pt1_idx < len(landmarks) and pt2_idx < len(landmarks):
                    pt1 = transform_hand_point(landmarks[pt1_idx][0], landmarks[pt1_idx][1])
                    pt2 = transform_hand_point(landmarks[pt2_idx][0], landmarks[pt2_idx][1])
                    
                    if (0 <= pt1[0] < canvas_width and 0 <= pt1[1] < canvas_height and
                        0 <= pt2[0] < canvas_width and 0 <= pt2[1] < canvas_height):
                        cv2.line(canvas, pt1, pt2, body_color, thickness)
        
        # 绘制手部关节点（与身体关节风格一致）
        important_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        for i in important_joints:
            if i < len(landmarks):
                pt = transform_hand_point(landmarks[i][0], landmarks[i][1])
                if 0 <= pt[0] < canvas_width and 0 <= pt[1] < canvas_height:
                    # 手腕和指尖稍大，其他关节较小
                    if i in [0, 4, 8, 12, 16, 20]:  # 手腕和5个指尖
                        radius = 6
                    else:  # 其他关节
                        radius = 5
                    cv2.circle(canvas, pt, radius, joint_color, -1)
                    cv2.circle(canvas, pt, radius, body_color, 2)
    
    return canvas

def draw_stickman(keypoints, canvas_width=500, canvas_height=700):
    """根据关键点在白色画布上绘制美观的火柴人"""
    # 创建白色画布
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # 找到关键点的边界来进行缩放和居中
    valid_points = []
    for x, y, conf in keypoints:
        if conf > 0.3:  # 降低阈值以获取更多关键点
            valid_points.append((x, y))
    
    if len(valid_points) == 0:
        return canvas
    
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
    
    # 定义身体各部分的线条粗细和颜色
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
        cv2.line(canvas, neck_bottom, neck_top, body_color, 5)
    
    # 3. 绘制头部（简单的圆球，在脖子上方）
    if head_center is not None:
        # 绘制头部轮廓 - 只是一个简单的圆球
        cv2.circle(canvas, head_center, head_radius, head_color, -1)
        cv2.circle(canvas, head_center, head_radius, body_color, 2)
    
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
            cv2.line(canvas, pt1, pt2, body_color, thickness)
    
    # 4. 绘制关节点（较大的圆圈）
    important_joints = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    for i in important_joints:
        if i < len(keypoints) and keypoints[i][2] > 0.3:
            pt = transform_point(keypoints[i][0], keypoints[i][1])
            cv2.circle(canvas, pt, 8, joint_color, -1)
            cv2.circle(canvas, pt, 8, body_color, 2)
    
    return canvas

def show_photo_and_stickman(frame, person_bbox, current_keypoints):
    """显示拍照的照片和火柴人形象"""
    if person_bbox is None:
        print("无法获取人的检测框")
        return
    
    x_min, y_min, x_max, y_max = person_bbox
    
    # 裁剪人的区域
    cropped_person = frame[y_min:y_max, x_min:x_max]
    
    if cropped_person.size == 0:
        print("检测框区域无效")
        return
    
    # 直接使用当前检测到的关键点（更准确）
    print("正在生成火柴人...")
    if current_keypoints is not None and len(current_keypoints) > 0:
        # 检测手部关键点
        print("正在检测手部...")
        hands_data = detect_hands(frame)
        
        if hands_data:
            print(f"检测到 {len(hands_data)} 只手")
            for hand in hands_data:
                print(f"  - {hand['label']} 手")
        else:
            print("未检测到手部，将绘制基础火柴人")
        
        # 绘制带手部的完整火柴人
        stickman_canvas = draw_stickman_with_hands(current_keypoints, hands_data)
        
        # 显示火柴人窗口
        cv2.namedWindow("Stickman - Punch Pose", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Stickman - Punch Pose", 500, 700)
        cv2.imshow("Stickman - Punch Pose", stickman_canvas)
        
        print("打卡成功！")
        
        # 打印关键点信息用于调试
        print(f"关键点数量: {len(current_keypoints)}")
        for i, (x, y, conf) in enumerate(current_keypoints):
            if i < 5:  # 只打印前5个（头部相关）
                print(f"关键点{i}: 置信度={conf:.2f}")
    else:
        print("无法获取关键点数据")
    
    # 也显示原始照片
    cv2.namedWindow("Punch Photo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Punch Photo", 600, 800)
    cv2.imshow("Punch Photo", cropped_person)

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
    

    # 优化暗光环境下的画面质量
    enhanced_frame = enhance_frame_for_dark_lighting(frame)
    
    # 进行姿态估计推理（使用增强后的画面）
    results = model(enhanced_frame, verbose=False)
    
    # 创建显示帧（带可视化）
    display_frame = frame.copy()
    
    # 绘制ROI区域
    cv2.rectangle(display_frame, (roi_x, roi_y), 
                  (roi_x + roi_width, roi_y + roi_height), (0, 255, 255), 2)
    # cv2.putText(display_frame, "Detection Area", (roi_x, roi_y - 10), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
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
                    # 保存当前手部数据
                    last_hands_data = detect_hands(frame)
                    # print("开始检测pose，请保持姿态3秒...")

                #已经摆pose了，开始检测pose是否稳定
                elif punch_state == "posing":
                    # 获取当前手部数据
                    current_hands = detect_hands(frame)
                    
                    # 检查姿态是否稳定（包括身体和手部）
                    pose_distance = calculate_pose_distance(person_keypoints, last_pose_keypoints, current_hands, last_hands_data)
                    current_time = time.time()
                    elapsed_time = current_time - pose_start_time
                    
                    # 显示当前姿态距离用于调试
                    # cv2.putText(display_frame, f"Pose Distance: {pose_distance:.1f}", 
                    #           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    if pose_distance < pose_stable_threshold:
                        # 姿态稳定，更新计时
                        remaining_time = pose_duration - elapsed_time
                        
                        if remaining_time <= 0:
                            # 时间到，拍照
                            punch_state = "capturing"
                            person_bbox = get_person_bounding_box_from_detection(results)
                            show_photo_and_stickman(frame, person_bbox, person_keypoints)  # 显示原始照片和火柴人
                            punch_state = "success"
                            print("打卡完成！")
                        else:
                            # 显示倒计时
                            cv2.putText(display_frame, f"Hold pose: {remaining_time:.1f}s", 
                                      (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            # 更新参考姿态（允许轻微移动）
                            last_pose_keypoints = person_keypoints.copy()
                            last_hands_data = current_hands
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
            else:
                # 人不在ROI内，重置状态
                if punch_state != "waiting":
                    punch_state = "waiting"
                    pose_start_time = None
                    print("请进入检测区域")
        
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
            if punch_state != "waiting":
                punch_state = "waiting"
                pose_start_time = None
                print("区域内无人，系统已重置")
    
    # 实时检测手部并在主窗口显示（根据配置决定是否显示）
    hands_data = detect_hands(frame)
    if show_detection_results and hands_data:
        # 绘制手部检测结果
        for hand in hands_data:
            landmarks = hand['landmarks']
            hand_label = hand['label']
            
            # 将归一化坐标转换为像素坐标
            h, w = frame.shape[:2]
            
            # 定义手部连接关系
            hand_connections = [
                # 拇指
                [0, 1], [1, 2], [2, 3], [3, 4],
                # 食指
                [0, 5], [5, 6], [6, 7], [7, 8],
                # 中指
                [0, 9], [9, 10], [10, 11], [11, 12],
                # 无名指
                [0, 13], [13, 14], [14, 15], [15, 16],
                # 小指
                [0, 17], [17, 18], [18, 19], [19, 20],
                # 手掌
                [5, 9], [9, 13], [13, 17]
            ]
            
            # 绘制手部骨架连接线
            for connection in hand_connections:
                pt1_idx, pt2_idx = connection
                if pt1_idx < len(landmarks) and pt2_idx < len(landmarks):
                    pt1 = (int(landmarks[pt1_idx][0] * w), int(landmarks[pt1_idx][1] * h))
                    pt2 = (int(landmarks[pt2_idx][0] * w), int(landmarks[pt2_idx][1] * h))
                    cv2.line(display_frame, pt1, pt2, (0, 255, 0), 2)
            
            # 绘制手部关键点
            for i, landmark in enumerate(landmarks):
                x = int(landmark[0] * w)
                y = int(landmark[1] * h)
                # 手腕和指尖用不同颜色和大小
                if i in [0, 4, 8, 12, 16, 20]:  # 手腕和指尖
                    cv2.circle(display_frame, (x, y), 6, (255, 0, 0), -1)
                else:
                    cv2.circle(display_frame, (x, y), 4, (0, 255, 255), -1)
            
            # 在手腕位置显示手的标签
            if len(landmarks) > 0:
                wrist_x = int(landmarks[0][0] * w)
                wrist_y = int(landmarks[0][1] * h)
                cv2.putText(display_frame, f"{hand_label} Hand", (wrist_x - 30, wrist_y - 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
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
    # cv2.putText(display_frame, f"FPS: {fps:.1f}", 
    #           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 显示状态（FPS下方）
    cv2.putText(display_frame, f"Status: {status_text.get(punch_state, 'Unknown')}", 
              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 显示检测到的人数
    # num_people = len(keypoints) if results[0].keypoints is not None else 0
    # cv2.putText(display_frame, f"People: {num_people}", (10, 60), 
    #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 显示检测到的手数
    # num_hands = len(hands_data) if hands_data else 0
    # cv2.putText(display_frame, f"Hands: {num_hands}", (10, 90), 
    #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
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
    
    # 显示画面
    cv2.imshow("Punch Clock System", display_frame)
    
    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
print("程序结束")