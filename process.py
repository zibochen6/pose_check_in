from ultralytics import YOLO
from punch_config import *
import cv2
import time
import numpy as np
import os
from datetime import datetime
import subprocess

class punch_system:
    
    def __init__(self):
        self.model = YOLO(model_path,task="pose")
        # 加载状态图标
        self.red_icon = cv2.imread("icon/red.png", cv2.IMREAD_UNCHANGED)
        self.green_icon = cv2.imread("icon/green.png", cv2.IMREAD_UNCHANGED)
        # 加载默认头像
        self.default_avatar = cv2.imread(default_avatar_path)
        if self.default_avatar is None:
            print(f"警告：无法加载默认头像 {default_avatar_path}")
        else:
            print(f"默认头像已加载: {default_avatar_path}")
        self.user_avatar = None  # 用户上传的头像（暂时为None，使用默认头像）
        self.cap = None
        self.photos_dir = "punch_photos"
        self.punch_state = "waiting"
        #存储打卡人形象
        self.frozen_stickman = None
        self.show_detect = False
        self.show_fps = False
        # 屏幕分辨率（自动获取）
        self.screen_width = 1920
        self.screen_height = 1080
        self.window_name = "Punch Clock System"
        self.init()

    def init(self):
        print("系统初始化...")
        self.cap = cv2.VideoCapture(camera_index)
        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # 创建照片保存目录
        if not os.path.exists(self.photos_dir):
            os.makedirs(self.photos_dir)
    
        # 获取屏幕分辨率
        try:
            result = subprocess.run(['xrandr'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if '*' in line:  # 当前使用的分辨率会有*标记
                    parts = line.split()
                    for part in parts:
                        if 'x' in part and part[0].isdigit():
                            self.screen_width, self.screen_height = map(int, part.split('x'))
                            break
                    break
            print(f"屏幕分辨率: {self.screen_width}x{self.screen_height}")
        except:
            print(f"无法自动获取屏幕分辨率，使用默认值{self.screen_width}x{self.screen_height}")
        
        # 创建全屏窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if fullscreen_mode:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("全屏模式已启用")
    

    def scale_to_fullscreen(self, image):
        """将图像缩放以填充整个屏幕"""
        resized = cv2.resize(image, (self.screen_width, self.screen_height), interpolation=cv2.INTER_LINEAR)
        return resized

    def set_user_avatar(self, avatar_path):
        """设置用户自定义头像
        
        Args:
            avatar_path: 头像图片路径，如果为None则使用默认头像
        """
        if avatar_path is None:
            self.user_avatar = None
            print("已重置为默认头像")
        else:
            avatar = cv2.imread(avatar_path)
            if avatar is not None:
                self.user_avatar = avatar
                print(f"用户头像已设置: {avatar_path}")
            else:
                print(f"错误：无法加载头像 {avatar_path}")

    def create_circular_avatar(self, avatar_image, size):
        """创建圆形头像
        
        Args:
            avatar_image: 原始头像图片
            size: 输出圆形头像的直径
            
        Returns:
            带Alpha通道的圆形头像 (BGRA格式)
        """
        if avatar_image is None:
            return None
        
        # 调整头像大小为正方形
        avatar_resized = cv2.resize(avatar_image, (size, size))
        
        # 创建BGRA图像
        avatar_bgra = cv2.cvtColor(avatar_resized, cv2.COLOR_BGR2BGRA)
        
        # 创建圆形蒙版
        center = size // 2
        radius = size // 2
        
        # 创建蒙版
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (center, center), radius, 255, -1)
        
        # 应用蒙版到Alpha通道
        avatar_bgra[:, :, 3] = mask
        
        return avatar_bgra

    def show_detect_result(self,results,display_frame):
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

    def calculate_pose_distance(self,keypoints1,keypoints2):
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


    def draw_stickman(self,keypoints,canvas_width=500, canvas_height=700):
        """根据关键点绘制头像身体+四肢的创意形象"""
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
        
        # 计算偏移，使形象居中
        offset_x = (canvas_width - person_width * scale) / 2 - x_min * scale
        offset_y = (canvas_height - person_height * scale) / 2 - y_min * scale
        
        # 转换关键点坐标到画布坐标
        def transform_point(x, y):
            new_x = int(x * scale + offset_x)
            new_y = int(y * scale + offset_y)
            return (new_x, new_y)
        
        # 定义身体各部分的线条粗细和颜色（RGB格式）
        limb_color = (50, 50, 50)  # 深灰色四肢
        joint_color = (100, 150, 255)  # 浅蓝色关节（更柔和）
        
        # 1. 计算身体中心位置（肩膀和髋部的中间点）
        body_center = None
        body_size = 100  # 默认身体大小
        
        # 获取肩膀和髋部关键点
        left_shoulder = None
        right_shoulder = None
        left_hip = None
        right_hip = None
        
        if keypoints[5][2] > 0.3:  # 左肩
            left_shoulder = transform_point(keypoints[5][0], keypoints[5][1])
        if keypoints[6][2] > 0.3:  # 右肩
            right_shoulder = transform_point(keypoints[6][0], keypoints[6][1])
        if keypoints[11][2] > 0.3:  # 左髋
            left_hip = transform_point(keypoints[11][0], keypoints[11][1])
        if keypoints[12][2] > 0.3:  # 右髋
            right_hip = transform_point(keypoints[12][0], keypoints[12][1])
        
        # 计算身体中心和大小
        if left_shoulder and right_shoulder and left_hip and right_hip:
            # 计算肩膀中心和髋部中心
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) // 2
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) // 2
            hip_center_x = (left_hip[0] + right_hip[0]) // 2
            hip_center_y = (left_hip[1] + right_hip[1]) // 2
            
            # 身体中心在肩膀和髋部之间
            body_center = ((shoulder_center_x + hip_center_x) // 2,
                          (shoulder_center_y + hip_center_y) // 2)
            
            # 根据肩宽和躯干高度计算身体大小
            shoulder_width = np.sqrt((left_shoulder[0] - right_shoulder[0])**2 + 
                                    (left_shoulder[1] - right_shoulder[1])**2)
            torso_height = np.sqrt((shoulder_center_x - hip_center_x)**2 + 
                                  (shoulder_center_y - hip_center_y)**2)
            
            # 身体大小取肩宽和躯干高度的最大值，再放大一些
            body_size = int(max(shoulder_width, torso_height) * 1.3)
        
        # 2. 绘制头像作为身体（圆形大头像）
        if body_center is not None:
            avatar_to_use = self.user_avatar if self.user_avatar is not None else self.default_avatar
            
            if avatar_to_use is not None:
                # 创建圆形头像作为身体
                circular_avatar = self.create_circular_avatar(avatar_to_use, body_size)
                
                if circular_avatar is not None:
                    avatar_x = body_center[0] - body_size // 2
                    avatar_y = body_center[1] - body_size // 2
                    
                    # 确保头像位置在画布范围内
                    if (avatar_x >= 0 and avatar_y >= 0 and 
                        avatar_x + body_size <= canvas_rgb.shape[1] and 
                        avatar_y + body_size <= canvas_rgb.shape[0]):
                        
                        # 提取头像的RGB和Alpha通道
                        avatar_rgb = circular_avatar[:, :, :3]
                        avatar_alpha = circular_avatar[:, :, 3] / 255.0
                        
                        # 获取画布中对应的区域
                        canvas_region = canvas_rgb[avatar_y:avatar_y+body_size, 
                                                   avatar_x:avatar_x+body_size]
                        
                        # Alpha混合
                        for c in range(3):
                            canvas_region[:, :, c] = (avatar_alpha * avatar_rgb[:, :, c] + 
                                                    (1 - avatar_alpha) * canvas_region[:, :, c]).astype(np.uint8)
                        
                        canvas_rgb[avatar_y:avatar_y+body_size, 
                                  avatar_x:avatar_x+body_size] = canvas_region
                        
                        # 绘制头像边框
                        cv2.circle(canvas_rgb, body_center, body_size // 2, limb_color, 3)
        
        # 3. 辅助函数：计算从圆心到某点的方向上，圆边缘的交点
        def get_edge_point(center, target_point, radius):
            """计算从圆心指向目标点方向上，圆边缘的点"""
            dx = target_point[0] - center[0]
            dy = target_point[1] - center[1]
            distance = np.sqrt(dx*dx + dy*dy)
            if distance == 0:
                return center
            # 单位向量
            ux = dx / distance
            uy = dy / distance
            # 圆边缘的点
            edge_x = int(center[0] + ux * radius)
            edge_y = int(center[1] + uy * radius)
            return (edge_x, edge_y)
        
        # 4. 绘制四肢（从身体边缘延伸出去）
        limb_thickness = 8
        
        if body_center is not None:
            body_radius = body_size // 2
            
            # 左臂 - 从身体边缘指向肘部方向延伸（自然跟随真实姿态）
            if keypoints[7][2] > 0.3:
                left_elbow = transform_point(keypoints[7][0], keypoints[7][1])
                
                # 计算手臂起点：从身体中心指向肘部的方向，在圆边缘找到起点
                arm_start = get_edge_point(body_center, left_elbow, body_radius)
                
                cv2.line(canvas_rgb, arm_start, left_elbow, limb_color, limb_thickness)
                cv2.circle(canvas_rgb, left_elbow, 6, joint_color, -1)
                cv2.circle(canvas_rgb, left_elbow, 6, limb_color, 2)
                
                # 小臂
                if keypoints[9][2] > 0.3:
                    left_wrist = transform_point(keypoints[9][0], keypoints[9][1])
                    cv2.line(canvas_rgb, left_elbow, left_wrist, limb_color, limb_thickness - 2)
                    cv2.circle(canvas_rgb, left_wrist, 5, joint_color, -1)
                    cv2.circle(canvas_rgb, left_wrist, 5, limb_color, 2)
            
            # 右臂 - 从身体边缘指向肘部方向延伸（自然跟随真实姿态）
            if keypoints[8][2] > 0.3:
                right_elbow = transform_point(keypoints[8][0], keypoints[8][1])
                
                # 计算手臂起点：从身体中心指向肘部的方向，在圆边缘找到起点
                arm_start = get_edge_point(body_center, right_elbow, body_radius)
                
                cv2.line(canvas_rgb, arm_start, right_elbow, limb_color, limb_thickness)
                cv2.circle(canvas_rgb, right_elbow, 6, joint_color, -1)
                cv2.circle(canvas_rgb, right_elbow, 6, limb_color, 2)
                
                # 小臂
                if keypoints[10][2] > 0.3:
                    right_wrist = transform_point(keypoints[10][0], keypoints[10][1])
                    cv2.line(canvas_rgb, right_elbow, right_wrist, limb_color, limb_thickness - 2)
                    cv2.circle(canvas_rgb, right_wrist, 5, joint_color, -1)
                    cv2.circle(canvas_rgb, right_wrist, 5, limb_color, 2)
            
            # 左腿
            if keypoints[11][2] > 0.3 and keypoints[13][2] > 0.3:
                left_hip = transform_point(keypoints[11][0], keypoints[11][1])
                left_knee = transform_point(keypoints[13][0], keypoints[13][1])
                
                hip_edge = get_edge_point(body_center, left_hip, body_radius)
                cv2.line(canvas_rgb, hip_edge, left_knee, limb_color, limb_thickness + 1)
                cv2.circle(canvas_rgb, left_knee, 6, joint_color, -1)
                cv2.circle(canvas_rgb, left_knee, 6, limb_color, 2)
                
                # 小腿
                if keypoints[15][2] > 0.3:
                    left_ankle = transform_point(keypoints[15][0], keypoints[15][1])
                    cv2.line(canvas_rgb, left_knee, left_ankle, limb_color, limb_thickness)
                    cv2.circle(canvas_rgb, left_ankle, 5, joint_color, -1)
                    cv2.circle(canvas_rgb, left_ankle, 5, limb_color, 2)
            
            # 右腿
            if keypoints[12][2] > 0.3 and keypoints[14][2] > 0.3:
                right_hip = transform_point(keypoints[12][0], keypoints[12][1])
                right_knee = transform_point(keypoints[14][0], keypoints[14][1])
                
                hip_edge = get_edge_point(body_center, right_hip, body_radius)
                cv2.line(canvas_rgb, hip_edge, right_knee, limb_color, limb_thickness + 1)
                cv2.circle(canvas_rgb, right_knee, 6, joint_color, -1)
                cv2.circle(canvas_rgb, right_knee, 6, limb_color, 2)
                
                # 小腿
                if keypoints[16][2] > 0.3:
                    right_ankle = transform_point(keypoints[16][0], keypoints[16][1])
                    cv2.line(canvas_rgb, right_knee, right_ankle, limb_color, limb_thickness)
                    cv2.circle(canvas_rgb, right_ankle, 5, joint_color, -1)
                    cv2.circle(canvas_rgb, right_ankle, 5, limb_color, 2)
        
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

    def save_stickman_image(self,image):
        """保存火柴人图片到本地"""
        if not os.path.exists(self.photos_dir):
            os.makedirs(self.photos_dir)
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stickman_{timestamp}.png"
        filepath = os.path.join(self.photos_dir, filename)
        
        # 保存图片
        cv2.imwrite(filepath, image)
        print(f"火柴人图片已保存: {filepath}")
        return filepath

    def process(self,results):
        # 获取关键点数据
        global current_keypoints,pose_start_time,last_pose_keypoints
        current_keypoints = None
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.data.cpu().numpy()
            num_people = len(keypoints)
            # 检查是否只有一个人
            if num_people == 1:
                person_keypoints = keypoints[0]
                current_keypoints = person_keypoints
                #还没人进入打卡区域
                if self.punch_state == "waiting":
                    self.punch_state = "detecting"
                #已经有人在打卡区域
                elif self.punch_state == "detecting":
                    self.punch_state = "posing"
                    pose_start_time = time.time()
                    last_pose_keypoints = person_keypoints.copy()
                #已经摆pose了，开始检测pose是否稳定
                elif self.punch_state == "posing":
                    # 检查姿态是否稳定,对比两个状态的骨骼点距离差距多大
                    pose_distance = self.calculate_pose_distance(person_keypoints, last_pose_keypoints)
                    current_time = time.time()
                    elapsed_time = current_time - pose_start_time
                    
                    # 显示当前姿态距离用于调试
                    cv2.putText(display_frame, f"Pose Distance: {pose_distance:.1f}", 
                              (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    if pose_distance < pose_stable_threshold:
                        # 姿态稳定，更新计时
                        remaining_time = pose_duration - elapsed_time
                        
                        if remaining_time <= 0:
                            # 时间到，打卡成功
                            self.punch_state = "success"
                            # 生成并固定火柴人图像
                            self.frozen_stickman = self.draw_stickman(person_keypoints)
                            # 保存火柴人图片
                            self.save_stickman_image(self.frozen_stickman)
                            print("打卡完成！")
                        else:
                            # 显示倒计时
                            cv2.putText(display_frame, f"Hold pose: {remaining_time:.1f}s", 
                                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            # 更新参考姿态（允许轻微移动）
                            last_pose_keypoints = person_keypoints.copy()
                    else:
                        # 姿态不稳定，重新开始
                        self.punch_state = "detecting"
                        pose_start_time = None
                        # print(f"姿态不稳定，距离: {pose_distance:.1f}，阈值: {pose_stable_threshold}")
                
                elif self.punch_state == "success":
                    # 成功状态，等待重置
                    cv2.putText(display_frame, "Punch Success!", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Move away to reset", (10, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            elif num_people > 1:
                # 多个人的情况
                if self.punch_state == "success":
                    # 打卡成功后，允许多人存在，不重置
                    cv2.putText(display_frame, "Punch Success!", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Move away to reset", (10, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    # 其他状态下多个人，重置状态
                    self.punch_state = "waiting"
                    pose_start_time = None
                    cv2.putText(display_frame, "Too many people! Only one person allowed", 
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            else:
                # 没有人，重置状态
                if self.punch_state == "success":
                    # 打卡成功后，人离开则重置整个系统
                    self.punch_state = "waiting"
                    pose_start_time = None
                    self.frozen_stickman = None  # 重置固定的火柴人
                    # print("系统已重置，等待下一位用户")
                elif self.punch_state != "waiting":
                    self.punch_state = "waiting"
                    pose_start_time = None
                    # print("区域内无人，系统已重置")
          

    def blend_icons(self, progress):
        """根据进度混合红绿图标
        progress: 0.0 (完全红色) -> 1.0 (完全绿色)
        """
        # 创建一个空白画布
        blended = np.zeros_like(self.red_icon)
        
        # 红色透明度：从1.0到0.0
        red_alpha = 1.0 - progress
        # 绿色透明度：从0.0到1.0
        green_alpha = progress
        
        # 混合两个图标
        if self.red_icon.shape[2] == 4:
            # 处理RGBA
            for c in range(3):
                blended[:, :, c] = (red_alpha * self.red_icon[:, :, c] + 
                                green_alpha * self.green_icon[:, :, c])
            # Alpha通道也混合
            blended[:, :, 3] = np.maximum(self.red_icon[:, :, 3], self.green_icon[:, :, 3])
        else:
            # 处理RGB
            blended = (red_alpha * self.red_icon + green_alpha * self.green_icon).astype(np.uint8)
        
        return blended.astype(np.uint8)

    def overlay_icon_with_alpha(self,background, icon, x, y, alpha=1.0):
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

    def create_split_screen(self,left_frame, right_stickman, canvas_width=1280, canvas_height=720):
        """创建左右分屏画布"""
        # 左边画面宽度，右边火柴人宽度
        left_width = int(canvas_width * 0.6)  # 左边占60%
        right_width = canvas_width - left_width  # 右边占40%
        
        # 创建画布
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        # 调整左边摄像头画面大小（使用INTER_NEAREST更快）
        left_resized = cv2.resize(left_frame, (left_width, canvas_height), interpolation=cv2.INTER_LINEAR)
        
        # 调整右边火柴人大小（保持纵横比）
        stickman_h, stickman_w = right_stickman.shape[:2]
        scale = min(right_width / stickman_w, canvas_height / stickman_h) * 0.9  # 留一些边距
        new_w = int(stickman_w * scale)
        new_h = int(stickman_h * scale)
        right_resized = cv2.resize(right_stickman, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
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

    def infer(self):
        global display_frame
        p_time = time.time()
        while True:
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                print("无法读取摄像头画面")
                break
                
            results = self.model(frame, verbose=False)
            display_frame = frame.copy()
            
            if self.show_detect and results[0].keypoints is not None:
                self.show_detect_result(results,display_frame)
            
            #处理结果
            self.process(results)

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
            if self.show_fps:
                cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示状态（FPS下方）
            cv2.putText(display_frame, f"Status: {status_text.get(self.punch_state, 'Unknown')}", 
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
            if self.punch_state == "waiting":
                # 等待状态：显示红色图标
                current_icon = self.red_icon
                progress = 0.0
            elif self.punch_state == "detecting":
                # 检测到人：红色开始淡化
                current_icon = self.blend_icons(0.2)
                progress = 0.2
            elif self.punch_state == "posing":
                # 正在打卡：根据倒计时显示进度
                if pose_start_time is not None:
                    elapsed = time.time() - pose_start_time
                    progress = min(elapsed / pose_duration, 1.0)  # 0.0 -> 1.0
                    current_icon = self.blend_icons(progress)
                else:
                    current_icon = self.red_icon
                    progress = 0.0
            elif self.punch_state == "success":
                # 成功：显示绿色图标
                current_icon = self.green_icon
                progress = 1.0
            else:
                current_icon = self.red_icon
                progress = 0.0
            
            # 缩放图标到指定大小
            resized_icon = cv2.resize(current_icon, (icon_size, icon_size))
            
            # 叠加图标到画面
            display_frame = self.overlay_icon_with_alpha(display_frame, resized_icon, icon_x, icon_y, alpha=0.9)
            
            # 生成右边的火柴人画面
            if self.punch_state == "success" and self.frozen_stickman is not None:
                # 打卡成功后，显示固定的火柴人（保持透明背景）
                right_stickman = self.frozen_stickman.copy()
                # 在火柴人上添加"打卡成功"文字
                cv2.putText(right_stickman, "Punch Success!", 
                        (right_stickman.shape[1]//2 - 150, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
            elif current_keypoints is not None and len(current_keypoints) > 0:
                # 实时显示火柴人（透明背景）
                right_stickman = self.draw_stickman(current_keypoints)
            else:
                # 没有检测到人，显示空白画布和提示
                right_stickman = np.ones((700, 500, 3), dtype=np.uint8) * 255
                cv2.putText(right_stickman, "Waiting for pose...", 
                        (right_stickman.shape[1]//2 - 140, right_stickman.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # 创建左右分屏画面
            split_screen = self.create_split_screen(display_frame, right_stickman, 
                                                   canvas_width=self.screen_width, 
                                                   canvas_height=self.screen_height)
            
            # 缩放到全屏并显示
            fullscreen_display = self.scale_to_fullscreen(split_screen)
            cv2.imshow(self.window_name, fullscreen_display)
            
            # 按'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()
        print("程序结束")
