from ultralytics import YOLO
from datetime import datetime
from config import *
import requests
import base64
from openai import OpenAI
import threading
from queue import Queue
import mediapipe as mp
import signal
import sys
import cv2
import os
import time
import numpy as np


class Image_Generator():
    def __init__(self):
        self.detect_model = YOLO(MODEL_PATH, task="pose")
        #0=general (快速), 1=landscape (精确但稍慢)
        self.background_removal_model = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
        self.openai_client = openai_client = OpenAI(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key="0b02eee6-5201-46c3-95a8-59594aa6dc38",
                timeout=API_TIMEOUT  #添加超时设置
                )
        #窗口名称
        self.window_name = "AI Punch Clock System"
        self.screen_width = 1920  # 默认值
        self.screen_height = 1080  # 默认值
        self.check_system()

        # 异步推理队列
        self.inference_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)

        # 最后推理结果
        self.last_results = None
        self.results_lock = threading.Lock()

       
    def run(self):
        # FPS计算变量
        p_time = time.time()
        global display_state,generated_image,is_generating,generation_angle,generation_lock,program_running,cleanup_done
         # UI显示相关的全局变量
        display_state = "camera"  # "camera" 或 "result"
        generated_image = None  # 生成的动漫图片
        is_generating = False  # 是否正在生成
        generation_angle = 0  # 旋转角度
        generation_lock = threading.Lock()
        # 🔧 修复3: 添加程序运行标志
        program_running = True
        cleanup_done = False
        
        # 🔧 修复: 初始化摄像头
        cap = self.init_camera(camera_index)
        if cap is None:
            print("✗ 摄像头初始化失败，程序退出")
            return
        
        # 🔧 修复: 初始化摄像头失败计数器
        camera_fail_count = 0
        
        # 🔧 修复: 初始化状态变量
        global punch_state, pose_start_time, last_pose_keypoints
        punch_state = "waiting"
        pose_start_time = None
        last_pose_keypoints = None
        
        # 🔧 修复: 创建推理工作线程
        def inference_worker():
            while program_running:
                try:
                    frame = self.inference_queue.get(timeout=0.1)
                    if frame is None:
                        break
                    
                    # 执行YOLO推理
                    results = self.detect_model(frame, verbose=False)
                    
                    # 更新结果（线程安全）
                    with self.results_lock:
                        self.last_results = results
                except:
                    continue
        
        inference_thread = threading.Thread(target=inference_worker, daemon=True)
        inference_thread.start()
        
        while program_running:
            try:
                ret, frame = cap.read()
                
                # 🔧 修复: 改进摄像头断开处理
                if not ret:
                    camera_fail_count += 1
                    print(f"⚠️ 无法读取摄像头画面 (失败次数: {camera_fail_count}/{MAX_CAMERA_FAIL_COUNT})")
                    
                    if camera_fail_count >= MAX_CAMERA_FAIL_COUNT:
                        print("尝试重新连接摄像头...")
                        cap.release()
                        time.sleep(CAMERA_RECONNECT_DELAY)
                        
                        cap = self.init_camera(camera_index)
                        if cap is None:
                            print("✗ 摄像头重连失败，程序退出")
                            break
                        else:
                            camera_fail_count = 0
                            print("✓ 摄像头重连成功")
                    
                    time.sleep(0.1)
                    continue
                else:
                    # 成功读取，重置计数器
                    camera_fail_count = 0
                
                frame = cv2.flip(frame, 1)
                
                # 计算FPS
                c_time = time.time()
                fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
                p_time = c_time
                
                
                
                # 异步推理（非阻塞）- 直接使用原始帧
                if not self.inference_queue.full():
                    try:
                        self.inference_queue.put_nowait(frame)
                    except:
                        pass
                
                # 获取最新的推理结果（线程安全）
                with self.results_lock:
                    results = self.last_results
                
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
                        if self.is_person_in_roi(person_keypoints, roi_x, roi_y, roi_width, roi_height):
                            
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
                                pose_distance = self.calculate_pose_distance(person_keypoints, last_pose_keypoints)
                                current_time = time.time()
                                elapsed_time = current_time - pose_start_time
                                
                                if pose_distance < pose_stable_threshold:
                                    # 姿态稳定，更新计时
                                    remaining_time = pose_duration - elapsed_time
                                    
                                    if remaining_time <= 0:
                                        # 时间到，拍照并启动后台生成
                                        punch_state = "generating"
                                        person_bbox = self.get_person_bounding_box_from_detection(results)
                                        
                                        # 标记开始生成
                                        with generation_lock:
                                            is_generating = True
                                            display_state = "camera"
                                            generated_image = None
                                        
                                        # 在后台线程生成图片
                                        generation_thread = threading.Thread(
                                            target=self.generate_anime_image_async, 
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
                    final_display = self.create_blur_with_spinner(
                        display_frame, 
                        generation_angle,
                        "Generating your avatar\nPlease wait..."
                    )
                elif current_display_state == "result" and current_generated is not None and punch_state == "success":
                    # 显示结果：只显示生成的动漫图片（单窗口）
                    # 使用屏幕分辨率
                    target_width = self.screen_width
                    target_height = self.screen_height
                    
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
                        current_icon = self.blend_icons(self.red_icon, self.green_icon, progress)
                        
                        # 叠加图标
                        resized_icon = cv2.resize(current_icon, (icon_size, icon_size))
                        final_display = self.overlay_icon_with_alpha(final_display, resized_icon, icon_x, icon_y, alpha=0.9)
                
                # 显示最终画面（缩放到全屏）
                if final_display is not None:
                    # 将画面缩放并裁剪以填充整个屏幕
                    fullscreen_display = self.scale_to_fullscreen(final_display, self.screen_width, self.screen_height)
                    cv2.imshow(self.window_name, fullscreen_display)
                
                # 按'q'键退出
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    program_running = False
                    break
                
                # 确保适当的帧率
                time.sleep(0.033)  # 约30 FPS
                
            except KeyboardInterrupt:
                print("\n收到键盘中断信号...")
                program_running = False
                break
            except Exception as e:
                print(f"主循环发生异常: {e}")
                import traceback
                traceback.print_exc()
                # 继续运行，不退出

        # 🔧 修复: 改进资源释放
        print("\n正在释放资源...")

        # 停止推理线程
        print("  ├─ 停止推理线程...")
        program_running = False
        self.inference_queue.put(None)
        if inference_thread.is_alive():
            inference_thread.join(timeout=2)

        # 释放摄像头
        print("  ├─ 释放摄像头...")
        if cap is not None:
            cap.release()

        # 关闭所有OpenCV窗口
        print("  ├─ 关闭显示窗口...")
        cv2.destroyAllWindows()

        # 🔧 修复: 关闭MediaPipe模型
        print("  ├─ 释放MediaPipe模型...")
        try:
            self.background_removal_model.close()
            print("  │  └─ MediaPipe模型已关闭")
        except Exception as e:
            print(f"  │  └─ 关闭MediaPipe模型时出错: {e}")

        print("  └─ 资源释放完成")
        print("\n程序正常退出")

    def check_system(self):
        #资源加载
        self.red_icon = cv2.imread("icon/red.png", cv2.IMREAD_UNCHANGED)
        self.green_icon = cv2.imread("icon/green.png", cv2.IMREAD_UNCHANGED)
        # 确保图标加载成功
        if self.red_icon is None or self.green_icon is None:
            print("警告：无法加载图标文件！")
            self.red_icon = np.zeros((100, 100, 4), dtype=np.uint8)
            self.green_icon = np.zeros((100, 100, 4), dtype=np.uint8)
        # 创建照片保存目录
        self.photos_dir = "punch_photos"
        if not os.path.exists(self.photos_dir):
            os.makedirs(self.photos_dir)
        # 创建全屏窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
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
        self.screen_width = screen_width
        self.screen_height = screen_height
        print(f"屏幕分辨率: {self.screen_width}x{self.screen_height}")


    #模糊背景+加载图片动画
    def create_blur_with_spinner(self,frame, angle, text="Generating your avatar, please wait..."):
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
    
    def scale_to_fullscreen(self,image, target_width, target_height):
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
    
    def blend_icons(self,red_icon, green_icon, progress):
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

    def is_person_in_roi(self,keypoints, roi_x, roi_y, roi_width, roi_height):
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

    def calculate_pose_distance(self,keypoints1, keypoints2):
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


    def get_person_bounding_box_from_detection(self,results):
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


    def generate_anime_style_image(self,person_image_base64, retry_count=0):
        """调用AI生成动漫风格图片（无水印）
        
        🔧 修复: 添加重试机制和更好的异常处理
        """
        try:
            # 优化5: 简化prompt以加快API响应速度
            images_response = self.openai_client.images.generate(
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
            print(f"生成风格图片失败 (尝试 {retry_count + 1}/{MAX_API_RETRIES + 1}): {e}")
            
            # 🔧 修复: 添加重试逻辑
            if retry_count < MAX_API_RETRIES:
                print(f"  等待 2 秒后重试...")
                time.sleep(2)
                return self.generate_anime_style_image(person_image_base64, retry_count + 1)
            else:
                print(f"✗ API调用失败，已达最大重试次数")
                return None
    def generate_anime_image_async(self,frame, person_bbox):
        """后台线程生成动漫风格图片
        
        🔧 修复: 改进异常处理和状态管理
        """
        global generated_image, is_generating, display_state, generation_lock
        
        # 记录开始时间
        start_time = time.time()
        
        # 初始化性能监控变量
        bg_removal_time = 0
        api_time = 0
        
        try:
            if person_bbox is None:
                print("无法获取人的检测框")
                raise ValueError("person_bbox is None")
            
            x_min, y_min, x_max, y_max = person_bbox
            
            # 裁剪人的区域（裁切目标框）
            cropped_person = frame[y_min:y_max, x_min:x_max]
            
            if cropped_person.size == 0:
                print("检测框区域无效")
                raise ValueError("cropped_person is empty")
            
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
            results = self.background_removal_model.process(rgb_frame)
            mask = results.segmentation_mask
            
            segmentation_time = time.time() - segmentation_start
            print(f"  ├─ 抠图完成: {segmentation_time:.2f}秒")
            
            # 创建 BGRA 图像（原始颜色 + 透明通道）
            # 提高阈值以获得更彻底的背景去除
            # mask 是 0.0-1.0 的浮点数，使用阈值处理
            threshold = SEGMENTATION_THRESHOLD  # 🔧 修复: 使用配置值
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
                debug_path = os.path.join(self.photos_dir, f"debug_nobg_{timestamp}.png")
                cv2.imwrite(debug_path, bg_removed)
                print(f"  ├─ 调试图片已保存: {debug_path}")
            
            # PNG 编码
            png_level = PNG_COMPRESSION_LEVEL
            encode_param = [cv2.IMWRITE_PNG_COMPRESSION, png_level]
            _, buffer = cv2.imencode('.png', bg_removed, encode_param)
            person_image_base64 = base64.b64encode(buffer).decode('utf-8')
            print(f"  └─ 图像编码完成，大小: {len(person_image_base64)/1024:.1f}KB")
            
            # 调用AI生成动漫风格图片
            print("正在生成风格图片...")
            api_start = time.time()
            anime_image_url = self.generate_anime_style_image(person_image_base64)
            api_time = time.time() - api_start
            print(f"API调用耗时: {api_time:.2f}秒")
            
            if anime_image_url:
                print(f"风格图片URL: {anime_image_url}")
                
                # 下载并保存生成的动漫风格图片
                # 🔧 修复: 添加下载超时
                response = requests.get(anime_image_url, timeout=DOWNLOAD_TIMEOUT)
                response.raise_for_status()
                
                anime_filename = f"punch_{timestamp}.jpg"
                anime_filepath = os.path.join(self.photos_dir, anime_filename)
                
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
                    
                    # 🔧 新增: 清理旧图片
                    self.cleanup_old_images()
            else:
                # 🔧 修复: API失败时重置状态
                raise RuntimeError("API返回空URL")
            
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
            
        except Exception as e:
            # 🔧 修复: 任何异常都重置状态
            print(f"\n✗ 生成图片过程中发生错误: {e}")
            with generation_lock:
                is_generating = False
                display_state = "camera"
                generated_image = None
            print("状态已重置，系统准备接受新的打卡\n")
    

    def cleanup_old_images(self):
        """清理旧的生成图片，保留最近的几张
        
        🔧 新增: 防止磁盘占满
        """
        try:
            # 获取所有生成的图片
            files = []
            for f in os.listdir(self.photos_dir):
                if f.startswith("punch_") and f.endswith(".jpg"):
                    filepath = os.path.join(self.photos_dir, f)
                    files.append((filepath, os.path.getmtime(filepath)))
            
            # 按修改时间排序
            files.sort(key=lambda x: x[1], reverse=True)
            
            # 删除旧文件
            if len(files) > KEEP_GENERATED_IMAGES_COUNT:
                for filepath, _ in files[KEEP_GENERATED_IMAGES_COUNT:]:
                    try:
                        os.remove(filepath)
                        print(f"已删除旧图片: {filepath}")
                    except Exception as e:
                        print(f"删除旧图片失败: {e}")
        except Exception as e:
            print(f"清理旧图片时出错: {e}")
    def init_camera(self,camera_idx, max_attempts=CAMERA_RECONNECT_MAX_ATTEMPTS):
        """初始化摄像头，支持重试"""
        for attempt in range(max_attempts):
            try:
                print(f"正在初始化摄像头 (尝试 {attempt + 1}/{max_attempts})...")
                cap = cv2.VideoCapture(camera_idx)
                
                if not cap.isOpened():
                    raise RuntimeError("无法打开摄像头")
                
                # 设置MJPG格式（Motion-JPEG, compressed）
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # 测试读取
                ret, test_frame = cap.read()
                if not ret:
                    cap.release()
                    raise RuntimeError("摄像头无法读取画面")
                
                print(f"✓ 摄像头初始化成功")
                return cap
                
            except Exception as e:
                print(f"✗ 摄像头初始化失败: {e}")
                if attempt < max_attempts - 1:
                    print(f"  等待 {CAMERA_RECONNECT_DELAY} 秒后重试...")
                    time.sleep(CAMERA_RECONNECT_DELAY)
                else:
                    print(f"✗ 摄像头初始化失败，已达最大重试次数")
                    return None
        
        return None

    def encode_image_to_base64(self,image):
        """将OpenCV图像转换为base64字符串"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
