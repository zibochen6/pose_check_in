#!/usr/bin/env python3
"""
MediaPipe Selfie Segmentation 性能测试
适用于 Jetson 平台的实时背景去除

使用方法:
1. 安装: pip install mediapipe opencv-python
2. 运行: python test_mediapipe_segmentation.py
3. 按 'q' 退出, 's' 保存当前帧
"""

import cv2
import numpy as np
import time
import mediapipe as mp
from datetime import datetime
import os

# 创建保存目录
output_dir = "mediapipe_test"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 初始化 MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# model_selection: 
#   0 = general model (适用于所有场景，速度快)
#   1 = landscape model (适用于风景/全身，精度高但稍慢)
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

# 初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# FPS 统计
fps_counter = 0
fps_start_time = time.time()
current_fps = 0

# 处理时间统计
processing_times = []

print("="*60)
print("MediaPipe Selfie Segmentation 性能测试")
print("="*60)
print("控制:")
print("  - 按 'q' 退出")
print("  - 按 's' 保存当前帧")
print("  - 按 '1' 切换到原始图像")
print("  - 按 '2' 切换到mask显示")
print("  - 按 '3' 切换到去背景效果")
print("="*60)

display_mode = 3  # 1=原始, 2=mask, 3=去背景

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头")
        break
    
    # 翻转图像（镜像效果）
    frame = cv2.flip(frame, 1)
    
    # 记录处理开始时间
    process_start = time.time()
    
    # 转换为 RGB (MediaPipe 需要 RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 执行分割
    results = segmentation.process(rgb_frame)
    
    # 获取 mask (值范围 0.0-1.0)
    mask = results.segmentation_mask
    
    # 记录处理时间
    process_time = (time.time() - process_start) * 1000  # 转换为毫秒
    processing_times.append(process_time)
    
    # 保持最近100帧的统计
    if len(processing_times) > 100:
        processing_times.pop(0)
    
    # 创建显示图像
    if display_mode == 1:
        # 显示原始图像
        display = frame.copy()
    elif display_mode == 2:
        # 显示 mask
        mask_visual = (mask * 255).astype(np.uint8)
        display = cv2.cvtColor(mask_visual, cv2.COLOR_GRAY2BGR)
    else:
        # 去背景效果
        # 方法1: 透明背景（黑色）
        condition = np.stack((mask,) * 3, axis=-1) > 0.5
        display = np.where(condition, frame, 0).astype(np.uint8)
        
        # 方法2: 模糊背景（取消注释使用）
        # bg_image = cv2.GaussianBlur(frame, (55, 55), 0)
        # condition = np.stack((mask,) * 3, axis=-1) > 0.5
        # display = np.where(condition, frame, bg_image).astype(np.uint8)
    
    # 计算 FPS
    fps_counter += 1
    if fps_counter % 10 == 0:
        current_fps = 10 / (time.time() - fps_start_time)
        fps_start_time = time.time()
    
    # 显示信息
    avg_process_time = np.mean(processing_times)
    
    # 添加文本信息
    cv2.putText(display, f"FPS: {current_fps:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"Process: {avg_process_time:.1f}ms", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    mode_text = {1: "Original", 2: "Mask", 3: "No Background"}
    cv2.putText(display, f"Mode: {mode_text[display_mode]}", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # 显示
    cv2.imshow('MediaPipe Segmentation Test', display)
    
    # 按键处理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # 保存当前帧
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存去背景图像（带透明通道）
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask_binary
        
        filename = os.path.join(output_dir, f"nobg_{timestamp}.png")
        cv2.imwrite(filename, bgra)
        print(f"✓ 已保存: {filename}")
        
        # 同时保存 mask
        mask_filename = os.path.join(output_dir, f"mask_{timestamp}.png")
        cv2.imwrite(mask_filename, mask_binary)
        print(f"✓ 已保存: {mask_filename}")
        
    elif key == ord('1'):
        display_mode = 1
    elif key == ord('2'):
        display_mode = 2
    elif key == ord('3'):
        display_mode = 3

# 释放资源
cap.release()
cv2.destroyAllWindows()
segmentation.close()

# 打印统计信息
print("\n" + "="*60)
print("性能统计:")
print(f"  平均处理时间: {np.mean(processing_times):.2f} ms")
print(f"  最快处理时间: {np.min(processing_times):.2f} ms")
print(f"  最慢处理时间: {np.max(processing_times):.2f} ms")
print(f"  理论最大FPS: {1000/np.mean(processing_times):.1f}")
print(f"  实际FPS: {current_fps:.1f}")
print("="*60)
print("测试完成！")

