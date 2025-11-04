#!/usr/bin/env python3
import cv2
import numpy as np
from rembg import remove

# 创建一个测试图像 - 纯红色
test_img = np.zeros((100, 100, 3), dtype=np.uint8)
test_img[:, :] = [0, 0, 255]  # BGR格式的红色

print("原始测试图像 - 应该是红色")
cv2.imwrite("test_original_red.png", test_img)

# 使用rembg处理
bg_removed_pil = remove(test_img)
bg_removed = np.array(bg_removed_pil)

print(f"rembg输出形状: {bg_removed.shape}")
print(f"rembg输出数据类型: {bg_removed.dtype}")

# 保存rembg原始输出
if len(bg_removed.shape) == 3 and bg_removed.shape[2] == 4:
    # RGBA格式，先保存RGB部分
    rgb_only = bg_removed[:, :, :3]
    cv2.imwrite("test_rembg_rgb.png", cv2.cvtColor(rgb_only, cv2.COLOR_RGB2BGR))
elif len(bg_removed.shape) == 3 and bg_removed.shape[2] == 3:
    # RGB格式
    cv2.imwrite("test_rembg_rgb.png", cv2.cvtColor(bg_removed, cv2.COLOR_RGB2BGR))

# 正确的颜色转换
if len(bg_removed.shape) == 3:
    if bg_removed.shape[2] == 4:  # RGBA格式
        alpha = bg_removed[:, :, 3]
        rgb = bg_removed[:, :, :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        final = np.dstack([bgr, alpha])
    elif bg_removed.shape[2] == 3:  # RGB格式
        final = cv2.cvtColor(bg_removed, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite("test_final.png", final)
    print("处理完成，请检查test_*.png文件")
