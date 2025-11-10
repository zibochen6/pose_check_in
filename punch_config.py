
model_path = "./models/yolo11m-pose.engine"
camera_index = 0
#姿态维持时间
pose_duration = 3

pose_stable_threshold = 2

# 全屏显示（自动获取屏幕分辨率）
fullscreen_mode = True

# 默认头像路径（用户未上传头像时使用）
default_avatar_path = "icon/images.jpg"
# 关键点连接关系（用于绘制骨架）
skeleton = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # 头部
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # 手臂
    [5, 11], [6, 12], [11, 12],  # 躯干
    [11, 13], [13, 15], [12, 14], [14, 16]  # 腿部
]