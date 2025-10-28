

MODEL_PATH = "models/yolo11m-pose.engine"
camera_index = 0

# 显示选项
show_detection_results = False  # True: 显示姿态检测结果, False: 不显示检测结果

# 打卡系统状态变量
punch_state = "waiting"  # waiting, detecting, posing, capturing, success
pose_start_time = None
pose_duration = 3.0  # 需要保持pose的秒数
last_pose_keypoints = None
pose_stable_threshold = 15.0  # 姿态稳定性阈值（像素距离）

# 定义检测区域 (ROI) - 屏幕中央区域
frame_width = 640
frame_height = 480
roi_x = int(frame_width * 0.1)  # 从10%开始
roi_y = int(frame_height * 0.1)  # 从10%开始
roi_width = int(frame_width * 0.8)  # 宽度80%
roi_height = int(frame_height * 0.9)  # 高度90%

# 关键点连接关系（用于绘制骨架）
skeleton = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # 头部
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # 手臂
    [5, 11], [6, 12], [11, 12],  # 躯干
    [11, 13], [13, 15], [12, 14], [14, 16]  # 腿部
]

# 关键点名称
keypoint_names = [
    "鼻子", "左眼", "右眼", "左耳", "右耳",
    "左肩", "右肩", "左肘", "右肘", "左腕", "右腕",
    "左髋", "右髋", "左膝", "右膝", "左踝", "右踝"
]