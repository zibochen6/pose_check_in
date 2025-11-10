

MODEL_PATH = "models/yolo11m-pose.engine"
camera_index = 0



# 全局配置参数 =============
# 网络请求超时配置（秒）
API_TIMEOUT = 30  # API调用超时
DOWNLOAD_TIMEOUT = 15  # 图片下载超时
MAX_API_RETRIES = 2  # API失败最大重试次数

# 摄像头重连配置
CAMERA_RECONNECT_DELAY = 2  # 重连延迟（秒）
CAMERA_RECONNECT_MAX_ATTEMPTS = 10  # 最大重连次数
CAMERA_READ_TIMEOUT = 3  # 读取超时（秒）

# 资源管理
KEEP_GENERATED_IMAGES_COUNT = 10  # 最多保留的生成图片数量（防止磁盘占满）

# GUI 支持
HAVE_GUI = True
#摄像头断开计数器
camera_fail_count = 0
MAX_CAMERA_FAIL_COUNT = 5

# 显示选项
show_detection_results = False  # True: 显示姿态检测结果, False: 不显示检测结果

# ===== 性能优化配置 =====
# 背景去除: 使用 MediaPipe Selfie Segmentation（实时高效，60+ FPS）
SEGMENTATION_THRESHOLD = 0.7  # 抠图阈值 0.0-1.0，越高背景去除越彻底。推荐: 0.5(保留更多)/0.7(平衡)/0.9(彻底)

# PNG压缩参数
PNG_COMPRESSION_LEVEL = 1  # PNG压缩级别 0-9，越小越快但文件越大。推荐: 1-3

# AI生图优化参数
AI_IMAGE_SIZE = "1k"  # 可选: "512"(最快), "1k"(推荐), "2k"(高质量但慢)

# 调试选项
SAVE_DEBUG_IMAGES = False  # True: 保存去背景图片用于调试, False: 不保存（隐私保护）

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