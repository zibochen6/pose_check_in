import os
# 配置 Qt 插件路径，确保 Qt 能找到 xcb 插件
qt_plugin_path = os.path.expanduser('~/.local/lib/python3.10/site-packages/cv2/qt/plugins')
qt_plugin_backup = qt_plugin_path + '.disabled'
if os.path.exists(qt_plugin_backup) and not os.path.exists(qt_plugin_path):
    try:
        os.rename(qt_plugin_backup, qt_plugin_path)
        # print(f"已恢复 Qt 插件目录")
    except Exception as e:
        print(f"警告：无法恢复 Qt 插件目录: {e}")

# 设置 Qt 插件路径，确保能找到 xcb 插件
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
from model_process import *


def main():
    generator = Image_Generator()
    generator.run()



if __name__ == "__main__":
    main()