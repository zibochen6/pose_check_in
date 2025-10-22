import cv2

# ✅ 指定设备路径而不是 index，避免 OpenCV 混乱
cap1 = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap2 = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)

# ✅ 注意这里区分格式
# Jetson 上某些 UVC 摄像头 MJPG 不稳定，可以尝试切换格式顺序
cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) 
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap1.set(cv2.CAP_PROP_FPS, 30)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap2.set(cv2.CAP_PROP_FPS, 30)
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1:
        print("❌ Error reading frame from cam1 (/dev/video0)")
        break
    if not ret2:
        print("❌ Error reading frame from cam2 (/dev/video2)")
        break

    cv2.imshow("cam1", frame1)
    cv2.imshow("cam2", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
