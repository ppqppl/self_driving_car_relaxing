import cv2
import numpy as np

# 设定斑马线颜色阈值
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])

# 设定形态学操作内核
kernel = np.ones((5,5), np.uint8)

# 读取摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取一帧图像
    ret, frame = cap.read()

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 分割斑马线区域
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 去除噪点和填充区域
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 计算斑马线区域面积
    area = cv2.countNonZero(mask)

    # 判断是否为斑马线
    if area > 1000:
        print("1")
    else:
        print("0")

    # 显示图像和斑马线区域
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # 按下q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()

# 关闭所有窗口
cv2.destroyAllWindows()
