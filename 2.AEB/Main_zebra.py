import cv2
import numpy as np

# import LZX_Ultrasonic_ranging_202108V1
import time
import sys
import GetValueV2
from pyvesc.VESC.messages import GetValues, SetDutyCycle, SetRPM
import serial
import time
import re
import g2
from g2 import *
# 设定斑马线颜色阈值
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])

# 设定车道线颜色阈值
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# 设定形态学操作内核
kernel = np.ones((5,5), np.uint8)

flag = 0

def stop():
    print("start stop")
    # Stop_SetMode = SetDutyCycle(-0.1)
    # GetValueV2.get_values_example(Stop_SetMode)
    # time.sleep(0.15)
    Static_Setmode = SetDutyCycle(0)  # 停车，占空比变为0
    GetValueV2.get_values_example(Static_Setmode)
    time.sleep(0.1)


def start():
    SetRPM_Values = 1500
    SetMode = SetRPM(SetRPM_Values)
    GetValueV2.get_values_example(SetMode)

if __name__ == "__main__":

    # 读取摄像头
    cap = cv2.VideoCapture(0)

    while True:

        if flag == 0:
            start()
        elif flag == 0:
            stop()
        # 读取一帧图像
        ret, frame = cap.read()

        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测边缘
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # 分割斑马线区域
        mask_z = cv2.inRange(frame, lower_white, upper_white)

        # 分割车道线区域
        mask_l = cv2.inRange(frame, lower_yellow, upper_yellow)

        # 去除噪点和填充区域
        mask_z = cv2.morphologyEx(mask_z, cv2.MORPH_OPEN, kernel)
        mask_z = cv2.morphologyEx(mask_z, cv2.MORPH_CLOSE, kernel)
        mask_l = cv2.morphologyEx(mask_l, cv2.MORPH_OPEN, kernel)
        mask_l = cv2.morphologyEx(mask_l, cv2.MORPH_CLOSE, kernel)

        # 检测车道线和斑马线
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

        zebra_lines = []
        lane_lines = []

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                # 计算直线斜率和截距
                k = (y2 - y1) / (x2 - x1 + 1e-5)
                b = y1 - k * x1

                # 判断车道线和斑马线
                if abs(k) > 0.5:
                    if b > 0 and x1 > frame.shape[1]/2 and x2 > frame.shape[1]/2:
                        lane_lines.append((k, b))
                    elif b < 0 and x1 < frame.shape[1]/2 and x2 < frame.shape[1]/2:
                        lane_lines.append((k, b))
                else:
                    zebra_lines.append((k, b))

        # 判断是否为斑马线或车道线
        if len(zebra_lines) == 2:
            print("1")
        elif len(zebra_lines) > 2:
            print("0")

        if len(lane_lines) > 0:
            print("0")
        else:
            print("1")
            flag = 1
            stop()
            time.sleep(4.5)
            start()
            falg = 0
            time.sleep(1.5)

        # 显示图像和斑马线、车道线区域
        cv2.imshow("Frame", frame)
        cv2.imshow("Zebra", mask_z)
        cv2.imshow("Lane", mask_l)

        # 按下q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()

    # 关闭所有窗口
    cv2.destroyAllWindows()