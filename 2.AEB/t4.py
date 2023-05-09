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
kernel = np.ones((5, 5), np.uint8)

def stop():
    print("start stop")
    # Stop_SetMode = SetDutyCycle(-0.1)
    # GetValueV2.get_values_example(Stop_SetMode)
    # time.sleep(0.15)
    Static_Setmode =  SetDutyCycle(0) #停车，占空比变为0
    GetValueV2.get_values_example(Static_Setmode)
    time.sleep(0.1)

def start():
   SetRPM_Values =1500
   SetMode = SetRPM(SetRPM_Values)
   GetValueV2.get_values_example(SetMode)

if __name__ == "__main__":

    start()

    # 读取摄像头
    cap = cv2.VideoCapture(0)

    while True:
        # 读取一帧图像
        ret, frame = cap.read()

        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 分割斑马线区域
        mask_z = cv2.inRange(hsv, lower_white, upper_white)

        # 分割车道线区域
        mask_l = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 去除噪点和填充区域
        mask_z = cv2.morphologyEx(mask_z, cv2.MORPH_OPEN, kernel)
        mask_z = cv2.morphologyEx(mask_z, cv2.MORPH_CLOSE, kernel)
        mask_l = cv2.morphologyEx(mask_l, cv2.MORPH_OPEN, kernel)
        mask_l = cv2.morphologyEx(mask_l, cv2.MORPH_CLOSE, kernel)

        # 计算斑马线和车道线区域面积
        area_z = cv2.countNonZero(mask_z)
        area_l = cv2.countNonZero(mask_l)

        # 判断是否为斑马线或车道线
        if area_z > 0:
            # 计算斑马线线段数量
            lines = cv2.HoughLinesP(mask_z, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
            if lines is not None and len(lines) == 2:
                print("1")
                stop()
                time.sleep(4.5)
                start()
            else:
                print("0")
        else:
            print("0")

        if area_l > 1000:
            print("0")
        else:
            print("1")
            stop()
            time.sleep(4.5)
            start()

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
