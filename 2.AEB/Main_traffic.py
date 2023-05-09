# import LZX_Ultrasonic_ranging_202108V1
import time
import sys
import GetValueV2
from pyvesc.VESC.messages import GetValues, SetDutyCycle, SetRPM
import serial
import re
import g2
from g2 import *
import cv2
import numpy as np

flag = 0

# 颜色阈值
lower_red = np.array([0, 150, 150])
upper_red = np.array([5, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])

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

# 形态学操作的内核
kernel = np.ones((5,5), np.uint8)

if __name__ == "__main__":
    start()
    # 读取摄像头
    cap = cv2.VideoCapture(0)

    while True:
        if flag == 0:
            start()
        elif flag == 1:
            stop()
        # 读取一帧图像
        ret, frame = cap.read()

        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 分割红色和绿色区域
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # 去除噪点和填充区域
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

        # 计算红色和绿色区域面积
        area_red = cv2.countNonZero(mask_red)
        area_green = cv2.countNonZero(mask_green)

        # 判断红绿灯颜色
        if area_red > area_green:
            print("Red: 1")
            flag = 1
            stop()
        elif area_green > area_red:
            print("Green: 0")
            flag = 0
            start()
        else:
            if flag == 0:
                start()
            elif flag == 1:
                stop()
            print("Other: 2")

        # 显示图像
        cv2.imshow("Frame", frame)
        cv2.imshow("Red", mask_red)
        cv2.imshow("Green", mask_green)

        # 按下q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()

    # 关闭所有窗口
    cv2.destroyAllWindows()
