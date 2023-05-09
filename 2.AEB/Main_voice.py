import numpy

import Python_Nano_Ultrasonic
from Python_Nano_Ultrasonic import *

# import LZX_Ultrasonic_ranging_202108V1
import time
import sys
import GetValueV2
from pyvesc.VESC.messages import GetValues, SetDutyCycle, SetRPM
import serial
import time
import re

'''
def warn():
   print("前方有障碍物")
'''


def decelerate():
    print("前方有障碍物，需要减速")
    # Stop_SetMode = SetDutyCycle(-0.16)
    # GetValueV2.get_values_example(Stop_SetMode)
    # time.sleep(0.01)
    Static_Setmode = SetDutyCycle(0.05)  # 减速，占空比改为0.05
    GetValueV2.get_values_example(Static_Setmode)
    time.sleep(0.03)


def start():
    SetRPM_Values = 1500
    SetMode = SetRPM(SetRPM_Values)
    GetValueV2.get_values_example(SetMode)


def stop():
    print("start stop")
    # Stop_SetMode = SetDutyCycle(-0.1)
    # GetValueV2.get_values_example(Stop_SetMode)
    # time.sleep(0.15)
    Static_Setmode = SetDutyCycle(0)  # 停车，占空比变为0
    GetValueV2.get_values_example(Static_Setmode)
    time.sleep(0.1)


def AEB():
    arr = []
    i = 0
    while i < 9:
        arr.append(get_value())
        i = i + 1
    arr.sort()
    ultrasonic_data = arr[4]
    print('距离:',ultrasonic_data)
    #start()
    if ultrasonic_data > 90:  # 距离大于90时，AEB不介入

        SetMode = SetRPM(1500)
        GetValueV2.get_values_example(SetMode)
        #time.sleep(0.1)
        start()

    elif ultrasonic_data < 90 and ultrasonic_data > 40:  # 距离大于60时，AEB不介入

        SetMode = SetRPM(900)
        GetValueV2.get_values_example(SetMode)
        #time.sleep(0.1)
        decelerate()

    elif ultrasonic_data <= 40 and ultrasonic_data > 0:  # 距离小于40时，完全制动
        #detection()
        stop()
        # sys.exit()
        #time.sleep(5)
        #start()


def detection():  # 检测函数
    i = 0
    i = get_value()
    while i <= 40:
        i = get_value()
        stop()
        time.sleep(3.5)
        i = 1
        if i > 40:
            start()
            break


def get_value():
    Distance = ultrasonicDetection()
    print("distance: ", Distance)
    return Distance


if __name__ == "__main__":
    # SetDutyCycle_Values = 0.1 #占空比控制电机转速，占空比为0.08
    #  SetMode = SetDutyCycle(SetDutyCycle_Values)
    # GetValueV2.get_values_example(SetMode)

    start()
    while True:
        AEB()
