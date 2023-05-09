#!/usr/bin/env python3
# coding:utf-8
import Jetson.GPIO as GPIO
import time

ultrasonicTrigPin = 13  # 超声波触发引脚
ultrasonicEchoPin = 11  # 超声波信号回响引脚

GPIO.setmode(GPIO.BOARD)  # 使用物理编码，及开发板上正面的引脚编号
GPIO.setup(ultrasonicTrigPin, GPIO.OUT)  # 设置触发引脚为输出模式
GPIO.setup(ultrasonicEchoPin, GPIO.IN)  # 设置信号引脚为输出模式
EchoRisingEdgeTime, EchoFallingEdgeTime = 0.0, 0.0  # 上升沿和下降沿的触发时间


def triggerPulse():
    """
    输出10us的脉冲

    :return:
    """

    GPIO.output(ultrasonicTrigPin, GPIO.HIGH)
    time.sleep(0.0001)
    GPIO.output(ultrasonicTrigPin, GPIO.LOW)


def ultrasonicDetection() -> float:
    """
    超声波测距函数，需放置在循环中执行

    :return: float, 距离，单位cm

        0 -> 超出检测范围
    """
    global EchoRisingEdgeTime, EchoFallingEdgeTime

    triggerPulse()  # 输出10us的触发脉冲做为超声波的控制信号
    timeOutCount = 1000
    while GPIO.input(ultrasonicEchoPin) == GPIO.LOW and timeOutCount > 0:
        timeOutCount -= 1
    EchoRisingEdgeTime = time.time()
    # print(EchoRisingEdgeTime, timeOutCount)

    timeOutCount = 1000
    while GPIO.input(ultrasonicEchoPin) == GPIO.HIGH and timeOutCount > 0:
        timeOutCount -= 1
    EchoFallingEdgeTime = time.time()
    # print(EchoFallingEdgeTime, timeOutCount)
    # 计算距离
    distance = (EchoFallingEdgeTime - EchoRisingEdgeTime) * 340 / 2 * 100  # 单位cm，

    EchoRisingEdgeTime, EchoFallingEdgeTime = 0, 0

    return distance


if __name__ == "__main__":
    while True:
        Distance = ultrasonicDetection()
        print("distance: ", Distance)
        time.sleep(0.05)
