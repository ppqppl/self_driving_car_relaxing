#from ctypes.wintypes import ULARGE_INTEGER
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import socket
from struct import pack
import math
import time
import Wrire_ReadV1

#-------------------------------------------------------------------------------#
#                                 图像处理部分                                        #
#-------------------------------------------------------------------------------#
def nothing(x):
    pass


cv.namedWindow('adjust')
cv.createTrackbar('canny_low_threshold', 'adjust', 7, 255, nothing)
cv.moveWindow('adjust', 0, 0)

# 灰度图转换
def grayscale(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  # 将图片变灰
    cv.imshow('Gray image', gray)  # 根据图像灰度 进行缩放
    cv.moveWindow('Gray image', 390, 0)  # 改变窗口的位置和尺寸  390，0窗口左上角的坐标
    return gray


# Canny边缘检测
def canny(image):
    low_threshold = cv.getTrackbarPos('canny_low_threshold', 'adjust')  # 产生调节板名字 low_threshold
    cannyedge = cv.Canny(image, low_threshold, low_threshold * 3)  # 处理过程的第一阈值，第二个阈值
    cv.imshow('Canny image', cannyedge)
    cv.moveWindow('Canny image', 1050, 0)
    return cannyedge


# 高斯滤波
def gaussian_blur(image):
    kernel_size = 3  # 高斯滤波器大小size,奇数
    blur = cv.GaussianBlur(image, (kernel_size, kernel_size), 0)
    # kernel_size = cv.getTrackbarPos('kernel', 'adjust')
    # blur = cv.GaussianBlur(image, (3*kernel_size, 4*kernel_size), 0)
    cv.imshow('Blur image', blur)
    cv.moveWindow('Blur image', 720, 0)
    return blur


# 生成即Mask掩模
def region_of_interest(image):
    imshape = image.shape  # 获取图像大小
    vertices = np.array([[(0,imshape[0]), (imshape[1], imshape[0]),
                          (imshape[1], int(imshape[0] * 2 / 3)),(0 ,int(imshape[0] * 2 / 3)) ]],
                        dtype=np.int32)
    mask = np.zeros_like(image)  # 生成图像大小一致的zeros矩

    # 填充顶点vertices中间区域
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # 填充函数
    cv.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv.bitwise_and(image, mask)
    cv.imshow('ROI image', masked_image)
    cv.moveWindow('ROI image', 390, 300)
    return masked_image

# Hough transform for line and curve detection
def hough_lines(image):
    # Convert the image to grayscale and apply Canny edge detection
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)

    # Apply Hough transform to detect lines and curves
    lines = cv.HoughLines(edges, 1, np.pi / 180, 100)

    # Draw detected lines and curves on the image
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
            cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the image with detected lines and curves
    cv.imshow('Hough lines', image)
    cv.moveWindow('Hough lines', 720, 300)
    return image

# Error detection
def error_detection(image):
    # Perform image processing to detect errors
    # ...

    # Display the image with detected errors
    cv.imshow('Error detection', image)
    cv.moveWindow('Error detection', 1050, 300)
    return image





#-------------------------------------------------------------------------------#
#                                 舵机转化                                        #
#-------------------------------------------------------------------------------#


def err_generator(err):
    err_range=80 #视觉处理的误差范围（-80，80）
    uart_range=2000 #uart转角的范围（1000 2000）
    k=float((uart_range-1500)/err_range) #换算比例
    b=uart_range-k*err_range  
    uart=k*err+b
    return uart



def process_image(image):
    # 灰度图转换
    gray = grayscale(image)
    # binary_img=img_binary(gray)#二值化图像
    # 高斯滤波
    blur_gray = gaussian_blur(gray)
    # Canny边缘检测
    edge_image = canny(blur_gray)
    #感兴趣区域截取
    masked_edges = region_of_interest(edge_image)
    
    #将截取后的图像用霍夫变换的方法识别出两条车道线，
    #将车道线的中心点与图像的中心点做误差处理，得出一个err值。
    

def main(): 
      
    #cap = cv2.VideoCapture("anticlockwise_rpm500.mp4")
    #cap = cv.VideoCapture("/media/rtech/FD26-F436/二次培训/3.动态赛/1.LKS/MP4Resize/anticlockwise_rpm700.mp4")
    cap = cv.VideoCapture(0)#参数是0，表示打开笔记本的内置摄像头/调用摄像头
    width = 320
    height = 240
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)  # 设置图像宽度
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)  # 设置图像高度
    cap.set(cv.CAP_PROP_BUFFERSIZE, 0)  # 限制缓冲区
    fps = cap.get(cv.CAP_PROP_FPS)  # 给出每秒帧数
    while (cap.isOpened()):
        _, frame = cap.read()


        [processed,err]= process_image(frame)


        uart=int(err_generator(-err))
        print("err=",err,"uart",uart)
        Wrire_ReadV1.servo_angle_write(uart)


        cv.imshow('image', processed)
        cv.moveWindow('image', 720, 300)
        cv.waitKey(1)



if __name__ == '__main__':
    main()