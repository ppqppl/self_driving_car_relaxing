import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import time

#-------------------------------------------------------------------------------#
#                                 图像处理部分                                        #
#-------------------------------------------------------------------------------#
def nothing(x):
    pass


cv.namedWindow('adjust')
cv.createTrackbar('canny_low_threshold', 'adjust', 7, 255, nothing)

# 灰度图转换
def grayscale(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  # 将图片变灰
    cv.imshow('Gray image', gray)  # 根据图像灰度 进行缩放
    return gray


# Canny边缘检测
def canny(image):
    # low_threshold = cv.getTrackbarPos('canny_low_threshold', 'adjust')  # 产生调节板名字 low_threshold
    low_threshold = 123
    cannyedge = cv.Canny(image, low_threshold, low_threshold * 3)  # 处理过程的第一阈值，第二个阈值
    cv.imshow('Canny image', cannyedge)
    return cannyedge


# 高斯滤波
def gaussian_blur(image):
    kernel_size = 3  # 高斯滤波器大小size,奇数
    blur = cv.GaussianBlur(image, (kernel_size, kernel_size), 0)
    # kernel_size = cv.getTrackbarPos('kernel', 'adjust')
    # blur = cv.GaussianBlur(image, (3*kernel_size, 4*kernel_size), 0)
    cv.imshow('Blur image', blur)
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
    return masked_image


def hough_lines(image):
    rho = 1
    theta = np.pi / 180
    threshold = 30
    min_line_len = 40
    max_line_gap = 20
    lines = cv.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                           maxLineGap=max_line_gap)
    return lines


def process_error(lines, image):
    center = image.shape[1] // 2
    left_line_pos = []
    right_line_pos = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if (y2 - y1) / (x2 - x1) < 0:
                left_line_pos.append((x1 + x2) // 2)
            else:
                right_line_pos.append((x1 + x2) // 2)

    if not left_line_pos or not right_line_pos:
        return 0

    left_line_center = sum(left_line_pos) // len(left_line_pos)
    right_line_center = sum(right_line_pos) // len(right_line_pos)
    line_center = (left_line_center + right_line_center) // 2

    err = center - line_center
    return err


def process_image(image):
    # 灰度图转换
    gray = grayscale(image)
    # 高斯滤波
    blur_gray = gaussian_blur(gray)
    # Canny边缘检测
    edge_image = canny(blur_gray)
    # 感兴趣区域截取
    masked_edges = region_of_interest(edge_image)

    # 霍夫线变换
    lines = hough_lines(masked_edges)

    if lines is None:
        return gray, 0

    err = process_error(lines, image)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return image, err

def main(): 
      
    #cap = cv2.VideoCapture("anticlockwise_rpm500.mp4")
    cap = cv.VideoCapture(0)
    #cap = cv.VideoCapture(0)#参数是0，表示打开笔记本的内置摄像头/调用摄像头
    width = 320
    height = 240
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)  # 设置图像宽度
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)  # 设置图像高度
    cap.set(cv.CAP_PROP_BUFFERSIZE, 0)  # 限制缓冲区
    fps = cap.get(cv.CAP_PROP_FPS)  # 给出每秒帧数
    while (cap.isOpened()):
        _, frame = cap.read()


        [processed,err]= process_image(frame)


        # uart=int(err_generator(-err))


        cv.imshow('image', processed)
        cv.waitKey(1)
        if cv.waitKey(1) == 27:
            break

if __name__ == '__main__':
    main()