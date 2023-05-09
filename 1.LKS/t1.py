# from ctypes.wintypes import ULARGE_INTEGER
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import socket
from struct import pack
import math
import time

def nothing(x):
    pass

cv.namedWindow('adjust')
cv.createTrackbar('canny_low_threshold', 'adjust', 7, 255, nothing)
cv.moveWindow('adjust', 0, 0)

def grayscale(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    cv.imshow('Gray image', gray)
    cv.moveWindow('Gray image', 390, 0)
    return gray

def canny(image):
    low_threshold = cv.getTrackbarPos('canny_low_threshold', 'adjust')
    cannyedge = cv.Canny(image, low_threshold, low_threshold * 3)
    cv.imshow('Canny image', cannyedge)
    cv.moveWindow('Canny image', 1050, 0)
    return cannyedge

def gaussian_blur(image):
    kernel_size = 3
    blur = cv.GaussianBlur(image, (kernel_size, kernel_size), 0)
    cv.imshow('Blur image', blur)
    cv.moveWindow('Blur image', 720, 0)
    return blur

def region_of_interest(image):
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]), (imshape[1], imshape[0]),
                          (imshape[1], int(imshape[0] * 2 / 3)),(0 ,int(imshape[0] * 2 / 3)) ]],
                        dtype=np.int32)
    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv.bitwise_and(image, mask)
    cv.imshow('ROI image', masked_image)
    cv.moveWindow('ROI image', 390, 300)
    return masked_image



def err_generator(err):
    err_range = 80
    uart_range = 2000
    k = float((uart_range - 1500) / err_range)
    b = uart_range - k * err_range
    uart = k * err + b
    return uart

def process_image(image):
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray)
    edge_image = canny(blur_gray)
    masked_edges = region_of_interest(edge_image)

def main():
    # cap = cv2.VideoCapture("anticlockwise_rpm500.mp4")
    cap = cv.VideoCapture("./video.mp4")
    # cap = cv.VideoCapture(0)#参数是0，表示打开笔记本的内置摄像头/调用摄像头
    width = 320
    height = 240
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)  # 设置图像宽度
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)  # 设置图像高度
    cap.set(cv.CAP_PROP_BUFFERSIZE, 0)  # 限制缓冲区
    fps = cap.get(cv.CAP_PROP_FPS)  # 给出每秒帧数
    while (cap.isOpened()):
        _, frame = cap.read()


        [processed, err] = process_image(frame)

        uart = int(err_generator(-err))

        cv.imshow('image', processed)
        cv.moveWindow('image', 720, 300)
        cv.waitKey(1)
        if cv.waitKey(1) == 27:
            break


if __name__ == '__main__':
    main()