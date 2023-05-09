import time

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from struct import pack
from GetValueV2 import SetRPM, get_values_example
from Wrire_ReadV1 import servo_angle_write


def nothing(x):
    pass


# 灰度变换
def grayscale(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    cv.imshow('Gray image', gray)
    cv.moveWindow('Gray image', 390, 0)
    return gray


# canny
def canny(image):
    low_threshold = 25
    cannyedge = cv.Canny(image, low_threshold, low_threshold * 3)
    cv.imshow('Canny image', cannyedge)
    cv.moveWindow('Canny image', 1050, 0)
    return cannyedge


# 高斯滤波
def gaussian_blur(image):
    kernel_size = 11
    blur = cv.GaussianBlur(image, (kernel_size, kernel_size), 0)
    cv.imshow('Blur image', blur)
    cv.moveWindow('Blur image', 720, 0)
    return blur


# 选择符合要求的直线
pre_l = []
pre_r = []
i = 0


def select_lines(image, lines):
    global i
    global pre_l
    global pre_r
    right_y_set = []
    right_x_set = []
    right_slope_set = []

    left_y_set = []
    left_x_set = []
    left_slope_set = []

    slope_min = .3  # 斜率低阈值
    slope_max = 15.85  # 斜率高阈值
    middle_x = image.shape[1] / 2  # 图像中线x坐标
    max_y = image.shape[0]  # 最大y坐标
    if i == 0:
        i = i + 1
        for line in lines:
            for x1, y1, x2, y2 in line:
                fit = np.polyfit((x1, x2), (y1, y2), 1)  # 拟合成直线
                slope = fit[0]  # 斜率

                if slope_min < np.absolute(slope) <= slope_max:

                    # 将斜率大于0且线段X坐标在图像中线右边的点存为右边车道线
                    if slope > 0 and x1 > middle_x and x2 > middle_x:
                        right_y_set.append(y1)
                        right_y_set.append(y2)
                        right_x_set.append(x1)
                        right_x_set.append(x2)
                        right_slope_set.append(slope)

                    # 将斜率小于0且线段X坐标在图像中线左边的点存为左边车道线
                    elif slope < 0 and x1 < middle_x and x2 < middle_x:
                        left_y_set.append(y1)
                        left_y_set.append(y2)
                        left_x_set.append(x1)
                        left_x_set.append(x2)
                        left_slope_set.append(slope)
        if left_y_set:
            lindex = left_y_set.index(min(left_y_set))  # 最高点
            left_x_top = left_x_set[lindex]
            left_y_top = left_y_set[lindex]
            lslope = np.median(left_slope_set)  # 计算平均值

            # 根据斜率计算车道线与图片下方交点作为起点
            left_x_bottom = int(left_x_top + (max_y - left_y_top) / lslope)

            # 绘制线段
            if len(pre_l) == 0:
                pre_l.append(left_x_bottom)
                pre_l.append(max_y)
                pre_l.append(left_x_top)
                pre_l.append(left_y_top)

        if right_y_set:
            rindex = right_y_set.index(min(right_y_set))  # 最高点
            right_x_top = right_x_set[rindex]
            right_y_top = right_y_set[rindex]
            rslope = np.median(right_slope_set)

            # 根据斜率计算车道线与图片下方交点作为起点

            right_x_bottom = int(right_x_top + (max_y - right_y_top) / rslope)

            # 绘制线段
            if len(pre_r) == 0:
                pre_r.append(right_x_top)
                pre_r.append(right_y_top)
                pre_r.append(right_x_bottom)
                pre_r.append(max_y)
        return ((left_x_bottom, max_y, left_x_top, left_y_top), (right_x_top, right_y_top, right_x_bottom, max_y))
    else:
        for line in lines:
            for x1, y1, x2, y2 in line:
                fit = np.polyfit((x1, x2), (y1, y2), 1)  # 拟合成直线
                slope = fit[0]  # 斜率

                if slope_min < np.absolute(slope) <= slope_max:

                    # 将斜率大于0且线段X坐标在图像中线右边的点存为右边车道线
                    if slope > 0 and x1 > middle_x and x2 > middle_x:
                        if np.absolute(slope - (pre_r[3] - pre_r[1]) / (pre_r[2] - pre_r[0]) < 7):
                            if np.absolute(pre_r[0] - x1) < 300 and np.absolute(pre_r[2] - x2) < 300:
                                right_y_set.append(y1)
                                right_y_set.append(y2)
                                right_x_set.append(x1)
                                right_x_set.append(x2)
                                right_slope_set.append(slope)

                    # 将斜率小于0且线段X坐标在图像中线左边的点存为左边车道线
                    elif slope < 0 and x1 < middle_x and x2 < middle_x:
                        if np.absolute(slope - (pre_l[3] - pre_l[1]) / (pre_l[2] - pre_l[0]) < 7):
                            if np.absolute(pre_l[0] - x1) < 300 and np.absolute(pre_l[2] - x2) < 300:
                                left_y_set.append(y1)
                                left_y_set.append(y2)
                                left_x_set.append(x1)
                                left_x_set.append(x2)
                                left_slope_set.append(slope)
        if left_y_set:
            lindex = left_y_set.index(min(left_y_set))  # 最高点
            left_x_top = left_x_set[lindex]
            left_y_top = left_y_set[lindex]
            lslope = np.median(left_slope_set)  # 计算平均值

            # 根据斜率计算车道线与图片下方交点作为起点
            left_x_bottom = int(left_x_top + (max_y - left_y_top) / lslope)

            # 绘制线段
            pre_l[0] = (left_x_bottom)
            pre_l[1] = (max_y)
            pre_l[2] = (left_x_top)
            pre_l[3] = (left_y_top)
        else:
            left_x_bottom = pre_l[0]
            max_y = pre_l[1]
            left_x_top = pre_l[2]
            left_y_top = pre_l[3]
        if right_y_set:
            rindex = right_y_set.index(min(right_y_set))  # 最高点
            right_x_top = right_x_set[rindex]
            right_y_top = right_y_set[rindex]
            rslope = np.median(right_slope_set)

            # 根据斜率计算车道线与图片下方交点作为起点

            right_x_bottom = int(right_x_top + (max_y - right_y_top) / rslope)
            pre_r[0] = (right_x_top)
            pre_r[1] = (right_y_top)
            pre_r[2] = (right_x_bottom)
            pre_r[3] = (max_y)
        else:
            right_x_top = pre_r[0]
            right_y_top = pre_r[1]
            right_x_bottom = pre_r[2]
            max_y = pre_r[3]
        return ((left_x_bottom, max_y, left_x_top, left_y_top), (right_x_top, right_y_top, right_x_bottom, max_y))


# 画出两条车道线
def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    line = select_lines(image, lines)
    cv.line(image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), color, thickness)
    cv.line(image, (line[1][0], line[1][1]), (line[1][2], line[1][3]), color, thickness)
    cv.imshow("line", image)


# 感兴趣区域截取
def region_of_interest(image):
    edge = 30
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (imshape[1] - edge, imshape[0]),
                          (imshape[1] - edge, int(imshape[0] * 12 / 24)), (0, int(imshape[0] * 12 / 24))]],
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


# 车道误差转舵机方向
def err_generator(err):
    err_range = 80
    uart_range = 2000
    k = float((uart_range - 1500) / err_range)
    b = uart_range - k * err_range
    uart = k * err + b
    print(err, uart)
    return uart


# 霍夫变换
def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv.HoughLinesP(image, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


# 计算车道线交点
def cross_point(line1, line2):  # 计算交点函数
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
        x = x3
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
        y = k1 * x * 1.0 + b1 * 1.0
    elif (x2 - x1) == 0:
        k1 = None
        b1 = 0
        x = x1
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
        y = k2 * x * 1.0 + b2 * 1.0
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
        b2 = y3 * 1.0 - x3 * k2 * 1.0
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


# 计算两条车道线实际位置
def get_err(image, lines):
    line = select_lines(image, lines)
    print(line[0][2], line[1][0], (line[0][2] + line[1][0]) / 2)
    return (line[0][2] + line[1][0]) / 2
    # res =  cross_point(line[0],line[1])
    # return res[0]


# 图像处理
def process_image(image):
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray)
    edge_image = canny(blur_gray)
    masked_edges = region_of_interest(edge_image)
    cv.imshow("masked_edges", masked_edges)

    rho = 1
    theta = np.pi / 180
    threshold = 30
    min_line_len = 40
    max_line_gap = 140

    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    if lines is not None:
        draw_lines(line_img, lines)
    weighted_img = cv.addWeighted(image, 0.8, line_img, 1, 0)
    cv.imshow("weighted_img", weighted_img)

    err = -1
    if lines is not None:
        err = get_err(image, lines)
    point2 = (int(err), int(image.shape[0] / 2))
    if err != -1:
        cv.circle(weighted_img, center=point2, radius=20, color=(0, 255, 0))
    return weighted_img, err


def main():
    cap = cv.VideoCapture(0)
    width = 320
    height = 240
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)  # 设置图像宽度
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)  # 设置图像高度
    cap.set(cv.CAP_PROP_BUFFERSIZE, 0)  # 限制缓冲区
    fps = cap.get(cv.CAP_PROP_FPS)  # 给出每秒帧数
    err2 = width / 2.0
    SetMode = SetRPM(1450)
    jjj = 0
    while (True):
        _, frame = cap.read()
        [processed, err] = process_image(frame)
        if err != -1:
            err2 = err
        uart = err_generator(-(frame.shape[1] / 2 - err2))
        print(err2)
        servo_angle_write(int(uart))
        cv.imshow('image', processed)
        cv.moveWindow('image', 720, 300)
        cv.waitKey(1)
        if cv.waitKey(1) == 27:
            break
        jjj = jjj+1
        if jjj >= 5:
            get_values_example(SetMode)
        else:
            time.sleep(0.1)
        if jjj == 6:
            time.sleep(4.3)
        if jjj == 37:
            time.sleep(4.3)
        if jjj == 43:
            break


if __name__ == '__main__':
    servo_angle_write(1500)
    main()