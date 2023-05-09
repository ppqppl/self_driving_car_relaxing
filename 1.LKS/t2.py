#from ctypes.wintypes import ULARGE_INTEGER
import numpy as np
import cv2 as cv

def nothing(x):
    pass

cv.namedWindow('adjust')
cv.createTrackbar('canny_low_threshold', 'adjust', 7, 255, nothing)
# cv.moveWindow('adjust', 0, 0)

def grayscale(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    cv.imshow('Gray image', gray)
    # cv.moveWindow('Gray image', 390, 0)
    return gray

def canny(image):
    low_threshold = cv.getTrackbarPos('canny_low_threshold', 'adjust')
    cannyedge = cv.Canny(image, low_threshold, low_threshold * 3)
    cv.imshow('Canny image', cannyedge)
    # cv.moveWindow('Canny image', 1050, 0)
    return cannyedge

def gaussian_blur(image):
    kernel_size = 3
    blur = cv.GaussianBlur(image, (kernel_size, kernel_size), 0)
    cv.imshow('Blur image', blur)
    # cv.moveWindow('Blur image', 720, 0)
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
    # cv.moveWindow('ROI image', 390, 300)
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

    # 使用霍夫变换识别车道线
    rho = 1  # 线段以像素为单位的距离精度
    theta = np.pi / 180  # 线段以弧度为单位的角度精度
    threshold = 20  # 投票阈值
    min_line_len = 20  # 线段的最小长度，小于此长度的线段将被忽略
    max_line_gap = 300  # 线段之间的最大间隙，超过此间隙的线段将被视为不同的线段
    lines = cv.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    # 计算车道线中心点的平均值
    left_lane_x = []
    left_lane_y = []
    right_lane_x = []
    right_lane_y = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue # 避免除以零错误
        k = (y2 - y1) / (x2 - x1)
        if k < 0:
            left_lane_x.extend([x1, x2])
            left_lane_y.extend([y1, y2])
        else:
            right_lane_x.extend([x1, x2])
            right_lane_y.extend([y1, y2])
    left_center = (int(np.mean(left_lane_x)), int(np.mean(left_lane_y)))
    right_center = (int(np.mean(right_lane_x)), int(np.mean(right_lane_y)))
    center = (masked_edges.shape[1] // 2, masked_edges.shape[0] // 2)

    # 计算误差并返回处理后的图像和误差值
    err = center[0] - (left_center[0] + right_center[0]) // 2
    cv.line(image, (left_center[0], left_center[1]), (right_center[0], right_center[1]), (0, 255, 0), thickness=2)
    cv.circle(image, left_center, 5, (0, 0, 255), thickness=-1)
    cv.circle(image, right_center, 5, (0, 0, 255), thickness=-1)
    cv.circle(image, center, 5, (255, 0, 0), thickness=-1)
    return [image, err]

def main():
    cap = cv.VideoCapture("./video.mp4")
    # cap = cv.VideoCapture(1)
    width = 320
    height = 240
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 0)
    fps = cap.get(cv.CAP_PROP_FPS)
    while (cap.isOpened()):
        _, frame = cap.read()

        [processed, err] = process_image(frame)

        uart = int(err_generator(-err))

        cv.imshow('image', processed)
        # cv.moveWindow('image', 720, 300)
        cv.waitKey(1)
        if cv.waitKey(1) == 27:
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
    