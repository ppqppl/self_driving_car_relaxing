#from ctypes.wintypes import ULARGE_INTEGER
import numpy as np
import cv2 as cv

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

    # 霍夫变换检测直线
    lines = cv.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 计算偏差值
    center_x = image.shape[1] / 2
    left_line, right_line = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        if k < 0:
            left_line.append((k, b))
        else:
            right_line.append((k, b))

    left_k, left_b = np.mean(left_line, axis=0) if len(left_line) > 0 else (0, 0)
    right_k, right_b = np.mean(right_line, axis=0) if len(right_line) > 0 else (0, 0)
    center_line_x = int((left_b - right_b) / (right_k - left_k)) if right_k != left_k else int(image.shape[1] / 2)
    err = center_line_x - center_x

    return [image, err]

def main():
    cap = cv.VideoCapture("./video.mp4")
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
        cv.moveWindow('image', 720, 300)
        cv.waitKey(1)
        if cv.waitKey(1) == 27:
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()