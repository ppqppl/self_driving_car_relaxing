import time
import os
from turtle import end_fill #os模块提供的就是各种 Python 程序与操作系统进行交互的接口
import cv2
import numpy as np
from struct import pack
from timeit import default_timer as timer
import threading
from threading import Lock,Thread
import time,os

# 感兴趣区域截取
def ROI(img,left_down_x, left_down_y, left_up_x, left_up_y, right_up_x, right_up_y, right_down_x, right_down_y):
    height, width= img.shape[:2]
    mask = np.zeros_like(img)#输入为矩阵img，输出为形状和img一致的矩阵，其元素全部为0
    
    #print(left_down_x, left_down_y, left_up_x, left_up_y, right_up_x, right_up_y, right_down_x, right_down_y)
    vertices = np.array([[(left_down_x, left_down_y), (left_up_x, left_up_y),(right_up_x, right_up_y),(right_down_x, right_down_y)]], dtype=np.int32)#截取图像的四点坐标
    #创建掩膜
    ignore_mask_color = (255,255,255)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    #cv2.imshow('mask',mask)
    #cv2.moveWindow('ROI image',390,300)
    return masked_image

#透视变换	
def birdeye(img,ldx,ldy,lux,luy,rux,ruy,rdx,rdy):
    h, w = img.shape[:2]  # 取img图像的高（行）、宽（列）
    # src：源图像中待测矩形的四点坐标
    
    src = np.float32([[rdx, rdy],    # br
                      [ldx, ldy],    # bl
                      [lux, luy],   # tl
                      [rux, ruy]])  # tr

    # dst：目标图像中矩形的四点坐标
    dst = np.float32([[w, h],       # br
                      [0, h],       # bl
                      [0, 0],       # tl
                      [w, 0]])      # tr

    M = cv2.getPerspectiveTransform(src, dst)  # 计算得到转换矩阵
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)  # 实现透视变换转换
    #cv2.imshow('birdeye image',warped) 
    #cv2.moveWindow('birdeye image', 390, 0)
    return warped, Minv



#预处理
           
def preprocessing(img):
    """
    取原始图像的蓝色通道并平滑过滤    
    """
    #图像腐蚀与膨胀
    kernel1 = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    #gray = img[ :, :, 0]
    gray1 = cv2.cvtColor(img ,cv2.COLOR_RGB2GRAY)
    gray2 = cv2.medianBlur(gray1,1)#中值滤波法是一种非线性平滑技术，它将每一像素点的灰度值设置为该点某邻域窗口内的所有像素点灰度值的中值。5为方框的尺寸，必须是奇数 
    gray3 = cv2.morphologyEx(gray2, cv2.MORPH_OPEN, kernel1,iterations=3)
    #开运算(open) ,先腐蚀后膨胀的过程。开运算可以用来消除噪点，迭代3次
    gray4 = cv2.morphologyEx(gray3, cv2.MORPH_CLOSE, kernel2,iterations=2)
    #闭运算(close)，先膨胀后腐蚀的过程。迭代2次
    return gray4



#canny边缘检测
def img_canny(img):
    cannyedge = cv2.Canny(img,30,90)    #边缘检测参数
    return cannyedge


if __name__ == "__main__":    
    #cap = cv2.VideoCapture(0)#参数是0，表示打开笔记本的内置摄像头
    cap = cv2.VideoCapture('/home/wjh/Desktop/Video/zebra.webm')#打开视频文件
    #截取位置
    cut_trapezoid_up = 120  #梯形上边长
    cut_trapezoid_down = 240    #梯形下边长
    cut_trapezoid_high = 50    #梯形高
    cut_trapezoid_middle_point_x = 320/2  #梯形底中点X
    cut_trapezoid_middle_point_y = 240-10    #梯形底中点Y
    left_down_x = int(cut_trapezoid_middle_point_x - cut_trapezoid_down/2)
    left_down_y = right_down_y = int(cut_trapezoid_middle_point_y)
    left_up_x = int(cut_trapezoid_middle_point_x - cut_trapezoid_up/2)
    left_up_y = right_up_y = int(cut_trapezoid_middle_point_y - cut_trapezoid_high)
    right_up_x = int(cut_trapezoid_middle_point_x + cut_trapezoid_up/2)
    right_down_x = int(cut_trapezoid_middle_point_x + cut_trapezoid_down/2)


    while True:       
        ret, frame = cap.read()
        img = cv2.resize(frame, (320,240))        
        

        ROI_img=ROI(img,left_down_x, left_down_y, left_up_x, left_up_y, right_up_x, right_up_y, right_down_x, right_down_y)#截取感兴趣区域
        cv2.imshow('ROI_img',ROI_img)
        cv2.moveWindow('ROI_img',400,0)
        #透视变换
        birdeye_img, Minv = birdeye(ROI_img,left_down_x,left_down_y,left_up_x,left_up_y,right_up_x,right_up_y,right_down_x,right_down_y) # 进行透视变换
        cv2.imshow('birdeye_img',birdeye_img)
        cv2.moveWindow('birdeye_img',800,0)


        gray = preprocessing(birdeye_img)#图像预处理
        cv2.imshow('gray', gray)
        cv2.moveWindow('gray',0,300)

        canny = cv2.Canny(gray,30,90,apertureSize = 3)#canny算子边缘检测
        cv2.imshow('canny', canny)
        cv2.moveWindow('canny',400,300)


        #画梯形
        line_color = (255,0,0)
        line_weight = 2
        cv2.line(img,(left_down_x, left_down_y), (left_up_x, left_up_y),line_color, line_weight)
        cv2.line(img,(left_up_x, left_up_y), (right_up_x, right_up_y),line_color, line_weight)
        cv2.line(img,(right_up_x, right_up_y), (right_down_x, right_down_y),line_color, line_weight)
        cv2.line(img,(right_down_x, right_down_y), (left_down_x, left_down_y),line_color, line_weight)
        cv2.imshow("img",img)
        cv2.moveWindow('img',0,0)
        

        k = cv2.waitKey(1) & 0xFF
        if k == 27: 
           break
    cap.release()
    cv2.destroyAllWindows()
