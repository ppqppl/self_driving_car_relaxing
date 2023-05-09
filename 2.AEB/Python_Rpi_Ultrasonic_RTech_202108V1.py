import RPi.GPIO as GPIO #导入 GPIO库
import time

TRIG = 17 #控制引脚gpio17
ECHO = 27 #接收引脚gpio27
 

GPIO.setmode(GPIO.BCM)  #设置gpio模式为bcm模式
#GPIO.setwarnings(False)
GPIO.setup(TRIG, GPIO.OUT) #设置控制引脚为输出模式
GPIO.setup(ECHO, GPIO.IN) #设置接收引脚为输入模式


def ultrasonic_distance():  
    GPIO.output(TRIG, 0)  #将控制引脚设置为0
    time.sleep(0.000002) #延时2us
    GPIO.output(TRIG, 1)  #将控制引脚设置为1
    time.sleep(0.00001)  #延时10us
    GPIO.output(TRIG, 0) #将控制引脚设置为0
    
    while GPIO.input(ECHO) == 0: #开始计算时间，在循环等待接收引脚的值变为0
        pass
    emitTime = time.time() #输入变为0，开始计时，将当前时间保存到emitTime
    
    while GPIO.input(ECHO) == 1:  #循环等待接收引脚变为1,
        pass
    
    acceptTime = time.time() #输入变为1，结束计时，将将当前时间保存到acceptTime
    totalTime = acceptTime - emitTime
    #time.sleep(0.05)
    distanceForReturn = int(totalTime * 340 / 2 * 100) #计算距离
    print('ultrasonic_distance = %d cm' %(distanceForReturn)) #控制台打印距离
    return distanceForReturn
 

if __name__ == "__main__":
    while True:
        time.sleep(0.05) #循环时间为0.5s
        ultrasonic_distance()
        #GPIO.cleanup()

