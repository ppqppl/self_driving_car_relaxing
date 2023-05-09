import RPi.GPIO as GPIO
import time

TRIG = 23
ECHO = 22

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

def ultrasonic_distance():
    GPIO.output(TRIG, 0)
    time.sleep(0.000002)
    GPIO.output(TRIG, 1)
    time.sleep(0.00001)
    GPIO.output(TRIG, 0)
    while GPIO.input(ECHO) == 0:
        pass
    emitTime =  time.time()
    while GPIO.input(ECHO) == 1:
        pass
    acceptTime = time.time()
    totalTime = acceptTime - emitTime
    time.sleep(0.01)
    distanceForReturn = int(totalTime * 340 / 2 * 100)
    if distanceForReturn>100:
        distanceForReturn=100
    else:
        pass
    # print('ultrasonic_distance = %d cm' %(distanceForReturn))
    return distanceForReturn

if __name__ == "__main__":
    while True:
        ultrasonic_distance()
    