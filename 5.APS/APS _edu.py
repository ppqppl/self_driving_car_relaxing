from scservo_sdk import *  # Uses SCServo SDK library
import pyvesc
from pyvesc.VESC.messages import GetValues, SetDutyCycle, SetRPM
import serial
import time
import sys
from Ultrasonic import ultrasonic_distance
import os
#-------------------------------------------------------------------------------#
#                               全局变量                                         #
#-------------------------------------------------------------------------------#
Time1=0
Time2=0
flag=0

#-------------------------------------------------------------------------------#
#                             电机和舵机控制                                       #
#-------------------------------------------------------------------------------#
# Control table address
ADDR_SCS_TORQUE_ENABLE = 40
ADDR_SCS_GOAL_ACC = 41
ADDR_SCS_GOAL_POSITION = 42
ADDR_SCS_GOAL_SPEED = 46
ADDR_SCS_PRESENT_POSITION = 56

# Default setting
SCS_ID = 1  # SCServo ID : 1
BAUDRATE = 1000000  # SCServo default baudrate : 1000000
DEVICENAME = '/dev/ttyUSB1'  # Check which port is being used on your controller
# ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

SCS_MINIMUM_POSITION_VALUE = 100  # SCServo will rotate between this value
SCS_MAXIMUM_POSITION_VALUE = 4000  # and this value (note that the SCServo would not move when the position value is out of movable range. Check e-manual about the range of the SCServo you use.)
SCS_MOVING_STATUS_THRESHOLD = 20  # SCServo moving status threshold
SCS_MOVING_SPEED = 0  # SCServo moving speed
SCS_MOVING_ACC = 50  # SCServo moving acc
protocol_end = 0  # SCServo bit end(STS/SMS=0, SCS=1)

# 初始化端口等项
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(protocol_end)
portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)
# Write SCServo acc
scs_comm_result, scs_error = packetHandler.write1ByteTxRx(portHandler, SCS_ID, ADDR_SCS_GOAL_ACC, SCS_MOVING_ACC)
if scs_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(scs_comm_result))
elif scs_error != 0:
    print("%s" % packetHandler.getRxPacketError(scs_error))

# Write SCServo speed
scs_comm_result, scs_error = packetHandler.write2ByteTxRx(portHandler, SCS_ID, ADDR_SCS_GOAL_SPEED, SCS_MOVING_SPEED)
if scs_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(scs_comm_result))
elif scs_error != 0:
    print("%s" % packetHandler.getRxPacketError(scs_error))


def servo_angle_write(target_position):
    scs_comm_result, scs_error = packetHandler.write2ByteTxRx(portHandler, SCS_ID, ADDR_SCS_GOAL_POSITION,
                                                              target_position)
    if scs_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(scs_comm_result))
    elif scs_error != 0:
        print("%s" % packetHandler.getRxPacketError(scs_error))

def servo_angle_read():
    global scs_present_position_speed
    scs_present_position_speed, scs_comm_result, scs_error = packetHandler.read4ByteTxRx(portHandler, SCS_ID,
                                                                                         ADDR_SCS_PRESENT_POSITION)
    if scs_comm_result != COMM_SUCCESS:
        print(packetHandler.getTxRxResult(scs_comm_result))
    elif scs_error != 0:
        print(packetHandler.getRxPacketError(scs_error))

    scs_present_position = SCS_LOWORD(scs_present_position_speed)
    scs_present_speed = SCS_HIWORD(scs_present_position_speed)
    print("[ID:%03d] GoalPos:%03d PresPos:%03d PresSpd:%03d"
          % (SCS_ID, scs_goal_position, scs_present_position, SCS_TOHOST(scs_present_speed, 15)))


def get_values_example(SetMode):
    serialport = '/dev/ttyUSB0'
    with serial.Serial(serialport, baudrate=115200, timeout=0.01) as ser:
        ser.write(pyvesc.encode(SetMode))
        ser.write(pyvesc.encode_request(GetValues))
        (response, consumed) = pyvesc.decode(ser.read(78))
        # print('consumed:', consumed)
        if consumed == 78:
            # cdb = 6.17  # 传动比为6.17
            # rad = 0.033  # 半径为0.033米
            # velocity = response.rpm * 2 * 3.14 * rad / (cdb * 60)  # 速度计算公式为车轮周长乘以电机转速（小车转速以分钟计算），得出的结果还要除以60，速度单位才为米每秒
            # print(response, consumed)
            # print('电机转速= ',velocity)
            return response.rpm
        else:
            return 900

#-------------------------------------------------------------------------------#
#                                 车位识别                                        #
#-------------------------------------------------------------------------------#
def Parking_space_detection():
    global Time1
    global Time2
    global flag
    #自身车辆x方向速度获取
    selfspeed = 2
    Distant1 = ultrasonic_distance()
    time.sleep(0.2)
    Distant2 = ultrasonic_distance()
    a = Distant2 - Distant1
    b = Distant1 - Distant2
    #print('b= %f ' % (b))
    #print('Distant2 = %f ' % (Distant2))
    if a>20:
        # 距离突变检测（如果下一个状态距离比前一个状态距离大20（车宽）就记录当时的时间）
        Time1= time.time()
        # print('a = %f ' % (a))
        # print('Time1 = %f ' % (Time1))
        flag=1
    else:
        pass
    if b>20 and flag==1:
        # 距离突变检测（如果前一个状态距离比下一个状态距离大20（车宽）就记录当时的时间）
        # print('b = %f ' % (b))
        Time2= time.time()
        
        # print('Time2 = %f ' % (Time2))
        devTime=Time2-Time1
        # print('Time2-Time1 = %f ' % (devTime))
        # 计算两次突变的长度
        rpm = get_values_example(SetMode)
        selfspeed = rpm * 2 * 3.14 * 0.033 / (6.17 * 60)
        # print('selfspeed = %f ' % (selfspeed))
        Parking_length = selfspeed * devTime *100
        print('Parking_length  = %f cm' % (Parking_length))
        if (Parking_length > 50 and Parking_length < 1000):
            Parking_space_status = '1'  # 检测到水平泊车位
            print(Parking_space_status)
        elif (Parking_length > 25 and Parking_length < 50):
            Parking_space_status = '2'  # 检测到垂直泊车位
        else:
            Parking_space_status = '0'  # 未检测到车位
    else:
        Parking_space_status = '0'

    return Parking_space_status
#-------------------------------------------------------------------------------#


if __name__ == "__main__":
    while True:
        SetRPM_Values = 900
        SetMode = SetRPM(SetRPM_Values)
        scs_goal_position = 1500  # Goal position
        servo_angle_write(scs_goal_position)
        while True:
            rpm = get_values_example(SetMode)
            #velocity = rpm * 2 * 3.14 * 0.033 // (6.17 * 60)
            #print('velocity = %f ' % (velocity))
            Parking_flag = Parking_space_detection()
            print('Parking_flag:', Parking_flag)

            if Parking_flag == '1':
                print('-----------------检测到平行泊车位-------------------')                   
                
    







                portHandler.closePort()
                sys.exit(0)
                
            elif Parking_flag == '2':
                print('-----------------检测到垂直泊车位-------------------')
                
                









                portHandler.closePort()
                sys.exit(0)                
            else:
                pass
                print('未检测到车位')