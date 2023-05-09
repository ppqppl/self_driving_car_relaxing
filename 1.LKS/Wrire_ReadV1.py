import os
from scservo_sdk import *                    # Uses SCServo SDK library

# Control table address
ADDR_SCS_TORQUE_ENABLE     = 40
ADDR_SCS_GOAL_ACC          = 41
ADDR_SCS_GOAL_POSITION     = 42
ADDR_SCS_GOAL_SPEED        = 46
ADDR_SCS_PRESENT_POSITION  = 56

# Default setting
SCS_ID                      = 1                 # SCServo ID : 1
BAUDRATE                    = 1000000           # SCServo default baudrate : 1000000
DEVICENAME                  = '/dev/ttyUSB1'    # Check which port is being used on your controller
                                                # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

SCS_MINIMUM_POSITION_VALUE  = 100         # SCServo will rotate between this value
SCS_MAXIMUM_POSITION_VALUE  = 4000        # and this value (note that the SCServo would not move when the position value is out of movable range. Check e-manual about the range of the SCServo you use.)
SCS_MOVING_STATUS_THRESHOLD = 20          # SCServo moving status threshold
SCS_MOVING_SPEED            = 0           # SCServo moving speed
SCS_MOVING_ACC              = 50           # SCServo moving acc
protocol_end                = 0           # SCServo bit end(STS/SMS=0, SCS=1)

scs_goal_position = 1800    # Goal position

#初始化端口等项
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
    scs_comm_result, scs_error = packetHandler.write2ByteTxRx(portHandler, SCS_ID, ADDR_SCS_GOAL_POSITION, target_position)
    if scs_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(scs_comm_result))
    elif scs_error != 0:
        print("%s" % packetHandler.getRxPacketError(scs_error))



def servo_angle_read():
    global scs_present_position_speed
    scs_present_position_speed, scs_comm_result, scs_error = packetHandler.read4ByteTxRx(portHandler, SCS_ID, ADDR_SCS_PRESENT_POSITION)
    if scs_comm_result != COMM_SUCCESS:
        print(packetHandler.getTxRxResult(scs_comm_result))
    elif scs_error != 0:
        print(packetHandler.getRxPacketError(scs_error))

    scs_present_position = SCS_LOWORD(scs_present_position_speed)
    scs_present_speed = SCS_HIWORD(scs_present_position_speed)
    print("[ID:%03d] GoalPos:%03d PresPos:%03d PresSpd:%03d" 
        % (SCS_ID, scs_goal_position, scs_present_position, SCS_TOHOST(scs_present_speed, 15)))



if __name__=='__main__':
    servo_angle_write(scs_goal_position)
    while True:
        servo_angle_read()
        if not (abs(scs_goal_position - scs_present_position_speed) > SCS_MOVING_STATUS_THRESHOLD):
            break
    portHandler.closePort()
