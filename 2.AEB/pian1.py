import pyvesc
from pyvesc.VESC.messages import GetValues, SetDutyCycle, SetRPM
import serial
import time

serialport = '/dev/ttyUSB0'

def stop():
    print("start stop")
    # Stop_SetMode = SetDutyCycle(-0.1)
    # GetValueV2.get_values_example(Stop_SetMode)
    # time.sleep(0.15)
    Static_Setmode = SetDutyCycle(0)  # 停车，占空比变为0
    get_values_example(Static_Setmode)
    time.sleep(0.1)

def get_values_example(SetMode):
    with serial.Serial(serialport, baudrate=115200, timeout=0.01) as ser:
        ser.write(pyvesc.encode(SetMode))
        ser.write(pyvesc.encode_request(GetValues))
        (response, consumed) = pyvesc.decode(ser.read(78))
        if consumed == 78:
            # print(response, consumed)
            print(response.rpm)
            return response.rpm
        else:
            return 'error'


if __name__ == "__main__":
    SetRPM_Values = 1450
    # SetDutyCycle_Values = 0.1
    # SetMode = SetDutyCycle(SetDutyCycle_Values)
    SetMode = SetRPM(SetRPM_Values)
    get_values_example(SetMode)
    time.sleep(4.3)
    jjj = 1
    while True:
        if jjj == 82:
            time.sleep(4.3)
        if jjj == 96:
            break
        get_values_example(SetMode)
        jjj = jjj + 1
