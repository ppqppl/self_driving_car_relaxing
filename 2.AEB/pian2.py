import pyvesc
from pyvesc.VESC.messages import GetValues, SetDutyCycle, SetRPM
import serial

serialport = '/dev/ttyUSB1'


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
    jjj = 1
    while True:
        if jjj == 85:
            break
        get_values_example(SetMode)
        jjj = jjj + 1


