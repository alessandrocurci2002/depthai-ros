
import depthai as dai
import json

with dai.Device() as device:
    calib = device.readCalibration2()
    print(json.dumps(calib.eepromToJson(), indent=2))


