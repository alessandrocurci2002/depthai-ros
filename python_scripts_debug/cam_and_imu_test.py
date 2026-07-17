#!/usr/bin/env python3

"""
Use 'T' to trigger autofocus, 'IOKL,.'
for manual exposure/focus:
  Control:      key[dec/inc]  min..max
  exposure time:     I   O      1..33000 [us]
  sensitivity iso:   K   L    100..1600
  focus:             ,   .      0..255 [far..near]
To go back to auto controls:
  'E' - autoexposure
  'F' - autofocus (continuous)

Other controls:
'1' - AWB lock (true / false)
'2' - AE lock (true / false)
'3' - Select control: AWB mode
'4' - Select control: AE compensation
'5' - Select control: anti-banding/flicker mode
'6' - Select control: effect mode
'7' - Select control: brightness
'8' - Select control: contrast
'9' - Select control: saturation
'0' - Select control: sharpness
'[' - Select control: luma denoise
']' - Select control: chroma denoise
BACKSLASH - Select control: scene mode
';' - Select control: control mode
QUOTE - Select control: capture intent
'*' - Select control: manual white balance
'a' 'd' - Increase/decrease dot projector intensity
'w' 's' - Increase/decrease flood LED intensity

For the 'Select control: ...' options, use these keys to modify the value:
  '-' or '_' to decrease
  '+' or '=' to increase

'/' to toggle printing camera settings: exposure, ISO, lens position, color temperature

IMU:
The script also enables the onboard IMU (accelerometer + gyroscope) and prints
the real (instantaneous and moving-average) sampling frequency for both sensors,
computed from the delta between consecutive device timestamps.
"""

import depthai as dai
import os

import cv2
import numpy as np
import argparse
import ast
import collections
import time
from pathlib import Path
import sys
import signal


cam_socket_opts = {
    'rgb': dai.CameraBoardSocket.CAM_A,
    'left': dai.CameraBoardSocket.CAM_B,
    'right': dai.CameraBoardSocket.CAM_C,
    'cama': dai.CameraBoardSocket.CAM_A,
    'camb': dai.CameraBoardSocket.CAM_B,
    'camc': dai.CameraBoardSocket.CAM_C,
    'camd': dai.CameraBoardSocket.CAM_D,
    'came': dai.CameraBoardSocket.CAM_E,
}
ALL_SOCKETS = list(cam_socket_opts.keys())
DEPTH_STREAM_NAME = "stereo_depth"
DEFAULT_RESOLUTION = (1280, 800)


def unpackRaw10(rawData, width, height, stride=None):
    if stride is None:
        stride = width * 10 // 8
    expectedSize = stride * height

    if len(rawData) < expectedSize:
        raise ValueError("Data too small: {} bytes, expected {}".format(len(rawData), expectedSize))

    packedData = np.frombuffer(rawData, dtype=np.uint8)
    result = np.zeros((height, width), dtype=np.uint16)

    for row in range(height):
        rowStart = row * stride
        rowData = packedData[rowStart:rowStart + stride]
        numGroups = (width + 3) // 4
        rowBytes = numGroups * 5
        if len(rowData) < rowBytes:
            break

        rowPacked = rowData[:rowBytes].reshape(-1, 5)
        rowUnpacked = np.zeros((rowPacked.shape[0], 4), dtype=np.uint16)

        rowUnpacked[:, 0] = rowPacked[:, 0].astype(np.uint16) << 2
        rowUnpacked[:, 1] = rowPacked[:, 1].astype(np.uint16) << 2
        rowUnpacked[:, 2] = rowPacked[:, 2].astype(np.uint16) << 2
        rowUnpacked[:, 3] = rowPacked[:, 3].astype(np.uint16) << 2

        rowUnpacked[:, 0] |= (rowPacked[:, 4] & 0b00000011)
        rowUnpacked[:, 1] |= (rowPacked[:, 4] & 0b00001100) >> 2
        rowUnpacked[:, 2] |= (rowPacked[:, 4] & 0b00110000) >> 4
        rowUnpacked[:, 3] |= (rowPacked[:, 4] & 0b11000000) >> 6

        rowFlat = rowUnpacked.flatten()
        result[row, :width] = rowFlat[:width]

    result16bit = (result * 64).astype(np.uint16)
    return result16bit


def socket_type_pair(arg):
    socket, type_ = arg.split(',')
    socket = socket.strip()
    if not (socket in ALL_SOCKETS):
        raise ValueError("")
    if not (type_ in ['m', 'mono', 'c', 'color', 't', 'tof', 'th', 'thermal']):
        raise ValueError("")
    is_color = True if type_ in ['c', 'color'] else False
    is_tof = True if type_ in ['t', 'tof'] else False
    is_thermal = True if type_ in ['th', 'thermal'] else False
    return [socket, is_color, is_tof, is_thermal]


def camera_tuning_item(arg):
    if ',' not in arg:
        return ("__all__", Path(arg))
    socket, path_str = arg.split(',', 1)
    if socket not in ALL_SOCKETS:
        raise argparse.ArgumentTypeError("Unknown camera socket: {}".format(socket))
    return (socket, Path(path_str))


def resolution_tuple(arg):
    try:
        value = ast.literal_eval(arg)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Resolution must be a tuple like (1280, 800)")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise argparse.ArgumentTypeError("Resolution must contain width and height")
    try:
        width = int(value[0])
        height = int(value[1])
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("Resolution values must be integers")
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution values must be positive")
    return (width, height)


def resolution_entry(arg):
    if ':' in arg:
        socket, res = arg.split(':', 1)
        socket_name = socket.strip()
        if socket_name not in ALL_SOCKETS:
            raise argparse.ArgumentTypeError(
                "Invalid socket '{}'. Use one of: {}.".format(socket_name, ', '.join(ALL_SOCKETS)))
        return socket_name, resolution_tuple(res.strip())
    return None, resolution_tuple(arg)


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-cams', '--cameras', type=socket_type_pair, nargs='+',
                    default=[],
                    help="Which camera sockets to enable, and type: c[olor] / m[ono] / t[of] / th[ermal].")
parser.add_argument('-res', '--resolution', type=resolution_entry, action='append', default=[],
                    metavar='[socket:](width,height)',
                    help="Select camera resolution as a tuple (width, height).")
parser.add_argument('-rot', '--rotate', const='all', choices={'all', 'rgb', 'mono'}, nargs="?",
                    help="Which cameras to rotate 180 degrees. All if not filtered")
parser.add_argument('-fps', '--fps', type=float, default=30,
                    help="FPS to set for all cameras")
parser.add_argument('-isp3afps', '--isp3afps', type=int, default=0,
                    help="3A FPS to set for all cameras")
parser.add_argument('-ds', '--isp-downscale', default=1, type=int,
                    help="Downscale the ISP output by this factor")
parser.add_argument('-rs', '--resizable-windows', action='store_true',
                    help="Make OpenCV windows resizable. Note: may introduce some artifacts")
parser.add_argument('-tun', '--camera-tuning', type=camera_tuning_item, nargs='+', default=[],
                    help="Camera tuning database. Single PATH, or SOCKET,PATH pairs.")
parser.add_argument('-raw', '--enable-raw', default=False, action="store_true",
                    help='Enable the RAW camera streams')
parser.add_argument('-tofraw', '--tof-raw', action='store_true',
                    help="Show just ToF raw output instead of post-processed depth")
parser.add_argument('-tofamp', '--tof-amplitude', action='store_true',
                    help="Show also ToF amplitude output alongside depth")
parser.add_argument('-tofcm', '--tof-cm', action='store_true',
                    help="Show ToF depth output in centimeters, capped to 255")
parser.add_argument('-tofmedian', '--tof-median', choices=[0, 3, 5, 7], default=5, type=int,
                    help="ToF median filter kernel size")
parser.add_argument('-rgbprev', '--rgb-preview', action='store_true',
                    help="Show RGB preview stream instead of full size isp")
parser.add_argument('-show', '--show-meta', action='store_true',
                    help="List frame metadata (seqno, timestamp, exp, iso etc).")

parser.add_argument('-d', '--device', default="", type=str,
                    help="Optional MX ID of the device to connect to.")

parser.add_argument('-ctimeout', '--connection-timeout', default=30000,
                    help="Connection timeout in ms.")

parser.add_argument('-btimeout', '--boot-timeout', default=30000,
                    help="Boot timeout in ms.")

parser.add_argument("-stereo", action="store_true", default=False,
                    help="Create a stereo depth node if the device has a stereo pair.")

parser.add_argument("-gui", action="store_true",
                    help="Use GUI instead of CLI")
parser.add_argument("-h", "--help", action="store_true", default=False,
                    help="Show this help message and exit")

parser.add_argument('-imu', '--enable-imu', action='store_true', default=False,
                    help="Enable IMU (accelerometer + gyroscope) and print real sampling frequency")
parser.add_argument('-imu-acc-hz', '--imu-accel-hz', type=int, default=480,
                    help="Requested accelerometer rate in Hz (default: %(default)s)")
parser.add_argument('-imu-gyro-hz', '--imu-gyro-hz', type=int, default=400,
                    help="Requested gyroscope rate in Hz (default: %(default)s)")

args = parser.parse_args()


resolution_default = DEFAULT_RESOLUTION
socket_resolution_overrides = {}
for socket, resolution in args.resolution:
    if socket:
        socket_resolution_overrides[socket] = resolution
    else:
        resolution_default = resolution


def get_socket_resolution(socket):
    return socket_resolution_overrides.get(socket, resolution_default)


os.environ["DEPTHAI_CONNECTION_TIMEOUT"] = str(args.connection_timeout)
os.environ["DEPTHAI_BOOT_TIMEOUT"] = str(args.boot_timeout)


if args.help:
    parser.print_help()
    exit(0)


if args.gui:
    import cam_test_gui
    cam_test_gui.main()


print("DepthAI version:", dai.__version__)
print("DepthAI path:", dai.__file__)


rotate = {
    'rgb': args.rotate in ['all', 'rgb'],
    'left': args.rotate in ['all', 'mono'],
    'right': args.rotate in ['all', 'mono'],
    'cama': args.rotate in ['all', 'rgb'],
    'camb': args.rotate in ['all', 'mono'],
    'camc': args.rotate in ['all', 'mono'],
    'camd': args.rotate in ['all', 'rgb'],
    'came': args.rotate in ['all', 'mono'],
}


def clamp(num, v0, v1):
    return max(v0, min(num, v1))


def timeDeltaToMilliS(delta):
    return delta.total_seconds() * 1000.0


class FPS:
    def __init__(self, window_size=30):
        self.dq = collections.deque(maxlen=window_size)
        self.fps = 0

    def update(self, timestamp=None):
        if timestamp is None:
            timestamp = time.monotonic()
        count = len(self.dq)
        if count > 0:
            self.fps = count / (timestamp - self.dq[0])
        self.dq.append(timestamp)

    def get(self):
        return self.fps


class IMUFreq:
    """
    Tracks the real (instantaneous) sampling frequency of an IMU sensor
    stream, computed from the delta between consecutive device timestamps.
    Also keeps a moving-average estimate for a more stable reading.
    """
    def __init__(self, window_size=30):
        self.prev_ts = None
        self.inst_freq = None
        self.dq = collections.deque(maxlen=window_size)

    def update(self, ts):
        if self.prev_ts is not None:
            dt_ms = timeDeltaToMilliS(ts - self.prev_ts)
            if dt_ms > 0:
                self.inst_freq = 1000.0 / dt_ms
                self.dq.append(dt_ms)
        self.prev_ts = ts

    def get_instant(self):
        return self.inst_freq

    def get_average(self):
        if len(self.dq) == 0:
            return None
        mean_dt_ms = sum(self.dq) / len(self.dq)
        if mean_dt_ms <= 0:
            return None
        return 1000.0 / mean_dt_ms


class Cycle:
    def __init__(self, enum_type, start_item=None):
        self.items = [item for name, item in vars(enum_type).items() if name.isupper()]
        self.index = self.items.index(start_item) if start_item else 0

    def step(self, n):
        self.index = (self.index + n) % len(self.items)
        return self.items[self.index]

    def next(self):
        return self.step(1)

    def prev(self):
        return self.step(-1)


def exit_cleanly(signum, frame):
    print("Exiting cleanly")
    cv2.destroyAllWindows()
    sys.exit(0)


def socket_to_socket_opt(socket):
    return str(socket).split('.')[-1].replace("_", "").lower()


signal.signal(signal.SIGINT, exit_cleanly)


dai_device_args = []
if args.device:
    success, device_info = dai.Device.getDeviceById(args.device)
    if success:
        dai_device_args.append(device_info)


with dai.Pipeline(dai.Device(*dai_device_args)) as pipeline:
    cam_list = []
    cam_type_color = {}
    cam_type_tof = {}
    cam_type_thermal = {}

    # ---- IMU node (accelerometer + gyroscope, real-frequency test) ----
    imu = None
    imuQueue = None
    accel_freq = None
    gyro_freq = None
    if args.enable_imu:
        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_UNCALIBRATED, args.imu_accel_hz)
        imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_UNCALIBRATED, args.imu_gyro_hz)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)
        accel_freq = IMUFreq()
        gyro_freq = IMUFreq()

    device = pipeline.getDefaultDevice()
    if not args.cameras:
        connected_cameras = device.getConnectedCameraFeatures()
        args.cameras = [(socket_to_socket_opt(cam.socket), cam.supportedTypes[0] ==
                         dai.CameraSensorType.COLOR, cam.supportedTypes[0] == dai.CameraSensorType.TOF,
                         cam.supportedTypes[0] == dai.CameraSensorType.THERMAL) for cam in connected_cameras]
        if not args.cameras:
            print("No cameras found!")
            exit(1)

    print("Enabled cameras:")
    for socket, is_color, is_tof, is_thermal in args.cameras:
        cam_list.append(socket)
        cam_type_color[socket] = is_color
        cam_type_tof[socket] = is_tof
        cam_type_thermal[socket] = is_thermal
        print(socket.rjust(7), ':', 'tof' if is_tof else 'color' if is_color else 'thermal' if is_thermal else 'mono')

    cam = {}
    tof = {}
    xout = {}
    control_queues = []
    streams = []
    for c in cam_list:
        print("CAM: ", c)
        tofEnableRaw = False
        if cam_type_tof[c]:
            tof[c] = pipeline.create(dai.node.ToF).build(cam_socket_opts[c])
            tof_name = c
            xout[tof_name] = tof[c].depth
            streams.append(tof_name)

            inTofConfig = tof[c].tofBaseInputConfig.createInputQueue()
            tofConfig = dai.ToFConfig()

            filter_ = dai.filters.params.MedianFilter.MEDIAN_OFF
            if args.tof_median == 3:
                filter_ = dai.filters.params.MedianFilter.KERNEL_3x3
            elif args.tof_median == 5:
                filter_ = dai.filters.params.MedianFilter.KERNEL_5x5
            elif args.tof_median == 7:
                filter_ = dai.filters.params.MedianFilter.KERNEL_7x7
            tofConfig.setMedianFilter(filter_)
            tof[c].setInitialConfig(tofConfig)
            if args.tof_amplitude:
                amp_name = 'tof_amplitude_' + c
                xout[amp_name] = tof[c].amplitude
                streams.append(amp_name)
        elif cam_type_thermal[c]:
            cam[c] = pipeline.create(dai.node.Thermal).build(cam_socket_opts[c])
            thermal_name = 'thermal_temperature_' + c
            xout[thermal_name] = cam[c].temperature
            thermal_color_name = c
            xout[thermal_color_name] = cam[c].color
            streams.append(thermal_color_name)
        else:
            cam[c] = pipeline.create(dai.node.Camera).build(cam_socket_opts[c])
            cap = dai.ImgFrameCapability()
            cap.size.fixed(get_socket_resolution(c))
            cap.fps.fixed(args.fps)
            stream_name = c
            xout[stream_name] = cam[c].requestOutput(cap, True)
            control_queues.append(cam[c].inputControl.createInputQueue())
            streams.append(stream_name)
            if args.enable_raw or tofEnableRaw:
                streamName = 'raw_' + c
                xout[streamName] = cam[c].raw
                streams.append(streamName)

    if args.camera_tuning:
        if len(args.camera_tuning) == 1 and args.camera_tuning[0][0] == "__all__":
            tuning_path = args.camera_tuning[0][1]
            print('Applying camera tuning from {} to all cameras'.format(tuning_path))
            pipeline.setCameraTuningBlobPath(str(tuning_path))
        else:
            for socket, tuning_path in args.camera_tuning:
                if socket == "__all__":
                    print('Cannot apply all tuning to all and specific sockets at the same time.')
                    exit(1)
                cam_socket = cam_socket_opts[socket]
                print('Applying camera tuning from {} to camera socket {}'.format(tuning_path, socket))
                pipeline.setCameraTuningBlobPath(cam_socket, str(tuning_path))

    stereo = None

    if args.stereo:
        try:
            try:
                calib = device.readCalibration2()
            except Exception:
                raise Exception("Device is not calibrated.")
            eeprom = calib.getEepromData()
            left, right = eeprom.stereoRectificationData.leftCameraSocket, eeprom.stereoRectificationData.rightCameraSocket
            left_sock_opt = socket_to_socket_opt(left)
            right_sock_opt = socket_to_socket_opt(right)
            left_cam = cam.get(left_sock_opt, None)
            right_cam = cam.get(right_sock_opt, None)
            if not left_cam:
                if left == dai.CameraBoardSocket.CAM_A:
                    left_sock_opt = "rgb"
                elif left == dai.CameraBoardSocket.CAM_B:
                    left_sock_opt = "left"
                elif left == dai.CameraBoardSocket.CAM_C:
                    left_sock_opt = "right"
                left_cam = cam.get(left_sock_opt, None)
            if not right_cam:
                if right == dai.CameraBoardSocket.CAM_A:
                    right_sock_opt = "rgb"
                elif right == dai.CameraBoardSocket.CAM_B:
                    right_sock_opt = "left"
                elif right == dai.CameraBoardSocket.CAM_C:
                    right_sock_opt = "right"
                right_cam = cam.get(right_sock_opt, None)

            if left_cam and right_cam:
                cam_features = device.getConnectedCameraFeatures()
                left_cam_features = next(filter(lambda c: c.socket == left, cam_features))
                right_cam_features = next(filter(lambda c: c.socket == right, cam_features))
                if left_cam_features.width > 1280:
                    if args.isp_downscale == 1:
                        raise Exception("Can't create stereo depth with left cam width > 1280. Use --isp-downscale.")
                if right_cam_features.width > 1280:
                    if args.isp_downscale == 1:
                        raise Exception("Can't create stereo depth with right cam width > 1280. Use --isp-downscale.")

                print("Device is calibrated and has a stereo pair, creating StereoDepth node.")
                stereo = pipeline.create(dai.node.StereoDepth)
                stereo.setLeftRightCheck(True)
                stereo.setSubpixel(True)
                left_cam.requestFullResolutionOutput().link(stereo.left)
                right_cam.requestFullResolutionOutput().link(stereo.right)
                xout[DEPTH_STREAM_NAME] = stereo.disparity
                streams.append(DEPTH_STREAM_NAME)
            else:
                print("Couldn't create stereo depth node. Device has invalid calibration.")
        except Exception as e:
            print("Couldn't create depth:", e)

    print('Connected cameras:')
    cam_name = {}
    for p in device.getConnectedCameraFeatures():
        print(' -socket {:6}: {:6} {:4} x {:4} focus:'.format(p.socket.name, p.sensorName, p.width, p.height), end='')
        print('auto ' if p.hasAutofocus else 'fixed', '- ', end='')
        print(*[t.name for t in p.supportedTypes])
        cam_name[p.socket.name] = p.sensorName
        if args.enable_raw:
            cam_name['raw_' + p.socket.name] = p.sensorName
        if args.tof_amplitude:
            cam_name['tof_amplitude_' + p.socket.name] = p.sensorName

    print('USB speed:', device.getUsbSpeed().name)

    q = {}
    fps_host = {}
    fps_capt = {}
    for c in streams:
        q[c] = xout[c].createOutputQueue(maxSize=4, blocking=False)
        if args.resizable_windows:
            cv2.namedWindow(c, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(c, (640, 480))
        fps_host[c] = FPS()
        fps_capt[c] = FPS()

    if args.enable_imu:
        imuQueue = imu.out.createOutputQueue(maxSize=50, blocking=False)

    pipeline.start()

    EXP_STEP = 500
    ISO_STEP = 50
    LENS_STEP = 1 / 1024
    DOT_STEP = 0.05
    FLOOD_STEP = 0.05
    DOT_MAX = 1
    FLOOD_MAX = 1

    lensPos = 0.59
    lensMin = 0.0
    lensMax = 1.0

    expTime = 20000
    expMin = 1
    expMax = 33000

    sensIso = 800
    sensMin = 100
    sensMax = 1600

    dotIntensity = 0
    floodIntensity = 0

    expTimeLimit = 700

    awb_mode = Cycle(dai.CameraControl.AutoWhiteBalanceMode)
    anti_banding_mode = Cycle(dai.CameraControl.AntiBandingMode)
    effect_mode = Cycle(dai.CameraControl.EffectMode)
    scene_mode = Cycle(dai.CameraControl.SceneMode)
    control_mode = Cycle(dai.CameraControl.ControlMode)
    capture_intent = Cycle(dai.CameraControl.CaptureIntent)

    def controlQueueSend(ctrl):
        for queue in control_queues:
            queue.send(ctrl)

    ae_comp = 0
    ae_lock = False
    awb_lock = False
    saturation = 0
    contrast = 0
    brightness = 0
    sharpness = 0
    luma_denoise = 0
    chroma_denoise = 0
    wb_manual = 5500
    control = 'none'
    show = args.show_meta

    jet_custom = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    jet_custom[0] = [0, 0, 0]

    print("Cam:", *['     ' + c.ljust(8) for c in cam_list], "[host | capture timestamp]")
    if args.enable_imu:
        print("IMU enabled: accelerometer @ {} Hz (requested), gyroscope @ {} Hz (requested)".format(
            args.imu_accel_hz, args.imu_gyro_hz))

    capture_list = []
    needs_newline = True
    while True:
        if args.enable_imu and imuQueue is not None:
            imuData = imuQueue.tryGet()
            if imuData is not None:
                assert isinstance(imuData, dai.IMUData)
                for imuPacket in imuData.packets:
                    acceleroValues = imuPacket.acceleroMeter
                    gyroValues = imuPacket.gyroscope

                    accel_ts = acceleroValues.getTimestamp()
                    gyro_ts = gyroValues.getTimestamp()

                    accel_freq.update(accel_ts)
                    gyro_freq.update(gyro_ts)

        for c in streams:
            try:
                pkt = q[c].tryGet()
                if pkt is not None:
                    fps_host[c].update()
                    fps_capt[c].update(pkt.getTimestamp().total_seconds())
                    width, height = pkt.getWidth(), pkt.getHeight()
                    capture = c in capture_list
                    if c.startswith('raw_') or c.startswith('tof_amplitude_'):
                        dataRaw = pkt.getData()
                        frame = unpackRaw10(dataRaw, pkt.getWidth(), pkt.getHeight(), pkt.getStride())
                    else:
                        frame = pkt.getCvFrame()
                    cam_skt = c.split('_')[-1]

                    if c == DEPTH_STREAM_NAME and stereo is not None:
                        maxDisp = stereo.initialConfig.getMaxDisparity()
                        disp = (pkt.getCvFrame() * (255.0 / maxDisp)).astype(np.uint8)
                        disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
                        cv2.imshow(c, disp)
                        continue

                    if cam_type_tof.get(cam_skt, None) and not (c.startswith('raw_') or c.startswith('tof_amplitude_')):
                        if args.tof_cm:
                            frame = (frame // 10).clip(0, 255).astype(np.uint8)
                        else:
                            frame = (frame.view(np.int16).astype(float))
                            frame = cv2.normalize(frame, frame, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            frame = cv2.applyColorMap(frame, jet_custom)
                    elif cam_type_thermal[cam_skt] and c.startswith('cam'):
                        frame = frame.astype(np.float32)
                        frame = cv2.normalize(frame, frame, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        frame = cv2.applyColorMap(frame, cv2.COLORMAP_MAGMA)

                    if show:
                        txt = "[{:5}, {:4}, {:.6f}] ".format(c, pkt.getSequenceNum(), pkt.getTimestamp().total_seconds())
                        txt += "Exp: {:6.3f} ms, ".format(pkt.getExposureTime().total_seconds() * 1000)
                        txt += "ISO: {:4}, ".format(pkt.getSensitivity())
                        txt += "Lens pos: {:3}, ".format(pkt.getLensPosition())
                        txt += "Color temp: {} K".format(pkt.getColorTemperature())
                        if needs_newline:
                            print()
                            needs_newline = False
                        print(txt)

                    if capture:
                        capture_file_info = ('capture_' + c + '_' + cam_name[cam_socket_opts[cam_skt].name]
                             + '_' + str(width) + 'x' + str(height)
                             + '_' + capture_time
                             + '_exp_' + str(int(pkt.getExposureTime().total_seconds() * 1e6))
                             + '_iso_' + str(pkt.getSensitivity())
                             + '_lens_' + str(pkt.getLensPosition())
                             + '_' + str(pkt.getColorTemperature()) + 'K'
                             + '_' + str(pkt.getSequenceNum()))
                        capture_list.remove(c)
                        print()

                    if c.startswith('raw_') or c.startswith('tof_amplitude_'):
                        if capture:
                            filename = capture_file_info + '_10bit.bw'
                            print('Saving:', filename)
                            frame.tofile(filename)
                    else:
                        if capture and args.enable_raw:
                            payload = pkt.getData()
                            filename = capture_file_info + '_P420.yuv'
                            print('Saving:', filename)
                            payload.tofile(filename)

                    if capture and not c.startswith('raw_') and not c.startswith('tof_amplitude_'):
                        filename = capture_file_info + '.png'
                        print('Saving:', filename)
                        cv2.imwrite(filename, frame)

                    cv2.imshow(c, frame)
            except Exception as e:
                print(e)
                exit_cleanly(0, 0)

        imu_status = ""
        if args.enable_imu:
            acc_i = accel_freq.get_instant()
            gyr_i = gyro_freq.get_instant()
            acc_a = accel_freq.get_average()
            gyr_a = gyro_freq.get_average()

            acc_i_s = "{:6.2f}".format(acc_i) if acc_i is not None else "  N/A "
            gyr_i_s = "{:6.2f}".format(gyr_i) if gyr_i is not None else "  N/A "
            acc_a_s = "{:6.2f}".format(acc_a) if acc_a is not None else "  N/A "
            gyr_a_s = "{:6.2f}".format(gyr_a) if gyr_a is not None else "  N/A "

            imu_status = " | IMU acc[inst|avg]:{}|{}Hz gyro[inst|avg]:{}|{}Hz".format(
                acc_i_s, acc_a_s, gyr_i_s, gyr_a_s)

        print("\rFPS:", *["{:6.2f}|{:6.2f}".format(fps_host[c].get(), fps_capt[c].get()) for c in cam_list],
              imu_status, end=' ', flush=True)
        needs_newline = True

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('/'):
            show = not show
            print("" if show else "Printing camera settings: OFF")
        elif key == ord('c'):
            capture_list = streams.copy()
            capture_time = time.strftime('%Y%m%d_%H%M%S')
        elif key == ord('t'):
            print("Autofocus trigger (and disable continuous)")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            ctrl.setAutoFocusTrigger()
            controlQueueSend(ctrl)
        elif key == ord('f'):
            print("Autofocus enable, continuous")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
            controlQueueSend(ctrl)
        elif key == ord('e'):
            print("Autoexposure enable")
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            controlQueueSend(ctrl)
        elif key in [ord(','), ord('.')]:
            if key == ord(','):
                lensPos -= LENS_STEP
            if key == ord('.'):
                lensPos += LENS_STEP
            lensPos = clamp(lensPos, lensMin, lensMax)
            print("Setting manual focus, lens position: ", lensPos)
            ctrl = dai.CameraControl()
            ctrl.setManualFocusRaw(lensPos)
            controlQueueSend(ctrl)
        elif key in [ord('i'), ord('o'), ord('k'), ord('l')]:
            if key == ord('i'):
                expTime -= EXP_STEP
            if key == ord('o'):
                expTime += EXP_STEP
            if key == ord('k'):
                sensIso -= ISO_STEP
            if key == ord('l'):
                sensIso += ISO_STEP
            expTime = clamp(expTime, expMin, expMax)
            sensIso = clamp(sensIso, sensMin, sensMax)
            print("Setting manual exposure, time: ", expTime, "iso: ", sensIso)
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(expTime, sensIso)
            controlQueueSend(ctrl)
        elif key in [ord('g'), ord('h')]:
            if key == ord('g'):
                expTimeLimit -= 50
            else:
                expTimeLimit += 50
            print("Exposure time limit: ", expTimeLimit)
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureLimit(expTimeLimit)
            controlQueueSend(ctrl)
        elif key == ord('1'):
            awb_lock = not awb_lock
            print("Auto white balance lock:", awb_lock)
            ctrl = dai.CameraControl()
            ctrl.setAutoWhiteBalanceLock(awb_lock)
            controlQueueSend(ctrl)
        elif key == ord('2'):
            ae_lock = not ae_lock
            print("Auto exposure lock:", ae_lock)
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureLock(ae_lock)
            controlQueueSend(ctrl)
        elif key == ord('a'):
            dotIntensity = dotIntensity - DOT_STEP
            if dotIntensity < 0:
                dotIntensity = 0
            device.setIrLaserDotProjectorIntensity(dotIntensity)
            print('IR Dot intensity:', dotIntensity)
        elif key == ord('d'):
            dotIntensity = dotIntensity + DOT_STEP
            if dotIntensity > DOT_MAX:
                dotIntensity = DOT_MAX
            device.setIrLaserDotProjectorIntensity(dotIntensity)
            print('IR Dot intensity:', dotIntensity)
        elif key == ord('w'):
            floodIntensity = floodIntensity + FLOOD_STEP
            if floodIntensity > FLOOD_MAX:
                floodIntensity = FLOOD_MAX
            device.setIrFloodLightIntensity(floodIntensity)
            print('IR Flood intensity:', floodIntensity)
        elif key == ord('s'):
            floodIntensity = floodIntensity - FLOOD_STEP
            if floodIntensity < 0:
                floodIntensity = 0
            device.setIrFloodLightIntensity(floodIntensity)
            print('IR Flood intensity:', floodIntensity)
        elif key >= 0 and chr(key) in "34567890[]p\\;'*":
            if key == ord('3'):
                control = 'awb_mode'
            elif key == ord('4'):
                control = 'ae_comp'
            elif key == ord('5'):
                control = 'anti_banding_mode'
            elif key == ord('6'):
                control = 'effect_mode'
            elif key == ord('7'):
                control = 'brightness'
            elif key == ord('8'):
                control = 'contrast'
            elif key == ord('9'):
                control = 'saturation'
            elif key == ord('0'):
                control = 'sharpness'
            elif key == ord('\\'):
                control = 'scene_mode'
            elif key == ord(';'):
                control = 'control_mode'
            elif key == ord("'"):
                control = 'capture_intent'
            elif key == ord('['):
                control = 'luma_denoise'
            elif key == ord(']'):
                control = 'chroma_denoise'
            elif key == ord('p'):
                control = 'tof_amplitude_min'
            elif key == ord('*'):
                control = 'wb_manual'
            print("Selected control:", control)
        elif key in [ord('-'), ord('_'), ord('+'), ord('=')]:
            change = 0
            if key in [ord('-'), ord('_')]:
                change = -1
            if key in [ord('+'), ord('=')]:
                change = 1
            ctrl = dai.CameraControl()
            if control == 'none':
                print("Please select a control first using keys 3..9 0 [ ] backslash ; quote")
            elif control == 'ae_comp':
                ae_comp = clamp(ae_comp + change, -9, 9)
                print("Auto exposure compensation:", ae_comp)
                ctrl.setAutoExposureCompensation(ae_comp)
            elif control == 'anti_banding_mode':
                abm = anti_banding_mode.step(change)
                print("Anti-banding mode:", abm)
                ctrl.setAntiBandingMode(abm)
            elif control == 'awb_mode':
                awb = awb_mode.step(change)
                print("Auto white balance mode:", awb)
                ctrl.setAutoWhiteBalanceMode(awb)
            elif control == 'effect_mode':
                eff = effect_mode.step(change)
                print("Effect mode:", eff)
                ctrl.setEffectMode(eff)
            elif control == 'scene_mode':
                sc = scene_mode.step(change)
                print("Scene mode:", sc)
                ctrl.setSceneMode(sc)
            elif control == 'control_mode':
                cm = control_mode.step(change)
                print("Control mode:", cm)
                ctrl.setControlMode(cm)
            elif control == 'capture_intent':
                ci = capture_intent.step(change)
                print("Capture intent:", ci)
                ctrl.setCaptureIntent(ci)
            elif control == 'brightness':
                brightness = clamp(brightness + change, -10, 10)
                print("Brightness:", brightness)
                ctrl.setBrightness(brightness)
            elif control == 'contrast':
                contrast = clamp(contrast + change, -10, 10)
                print("Contrast:", contrast)
                ctrl.setContrast(contrast)
            elif control == 'saturation':
                saturation = clamp(saturation + change, -10, 10)
                print("Saturation:", saturation)
                ctrl.setSaturation(saturation)
            elif control == 'sharpness':
                sharpness = clamp(sharpness + change, 0, 4)
                print("Sharpness:", sharpness)
                ctrl.setSharpness(sharpness)
            elif control == 'luma_denoise':
                luma_denoise = clamp(luma_denoise + change, 0, 4)
                print("Luma denoise:", luma_denoise)
                ctrl.setLumaDenoise(luma_denoise)
            elif control == 'chroma_denoise':
                chroma_denoise = clamp(chroma_denoise + change, 0, 4)
                print("Chroma denoise:", chroma_denoise)
                ctrl.setChromaDenoise(chroma_denoise)
            elif control == 'wb_manual':
                wb_manual = wb_manual + change * 100
                print("White balance:", wb_manual)
                ctrl.setManualWhiteBalance(wb_manual)

            controlQueueSend(ctrl)

    print()