#!/usr/bin/env python3
import depthai as dai


def timeDeltaToMilliS(delta) -> float:
    return delta.total_seconds() * 1000.0


# Create pipeline
with dai.Pipeline() as pipeline:
    imu = pipeline.create(dai.node.IMU)

    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_UNCALIBRATED, 480)
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_UNCALIBRATED, 400)

    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    imuQueue = imu.out.createOutputQueue(maxSize=50, blocking=False)

    pipeline.start()

    prev_accel_ts = None
    prev_gyro_ts = None

    while pipeline.isRunning():
        try:
            imuData = imuQueue.get()
        except KeyboardInterrupt:
            break

        assert isinstance(imuData, dai.IMUData)
        imuPackets = imuData.packets

        for imuPacket in imuPackets:
            acceleroValues = imuPacket.acceleroMeter
            gyroValues = imuPacket.gyroscope

            acceleroTs = acceleroValues.getTimestamp()
            gyroTs = gyroValues.getTimestamp()

            imuF = "{:.06f}"
            tsF = "{:.03f}"

            # Frequenza accelerometro
            accel_freq_str = "N/A"
            if prev_accel_ts is not None:
                dt_acc_ms = timeDeltaToMilliS(acceleroTs - prev_accel_ts)
                if dt_acc_ms > 0:
                    accel_freq = 1000.0 / dt_acc_ms
                    accel_freq_str = f"{accel_freq:.2f} Hz"

            # Frequenza giroscopio
            gyro_freq_str = "N/A"
            if prev_gyro_ts is not None:
                dt_gyro_ms = timeDeltaToMilliS(gyroTs - prev_gyro_ts)
                if dt_gyro_ms > 0:
                    gyro_freq = 1000.0 / dt_gyro_ms
                    gyreo_freq_str = f"{gyro_freq:.2f} Hz"

            prev_accel_ts = acceleroTs
            prev_gyro_ts = gyroTs

            # print(f"Accelerometer timestamp: {acceleroTs}")
            # print(f"Latency [ms]: {dai.Clock.now() - acceleroValues.getTimestamp()}")
            print(f"Accelerometer [m/s^2]: x: {imuF.format(acceleroValues.x)} y: {imuF.format(acceleroValues.y)} z: {imuF.format(acceleroValues.z)}")
            # print(f"Accelerometer freq [Hz]: {accel_freq_str}")

            # print(f"Gyroscope timestamp: {gyroTs}")
            # print(f"Gyroscope [rad/s]: x: {imuF.format(gyroValues.x)} y: {imuF.format(gyroValues.y)} z: {imuF.format(gyroValues.z)}")
            # print(f"Gyroscope freq [Hz]: {gyro_freq_str}")
            print()