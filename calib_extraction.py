#this script takes care of "reshaping" the data of the OAKDPRO calibration (stored in oak_calib.json) 
#into a format that is compatible with the openvins framework.

#in particular: 

# IMU noise densities and random walks are made scalar (from x y z noise densities to a single value
# by taking the maximum of the three values. This is done because openvins expects a single value for each of these parameters.

# this script takes care of "reshaping" the data of the OAKDPRO calibration
# (stored in oak_calib.json) into a format that is compatible with OpenVINS.
#
# In particular:
# - IMU noise densities and random walks are reduced from per-axis values
#   (x, y, z) to a single scalar by taking the maximum of the three values.
# - This is a conservative choice, since OpenVINS expects a single value for
#   each of these parameters.

import json

with open("oak_calib.json", "r") as f:
    data = json.load(f)

noise   = data["imuCalibrationParams"]["noise"]
acc_cal = data["imuCalibrationParams"]["accelerometer"]
gyr_cal = data["imuCalibrationParams"]["gyroscope"]

def max_from_axes(sensor_block, field):
    x = sensor_block["x"][field]
    y = sensor_block["y"][field]
    z = sensor_block["z"][field]
    return max(x, y, z)

def mat3_from_3x4(block):
    return [row[:3] for row in block]

def vec3_from_3x4_lastcol(block):
    return [row[3] for row in block]

def print_yaml_matrix(name, M, indent=2):
    sp = " " * indent
    print(f"{sp}{name}:")
    for row in M:
        print(f"{sp}  - [{row[0]}, {row[1]}, {row[2]}]")

def print_yaml_vector_comment(name, v, indent=2):
    sp = " " * indent
    print(f"{sp}# {name}: [{v[0]}, {v[1]}, {v[2]}]")

acc_noise_density = max_from_axes(noise["accelerometer"], "noiseDensity")
acc_random_walk   = max_from_axes(noise["accelerometer"], "randomWalk")
gyr_noise_density = max_from_axes(noise["gyroscope"], "noiseDensity")
gyr_random_walk   = max_from_axes(noise["gyroscope"], "randomWalk")

Ta = mat3_from_3x4(acc_cal)
Tw = mat3_from_3x4(gyr_cal)
gyro_bias_like = vec3_from_3x4_lastcol(gyr_cal)
acc_bias_like = vec3_from_3x4_lastcol(acc_cal)

R_IMUtoGYRO = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]

R_IMUtoACC = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]

Tg = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
]

print("imu0:")
print("  model: \"kalibr\"")
print("  update_rate: 200")
print("  time_offset: 0.0")

print(f"  accelerometer_noise_density: {acc_noise_density}")
print(f"  accelerometer_random_walk: {acc_random_walk}")
print(f"  gyroscope_noise_density: {gyr_noise_density}")
print(f"  gyroscope_random_walk: {gyr_random_walk}")

print_yaml_matrix("Tw", Tw)
print_yaml_matrix("R_IMUtoGYRO", R_IMUtoGYRO)
print_yaml_matrix("Ta", Ta)
print_yaml_matrix("R_IMUtoACC", R_IMUtoACC)
print_yaml_matrix("Tg", Tg)

print_yaml_vector_comment("luxonis_accel_offset_3x1", acc_bias_like)
print_yaml_vector_comment("luxonis_gyro_offset_3x1", gyro_bias_like)