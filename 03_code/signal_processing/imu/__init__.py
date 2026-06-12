"""
IMU signal-processing subpackage -- Spinal Movement Risk Monitor.

Turns raw accelerometer/gyroscope CSV from the four spine-mounted IMUs
(Pelvis, L3, T12, T4) into trunk-orientation angles for the risk pipeline:

    * convert   -- raw count -> physical units, gyro-bias calibration, validation
    * madgwick  -- Madgwick AHRS sensor fusion to quaternions / Euler angles,
                   plus the inter-segment angle differences (e.g. Pelvis-L3)
                   used as the lumbar flexion proxy

LDLJ (log-dimensionless jerk) smoothness lives alongside these as a
movement-quality metric derived from the fused orientation stream.
"""
from signal_processing.imu.convert import RawConverter, GyroBiasCalibrator, ConversionValidator
from signal_processing.imu.madgwick import MadgwickAHRS, fuse_imu_dataframe, fuse_four_imu_dataframe
