import numpy as np
from config.ServoCalibration import MICROS_PER_RAD, NEUTRAL_ANGLE_DEGREES
from enum import Enum

# TODO: put these somewhere else
class PWMParams:
    def __init__(self):
        self.pins = np.array([[2, 14, 18, 23], [3, 15, 27, 24], [4, 17, 22, 25]])
        self.range = 4000
        self.freq = 250


class ServoParams:
    def __init__(self):
        self.neutral_position_pwm = 1500  # Middle position
        self.micros_per_rad = MICROS_PER_RAD  # Must be calibrated

        # The neutral angle of the joint relative to the modeled zero-angle in degrees, for each joint
        self.neutral_angle_degrees = NEUTRAL_ANGLE_DEGREES
        
        # modified multipliers
        self.servo_multipliers = np.array(
            [[-1, -1, -1, -1], [1, -1, 1, -1], [-1, 1, -1, 1]]
        )

    @property
    def neutral_angles(self):
        return self.neutral_angle_degrees * np.pi / 180.0  # Convert to radians
