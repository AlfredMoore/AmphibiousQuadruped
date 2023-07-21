import numpy as np


class Command:
    """Stores movement command
    """

    def __init__(self):
        self.horizontal_velocity = np.array([0, 0])
        self.yaw_rate = 0.0
        self.height = -0.16
        self.pitch = 0.0
        self.roll = 0.0
        self.activation = 0
        
        self.hop_event = False
        self.trot_event = False
        self.activate_event = False

    def __str__(self) -> str:
        dict = {
            "horizontal_velocity": self.horizontal_velocity,
            "yaw_rate": self.yaw_rate,
            "height": self.height,
            "pitch": self.pitch,
            "roll": self.roll,
            "activation": self.activation,
            "hop_event": self.hop_event,
            "trot_event": self.trot_event,
            "activate_event": self.activate_event
        }
        return dict