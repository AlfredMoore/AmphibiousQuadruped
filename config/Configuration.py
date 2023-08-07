import numpy as np
from config.ServoCalibration import MICROS_PER_RAD, NEUTRAL_ANGLE_DEGREES
from enum import Enum

class PPO_config:
    def __init__(self) -> None:
        self.max_train_steps = int(3e6)                # int, default=int(3e6), help=" Maximum number of training steps")
        self.evaluate_freq = float(5e3)                   # float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
        self.save_freq = int(20)                       # int, default=20, help="Save frequency")
        self.policy_dist = "Gaussian"                     # str, default="Gaussian            help="Beta or Gaussian")
        self.batch_size = int(2048)                      # int, default=2048, help="Batch size")
        self.mini_batch_size = int(64)                # int, default=64, help="Minibatch size")
        self.hidden_width = int(64)                   # int, default=64, help="The number of neurons in hidden layers of the neural network")
        self.lr_a = float(3e-4)                           # float, default=3e-4, help="Learning rate of actor")
        self.lr_c = float(3e-4)                           # float, default=3e-4, help="Learning rate of critic")
        self.gamma = float(0.99)                          # float, default=0.99, help="Discount factor")
        self.lamda = float(0.95)                          # float, default=0.95, help="GAE parameter")
        self.epsilon = float(0.2)                        # float, default=0.2, help="PPO clip parameter")
        self.K_epochs = int(10)                       # int, default=10, help="PPO parameter")
        self.use_adv_norm = True                   # bool, default=True, help="Trick 1:advantage normalization")
        self.use_state_norm = True                 # bool, default=True, help="Trick 2:state normalization")
        self.use_reward_norm = True                # bool, default=False, help="Trick 3:reward normalization")
        self.use_reward_scaling = True             # bool, default=True, help="Trick 4:reward scaling")
        self.entropy_coef = float(0.01)                   # float, default=0.01, help="Trick 5: policy entropy")
        self.use_lr_decay = True                   # bool, default=True, help="Trick 6:learning rate Decay")
        self.use_grad_clip = True                  # bool, default=True, help="Trick 7: Gradient clip")
        self.use_orthogonal_init = True            # bool, default=True, help="Trick 8: orthogonal initialization")
        self.set_adam_eps = True                   # float, default=True, help="Trick 9: set Adam epsilon=1e-5")
        self.use_tanh = int(1)                       # float, default=True, help="Trick 10: tanh activation function")


class Env_config:
    def __init__(self) -> None:
        self.quadruped_mode=True
        self.tradeoff_param=0.5
        self.empirical_model=False

        

class IMU_config:
    def __init__(self) -> None:
        self.addr=0x50
        self.i2cbus=0
        self.GRAVITY = 9.8
        self.buffer = {"acceleration":0x34, "angular velocity":0x37, 
                       "RollPitchYaw":0x3d, "Quaterions":0x51}
        self.data_range = {"acceleration":16 * self.GRAVITY, "angular velocity":2000, 
                           "RollPitchYaw":180, "Quaterions":1}
        # data structure from IMU
        # 0x34 - 0x36 : xyz_accelerations
        # 0x37 - 0x39 : xyz_angular_velocity
        # 0x3d - 0x3f : Roll_Pitch_Yaw_xyz
        # 0x51 - 0x54 : Quaterions Q0-Q3


class Socket_config:
    def __init__(self) -> None:
        
        self.HOST_IP = "192.168.31.52"
        self.HOST_PORT = 50000
        self.message_rate = 50

        


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