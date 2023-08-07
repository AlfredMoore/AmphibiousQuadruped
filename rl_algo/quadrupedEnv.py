import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.env_checker import check_env

from testComms.SocketInterface_Pi_RL import SocketInterface_PC



def stability_evaluation(quaternion):
    # TODO: read article and calculate sability
    
    stability_reward = None
    return stability_reward


def speed_evaluation(quaternion):
    # TODO: calculate speed
    
    speed_reward = None
    return speed_reward


class QuadrupEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """

    # # Because of google colab, we cannot implement the GUI ('human' render mode)
    # metadata = {"render_modes": ["console"]}

    # Define constants for clearer code
    
    GRAVITY = 9.8
    IMU_data_range = {"acceleration":16 * GRAVITY, "angular velocity":2000, 
                        "RollPitchYaw":180, "Quaterions":1}

    def __init__(self, quadruped_mode=True, tradeoff_param=0.5, empirical_model=False ):
        """
        quadruped_mode(bool): Control of single leg or the quadruped_mode
        tradeoff_param(float): param * speed + (1-param) * stability
        empirical_model(bool): use empirical model of the thrust from joint angle
        """
        
        super(QuadrupEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        
        self.TRADEOFF_PARAM = tradeoff_param
        
        # Action Space Params
        if quadruped_mode:
            # quadruped_mode mode
            action_shape = (3,4)
            action_shape_flatten = action_shape[0] * action_shape[1]
            # quadruped_mode joint angle
            action_low = -1.0 * np.ones(action_shape)
            action_low_flatten = action_low.flatten
            
            action_high = np.ones(action_shape)
            action_high_flatten = action_high.flatten
        
        else:
            # single leg mode
            action_shape = (3,)
            action_low_flatten = -1.0 * np.ones(action_shape)
            action_high_flatten = np.ones(action_shape)
        
        
        
        self.action_space = spaces.Box(
            low=action_low_flatten, high=action_high_flatten, 
            shape=action_shape, dtype=np.float32
            )
        
        
        # Observation Space Param -- data from IMU, other sensors and empirical_model
        IMU_shape = (13,)     # 3+3+3+4=13
        
        acceleration_range = np.array([self.IMU_data_range["acceleration"]]*3)
        angular_velocity_range = np.array([self.IMU_data_range["angular velocity"]]*3)
        RPY_range = np.array([self.IMU_data_range["RollPitchYaw"]]*3)
        Quaterions_range = np.array([self.IMU_data_range["Quaterions"]]*4)
        
        IMU_range = np.concatenate((acceleration_range, angular_velocity_range, RPY_range, Quaterions_range), axis=0)
        
        # data structure from IMU
        # 0x34 - 0x36 : xyz_accelerations
        # 0x37 - 0x39 : xyz_angular_velocity
        # 0x3d - 0x3f : Roll_Pitch_Yaw_xyz
        # 0x51 - 0x54 : Quaterions Q0-Q3
        
        if empirical_model:
            if quadruped_mode:
                empirical_thrust_shape = (4,)
                empirical_thrust_range = np.array([10]*4)
            else:
                empirical_thrust_shape = (1,)
                empirical_thrust_range = np.array([10])
                
            obs_range = np.concatenate((IMU_range, empirical_thrust_range), axis=0)
                
        else:
            empirical_thrust_shape = (0,)
            obs_range = IMU_range
                
        obs_shape = (IMU_shape[0] + empirical_thrust_shape[0], )
        
        self.observation_space = spaces.Box(
            low=obs_range * 1.0, high=obs_range * -1.0, shape=obs_shape, dtype=np.float32
        )
        
        # initialize servo degree
        self.init_servo_degree = np.array([[  0,  0,  0,  0],
                                           [ 45, 45, 45, 45],
                                           [-45,-45,-45,-45]])
        
        self.servo_degree = 1.0 * self.init_servo_degree
        
        # init socket connection
        
        # init servo 
        


    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)

        # TODO: send msg that set the servo to [0, -45, 45]
        # initialize the env and the policy
        
        
        # initialize servo degree
        self.servo_degree = 1.0 * self.init_servo_degree

        
        # return (obs, info)
        return np.array([0,0,0,0]).astype(np.float32), {}  # quaternion, empty info dict


    def step(self, action: np.array):
        """
        action: difference of joint servo degree
        [[x, x, x, x],[x, x, x, x],[x, x, x, x]]
        """

        # TODO: convert the action into servo degree and send msg to the RPi
        # action shape should be (3,4) or (3,)
        servo_degree_max_rate = np.array([self.ABDUCTION_JOINT_MAX_RATE, 
                                          self.INNER_JOINT_MAX_RATE,
                                          self.OUTER_JOINT_MAX_RATE])
        
        if action.shape == (3,):
            servo_degree_diff = servo_degree_max_rate * action
            self.servo_degree += servo_degree_diff
            
        elif action.shape == (3,4):
            servo_degree_diff = np.tile( servo_degree_max_rate.reshape((3,1)), (1,4) )
            self.servo_degree += servo_degree_diff
        



        # TODO: what is the cornor case of terminated,
        # too large IMU difference, drift from the supposed line?
        terminated = False
        truncated = False  # we do not limit the number of steps here

        # TODO: reward = trade-off between speed and stablity
        # 1. Receive socket msg
        # 2. Calculate reward function
        speed_reward = speed_evaluation()
        stability_reward = stability_evaluation()
        reward = self.TRADEOFF_PARAM * speed_reward + (1-self.TRADEOFF_PARAM) * stability_reward

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            np.array([self.agent_pos]).astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        print()

    def close(self):
        pass