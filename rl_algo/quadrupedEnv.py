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
    # TODO: Modify MAX_RATE
    ABDUCTION_JOINT_MAX_RATE = 5
    INNER_JOINT_MAX_RATE = 10
    OUTER_JOINT_MAX_RATE = 10
    QUATERNION_BOUND = 1.0

    def __init__(self):
        super(QuadrupEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        
        self.TRADEOFF_PARAM = 0.5
        
        # Action Space Params
        action_shape_3 = (3,)
        action_low_3 = np.array([-1.0,-1.0,-1.0])
        action_high_3 = np.array([1.0,1.0,1.0])
        
        action_shape_12 = (3,4)
        action_low_12 = np.array([[-1.0,-1.0,-1.0,-1.0],
                                  [-1.0,-1.0,-1.0,-1.0],
                                  [-1.0,-1.0,-1.0,-1.0]])
        
        action_high_12 = np.array([[1.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0, 1.0]])
        
        self.action_space = spaces.Box(
            low=action_low_12, high=action_high_12, 
            shape=action_shape_12, dtype=np.float32
            )
        
        # The observation will be the coordisabilitynate of the agent
        # this can be described both by Discrete and Box space
        
        # Observation Space Param -- data from IMU and sensors(optional)
        obs_shape = (4,)    # quaternions
        obs_low_3 = np.array([-1.0,-1.0,-1.0,-1.0])
        obs_high_3 = np.array([1.0, 1.0, 1.0, 1.0])
        
        obs_low_12 = np.array([[-1.0,-1.0,-1.0,-1.0],
                               [-1.0,-1.0,-1.0,-1.0],
                               [-1.0,-1.0,-1.0,-1.0]])
        
        obs_high_12 = np.array([[1.0, 1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0, 1.0]])
        
        self.observation_space = spaces.Box(
            low=obs_low_12, high=obs_high_12, shape=obs_shape, dtype=np.float32
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