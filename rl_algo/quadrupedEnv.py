import numpy as np
import gymnasium as gym
import time
from gymnasium import spaces


def stability_evaluation(imu_data):
    # TODO: read article and calculate sability
    
    stability_reward = None
    return stability_reward


def speed_evaluation(imu_data):
    # TODO: calculate speed
    
    speed_reward = None
    return speed_reward


class QuadruppedEnv(gym.Env):
    """
    Customize Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """

    # # Because of google colab, we cannot implement the GUI ('human' render mode)
    # metadata = {"render_modes": ["console"]}

    # Define constants for clearer code
    
    def __init__(self,  Env_config, IMU_config, Socket_config, socket_handler ):
        """
        quadruped_mode(bool): Control of single leg or the quadruped_mode
        tradeoff_param(Dict): {"speed": a, "stability": b, "thrust": c}
        empirical_model(bool): use empirical model of the thrust from joint angle
        """
        
        super(QuadruppedEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        self.quadruped_mode = Env_config.quadruped_mode
        self.empirical_model = Env_config.empirical_model
        self.TRADEOFF_PARAM = Env_config.tradeoff_param
        self.max_steps = Env_config.max_steps
        
        self.IMU_data_range = IMU_config.data_range
        
        self.socket_handler = socket_handler
        self.Socket_config = Socket_config
        
        self.step_counter = 0
        
        # Action Space Params
        if self.quadruped_mode:
            # quadruped_mode mode
            action_shape = (3,4)
            action_shape = action_shape[0] * action_shape[1]
            # quadruped_mode joint angle
            action_low = -1.0 * np.ones(action_shape)  
            action_high = np.ones(action_shape)

        
        else:
            # single leg mode
            action_shape = (3,)
            action_low = -1.0 * np.ones(action_shape)
            action_high = np.ones(action_shape)
        
        
        self.action_space = spaces.Box(
            low=action_low, high=action_high, 
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
        
        if self.empirical_model:
            if self.quadruped_mode:
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
        action = 1.0 * self.init_servo_degree
        
        # action to init
        # 1. send a cmd that indicates what's next
        cmd = {"next":self.Socket_config.next_action, "action": None, "info": None, "message": "Quadrupped Env init"}
        self.socket_PC.publisher_cmd( command=cmd )
        
        # 2. send action or receive state
        cmd = {"next": None, "action": action, "info": None, "message": None}
        self.socket_PC.publisher_cmd( command=cmd )
        


    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        return: obs: (imu, thrust), info: dict
        """
        super().reset(seed=seed, options=options)

        # TODO: send msg that set the servo to [0, -45, 45]
        # initialize the env and the policy
        
        self.step_counter = 0
        # initialize servo degree
        action = 1.0 * self.init_servo_degree
        
        # init socket connection and init servo
        # action to reset
        # 1. send a cmd that indicates what's next
        cmd = {"next":self.Socket_config.next_action, "action": None, "info": None, "message": "Quadrupped Env reset"}
        self.socket_PC.publisher_cmd( command=cmd )
        
        # 2. send action or receive state
        cmd = {"next": None, "action": action, "info": None, "message": None}
        self.socket_PC.publisher_cmd( command=cmd )
        
        
        # state after reset
        # 1. send a cmd that indicates what's next
        cmd = {"next":self.Socket_config.next_state, "action": None, "info": None, "message": None}
        self.socket_PC.publisher_cmd( command=cmd )
        
        # 2. send action or receive state
        msg = self.socket_PC.subscriber_env()
        imu_data = msg["imu_data"]
        
        # TODO: build empirical model
        # thrust = empirical_model_function( action )
        thrust = np.zeros((1,))
        obs = np.concatenate((imu_data, thrust))

        return obs.astype(np.float32), {}  # obs: (imu, thrust), info: dict


    def step(self, action: np.array):
        """
        ### Return: obs, reward, done, truncated, info
        """
        self.step_counter += 1
        # action to step
        # 1. send a cmd that indicates what's next
        cmd = {"next":self.Socket_config.next_action, "action": None, "info": None, "message": "Quadrupped Env reset"}
        self.socket_PC.publisher_cmd( command=cmd )
        
        # 2. send action or receive state
        cmd = {"next": None, "action": action, "info": None, "message": None}
        self.socket_PC.publisher_cmd( command=cmd )

        # state
        # 1. send a cmd that indicates what's next
        cmd = {"next":self.Socket_config.next_state, "action": None, "info": None, "message": None}
        self.socket_PC.publisher_cmd( command=cmd )
        
        # 2. send action or receive state
        msg = self.socket_PC.subscriber_env()
        imu_data = msg["imu_data"]

        # TODO: build empirical model
        # thrust = empirical_model_function( action )
        thrust = np.zeros((1,))
        obs = np.concatenate((imu_data, thrust))
        
        # TODO: build reward model
        # reward = trade-off between speed, stability and thrust
        reward = None
        
        # TODO: what is the cornor case of terminated,
        # too large IMU difference, drift from the supposed line?
        done = False
        
        # reach max steps
        if self.step_counter >= self.max_steps:
            truncated = True
        else:
            truncated = False
        
        
        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return obs, reward, done, truncated, info
        

    def render(self):
        print()

    def close(self):
        pass