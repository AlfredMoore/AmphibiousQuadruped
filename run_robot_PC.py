from rl_algo.quadrupedEnv import QuadruppedEnv
from communication.SocketInterface_PC_RL import SocketInterface_PC

from config import Configuration
import time
import numpy as np

def main():
    
    # config
    PPO_config = Configuration.PPO_config()
    Env_config = Configuration.Env_config()
    IMU_config = Configuration.IMU_config()
    Socket_config = Configuration.Socket_config()
    
    # Socket communication
    socket_PC = SocketInterface_PC( Socket_config )
    time.sleep(1)
    
    # PPO init
    Env = QuadruppedEnv( Env_config=Env_config, IMU_config=IMU_config )
    
    start_time = time.time()
    # Before traning, Warm up!
    expert_serial = None
    
    action = 1.0 * Env.init_servo_degree
    
    for _ in range(PPO_config.batch_size):
        # 1.receive state and store
        state = socket_PC.subscriber_env()
        imu_data = state["imu_data"]
        # TODO: 
        # thrust = empirical_model_function( action )
        # obs = np.concatenate((imu_data, thrust))
        # store obs in the rl_algo.replaybuff
        
        # 2.publish action and store
        t = time.time() - start_time
        # TODO: action = expert_serial_function(t)
        action = None
        
        cmd = {"action": action, "info": None}
        socket_PC.publisher_cmd( command=cmd )
        
    
    while True:
        # 1.receive state
        state = socket_PC.subscriber_env()
        # TODO: 
        # thrust = empirical_model_function( action )
        # obs = np.concatenate((state, thrust))
        # run PPO algorithm
         
        # 2.publish action
        # TODO: action = PPO(obs)
        cmd = {"action": action, "info": None}
        
    
if __name__ == "__main__":
    main()