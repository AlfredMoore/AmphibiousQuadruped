import numpy as np
import time

from hardware.imu import ImuInterface_I2C
from hardware.HardwareInterface import HardwareInterface
from hardware.Config import ServoParams, PWMParams

from rl_algo.quadrupedEnv import QuadruppedEnv

from config import Configuration

from communication.SocketInterface_Pi_RL import SocketInterface_Pi


def main():
    """Main program
    """

    # config
    PPO_config = Configuration.PPO_config()
    Env_config = Configuration.Env_config()
    IMU_config = Configuration.IMU_config()
    Socket_config = Configuration.Socket_config()
    
    # IMU
    IMU_buffer = IMU_config.buffer
    IMU_data_range = IMU_config.data_range
    imu_interface = ImuInterface_I2C( IMU_config=IMU_config )
    
    # Servo controller
    hardware_interface = HardwareInterface( ServoParams, PWMParams )
    
    # Socket communication
    socket_Pi = SocketInterface_Pi( Socket_config )
    time.sleep(1)

    # PPO init (PC)
    # Env = QuadruppedEnv( Env_config=Env_config, IMU_config=IMU_config )
    
    start_time = time.time()
    COMM_freq = 100
    

    while True:
        # 1. receive a cmd that indicates what's next
        cmd = socket_Pi.subscriber_cmd()
        if cmd["message"] is not None:
            print(cmd["message"])
            
        # 2. receive action or send state
        # action
        if cmd["next"]==Socket_config.next_action:
            cmd = socket_Pi.subscriber_cmd()
            angle_degree = cmd["action"]
            hardware_interface.set_actuator_postions_radians(angle_degree)
        
        # state
        if cmd["next"]==Socket_config.next_state:
            imu_data = imu_interface.read_all()        # read 9-axis IMU
            msg = {"imu_data": imu_data, "info": None}
            socket_Pi.publisher_env( msg_state=msg )
        
        # time.sleep(1/COMM_freq)
        # data structure from IMU
        # 0x34 - 0x36 : xyz_accelerations
        # 0x37 - 0x39 : xyz_angular_velocity
        # 0x3d - 0x3f : Roll_Pitch_Yaw_xyz
        # 0x51 - 0x54 : Quaterions Q0-Q3


if __name__ == "__main__":
    main()
