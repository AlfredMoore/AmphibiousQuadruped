#coding:utf-8
#import necessary package
import socket
import time
import sys
import json
import numpy as np
from typing import List, Tuple, Dict
from src.Command import Command
from src.State import State, BehaviorState
from src.Utilities import deadband, clipped_first_order_filter
from pupper.Config import Configuration

 
class SocketInterface_Pi:
    def __init__(self, HOST_IP="192.168.31.52", HOST_PORT=50000) -> None:
        
        # parameters
        self.HOST_IP = HOST_IP
        self.HOST_PORT = HOST_PORT
        # # IP address of Rasberry Pi
        # HOST_IP = "192.168.31.87" 
        # HOST_PORT = 8888

        self.config = Configuration()
        self.previous_gait_toggle = 0
        self.previous_state = BehaviorState.REST
        self.previous_hop_toggle = 0
        self.previous_activate_toggle = 0

        self.message_rate = 50


        # socket connection
        print("Starting socket: TCP...")
        #1.create socket object:socket=socket.socket(family,type)
        self.socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        print("TCP server listen @ %s:%d!" %(self.HOST_IP, self.HOST_PORT) )
        self.host_addr = (self.HOST_IP, self.HOST_PORT)
        #2.bind socket to addr:socket.bind(address)
        self.socket_tcp.bind(self.host_addr)
        #3.listen connection request:socket.listen(backlog)
        self.socket_tcp.listen(1)
        #4.waite for client:connection,address=socket.accept()
        self.socket_con, (client_ip, client_port) = self.socket_tcp.accept()
        print("Connection accepted from %s." %client_ip)
        msg = "Welcome to Pupper Rpi TCP server!"
        # msg=msg.encode(encoding='utf_8', errors='strict')
        msg = json.dumps(msg)
        self.socket_con.send(bytes(msg.encode('utf-8')))


    def subscriber_cmd(self):
        while True:
            # print("Receiving package...")
            data = self.socket_con.recv(512)
            # print("Receiving package...")

            if len(data)>0:
                msg_recv = json.loads(data)
                return msg_recv


    def publisher_env(self, msg_state: Dict):
        try:
            env_pub = json.dumps(msg_state)
            self.socket_con.send(bytes(env_pub.encode('utf-8')))

        except Exception:
            self.socket_con.close()
            sys.exit(1)

    
    def get_command(self, state, msg, do_print=False):

        # msg = 
        command = Command()
        
        ####### Handle discrete commands ########
        # Check if requesting a state transition to trotting, or from trotting to resting
        gait_toggle = msg["R1"]
        command.trot_event = (gait_toggle == 1 and self.previous_gait_toggle == 0)

        # Check if requesting a state transition to hopping, from trotting or resting
        hop_toggle = msg["x"]
        command.hop_event = (hop_toggle == 1 and self.previous_hop_toggle == 0)
        
        activate_toggle = msg["L1"]
        command.activate_event = (activate_toggle == 1 and self.previous_activate_toggle == 0)
        print("activate_event:",command.activate_event)
        # Update previous values for toggles and state
        self.previous_gait_toggle = gait_toggle
        self.previous_hop_toggle = hop_toggle
        self.previous_activate_toggle = activate_toggle

        ####### Handle continuous commands ########
        x_vel = msg["ly"] * self.config.max_x_velocity
        y_vel = msg["lx"] * -self.config.max_y_velocity
        command.horizontal_velocity = np.array([x_vel, y_vel])
        command.yaw_rate = msg["rx"] * -self.config.max_yaw_rate

        message_rate = msg["message_rate"]
        message_dt = 1.0 / message_rate

        pitch = msg["ry"] * self.config.max_pitch
        deadbanded_pitch = deadband(
            pitch, self.config.pitch_deadband
        )
        pitch_rate = clipped_first_order_filter(
            state.pitch,
            deadbanded_pitch,
            self.config.max_pitch_rate,
            self.config.pitch_time_constant,
        )
        command.pitch = state.pitch + message_dt * pitch_rate

        height_movement = msg["dpady"]
        command.height = state.height - message_dt * self.config.z_speed * height_movement

        roll_movement = - msg["dpadx"]
        command.roll = state.roll + message_dt * self.config.roll_speed * roll_movement

        return command


    def __del__(self):
        self.socket_con.close()
        self.socket_tcp.close()
        print("Terminate Pi Socket Connection!")


if __name__ == "__main__":
    # connection test
    # Establish connection
    socket_Pi = SocketInterface_Pi()
    
    # PC -> Pi
    cmd_recv = socket_Pi.subscriber_cmd()
    print("Pi receive:",cmd_recv)

    # Pi -> PC
    state = {"a":1, "b":2}
    print("Pi send:", state)
    socket_Pi.publisher_env(state)

    