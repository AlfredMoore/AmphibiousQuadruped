#coding:utf-8
#import necessary package
import socket
import time
import sys
import json
import numpy as np
from typing import List, Tuple, Dict

from src.Utilities import deadband, clipped_first_order_filter
# from pupper.Config import Configuration

 
class SocketInterface_Pi:
    def __init__(self, HOST_IP="192.168.31.52", HOST_PORT=50000) -> None:
        
        # parameters
        self.HOST_IP = HOST_IP
        self.HOST_PORT = HOST_PORT
        # # IP address of Rasberry Pi
        # HOST_IP = "192.168.31.87" 
        # HOST_PORT = 8888

        # self.config = Configuration()

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

    
    def get_command(self, msg, do_print=False):
        """
        msg: {"servo_degree": np.array, "info": Dict}
        return (servo_degree, info) 
        """
        
        # command = {"servo_degree": np.array, "info": Dict}
        
        set_points = msg["servo_degree"]
        info = msg["info"]
        return set_points, info


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

    