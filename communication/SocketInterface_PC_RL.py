import socket
import time
import sys
import json
import pickle
from typing import List, Tuple, Dict


class SocketInterface_PC:
    """
    subscriber_env()
    publisher_cmd(command:Dict)
    """

    def __init__(self, Socket_config) -> None:
        self.SERVER_IP = Socket_config.HOST_IP
        self.SERVER_PORT = Socket_config.HOST_PORT
        # IP address of Rasberry Pi
        # self.HOST_IP = "192.168.31.52"
        # self.HOST_PORT = 50000
    
        print("Starting socket: TCP...")
        self.server_addr = (self.SERVER_IP, self.SERVER_PORT)
        # socket.AF_INET: IP protocol       socket.SOCK_STREAM: TCP Comm
        self.socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
        while True:
            try:
                print("Connecting to server @ %s:%d..." %(self.SERVER_IP, self.SERVER_PORT))
                self.socket_tcp.connect(self.server_addr)
                break
            except Exception:
                print("Can't connect to server, try it latter!")
                time.sleep(1)
                continue
        
        data_recv = self.socket_tcp.recv(512)       # buffer lenth: max 512 Bytes each time
        if len(data_recv)>0:
            dict_recv = json.loads(data_recv)
            print("Receive:", dict_recv)           # receive "Welcome to Pupper Rpi TCP server!"


    def subscriber_env(self) -> Dict:

        data = self.socket_tcp.recv(512)
        if len(data)>0:
            # dict
            dict_recv = json.loads(data)

            ### TODO feed pickle_recv to the RL Model ##########
            # print("State of the robot: ticks\n", dict_recv)   # current state of robot
            ##############################################

            return dict_recv


    def publisher_cmd(self, command: Dict):
        try:

            # command = {"servo_degree": np.array, "info": Dict}

            msg_cmd = json.dumps(command)
            self.socket_tcp.send(bytes(msg_cmd.encode('utf-8')))
            print("published...")

        except Exception:
            self.socket_tcp.close()
            self.socket_tcp=None
            print("publisher error")
            sys.exit(1)


    def __del__(self):
        self.socket_tcp.close()
        print("Terminate PC Socket Connection!")


if __name__ == "__main__":
    
    # Establish connection
    socket_PC = SocketInterface_PC()

    t0 = time.time()
    # PC -> Pi
    cmd = {"a":1, "b":2}
    print("PC send:", cmd)
    socket_PC.publisher_cmd(cmd)

    # Pi -> PC
    env = socket_PC.subscriber_env()
    print("PC receive:", env)

    t1 = time.time()
    # time period: 0.007 s
    print("time period of PC -> Pi -> PC:\n", t1 - t0,"s")



    # TODO reconnection if current connection breaks up
    # 
    # 
    # 

