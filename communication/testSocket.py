from communication.SocketInterface_PC_RL import SocketInterface_PC
socket_PC = SocketInterface_PC()  # server

# action
# 1. send a cmd that indicates what's next
cmd = {"next":Socket_config.next_action, "action": None, "info": None, "message": None}
socket_PC.publisher_cmd( command=cmd )

# 2. send action or receive state
cmd = {"next": None, "action": action, "info": None, "message": None}
socket_PC.publisher_cmd( command=cmd )


# state
# 1. send a cmd that indicates what's next
cmd = {"next":Socket_config.next_state, "action": None, "info": None, "message": None}
socket_PC.publisher_cmd( command=cmd )

# 2. send action or receive state
msg = socket_PC.subscriber_env()
state = msg["imu_data"]