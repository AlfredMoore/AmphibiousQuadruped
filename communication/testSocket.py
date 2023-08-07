from SocketInterface_Pi_v3 import SocketInterface_Pi
socket_Pi = SocketInterface_Pi()  # server

cmd_recv = socket_Pi.subscriber_cmd()
print("Pi receive:",cmd_recv)