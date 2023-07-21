import numpy as np
import time
from src.IMU import IMU
from src.Controller import Controller
# from src.JoystickInterface import JoystickInterface
from testComms.SocketInterface_Pi_v3 import SocketInterface_Pi
from src.State import State
from pupper.HardwareInterface import HardwareInterface
from pupper.Config import Configuration
from pupper.Kinematics import four_legs_inverse_kinematics

def robot(use_imu=False):
    """Main program
    """

    # Create config
    config = Configuration()
    hardware_interface = HardwareInterface()
    # hardware_interface.set_actuator_postions(hardware_interface.servo_params.neutral_angle_degrees)

    # print("hardware interface established")
    # Create imu handle
    if use_imu:
        imu = IMU(port="/dev/ttyACM0")
        imu.flush_buffer()

    # Create controller and user input handles
    controller = Controller(
        config,
        four_legs_inverse_kinematics,
    )
    state = State()
    # print("Creating joystick listener...")exit
    # joystick_interface = JoystickInterface(config)
    socket_Pi = SocketInterface_Pi(HOST_IP="192.168.31.101", HOST_PORT=50000)
    # print("Done.")

    last_loop = time.time()

    # print("Summary of gait parameters:")
    # print("overlap time: ", config.overlap_time)
    # print("swing time: ", config.swing_time)
    # print("z clearance: ", config.z_clearance)
    # print("x shift: ", config.x_shift)

    # Wait until the activate button has been pressed
    while True:
        print("Waiting for L1 to activate robot.")

        while True:
            # command = joystick_interface.get_command(state)
            # print(command.activate_event)
            # joystick_interface.set_color(config.ps4_deactivated_color)

            msg_recv = socket_Pi.subscriber_cmd()
            command = socket_Pi.get_command(state, msg_recv)

            if command.activate_event == 1:
                break
            print("You should firstly activate the Robot!")
            time.sleep(0.1)
        print("Robot activated.")
        # joystick_interface.set_color(config.ps4_color)

        while True:
            now = time.time()
            # 0.01s between two commands
            if now - last_loop < config.dt:
                continue
            last_loop = time.time()

            # Parse the udp joystick commands and then update the robot controller's parameters

            print("Waiting for msg...")
            msg_recv = socket_Pi.subscriber_cmd()
            print("Pi receive:", msg_recv)
            command = socket_Pi.get_command(state, msg_recv)

            if command.activate_event == 1:
                print("Deactivating Robot")
                break

            # Read imu data. Orientation will be None if no data was available
            quat_orientation = (
                imu.read_orientation() if use_imu else np.array([1, 0, 0, 0])
            )
            state.quat_orientation = quat_orientation

            # Step the controller forward by dt
            controller.run(state, command)

            # Update the pwm widths going to the servos
            print("jointly angles:",state.joint_angles)
            hardware_interface.set_actuator_postions(state.joint_angles)