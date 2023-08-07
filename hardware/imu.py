import smbus
import time
# import math
# import types
# import ctypes


class ImuInterface_I2C(object):
    def __init__(self, addr=0x50, i2cbus=0):
        self.addr = addr
        # i2c_0
        self.i2c = smbus.SMBus(i2cbus)
        GRAVITY = 9.8
        self.buffer = {"acceleration":0x34, "angular velocity":0x37, 
                       "RollPitchYaw":0x3d, "Quaterions":0x51}
        self.data_range = {"acceleration":16 * GRAVITY, "angular velocity":2000, 
                           "RollPitchYaw":180, "Quaterions":1}

        
    def get_data_xyz(self, package_head: int, data_range: int):
        """
        get data of axis xyz
        package_head ( # of register ): hexadecimal int in dict self.buffer
        data range: ranges in dic self.data_range
        """
        package_heads = (package_head, package_head+1, package_head+2)
        try:
            raw_data_x = self.i2c.read_i2c_block_data(self.addr, package_heads[0], 2)
            raw_data_y = self.i2c.read_i2c_block_data(self.addr, package_heads[1], 2)
            raw_data_z = self.i2c.read_i2c_block_data(self.addr, package_heads[2], 2)
        
        except IOError:
            print("ReadError:xyz")
            return (0,0,0)
        
        else:

            data_x = (raw_data_x[1] << 8 | raw_data_x[0]) / 32768 * data_range
            data_y = (raw_data_y[1] << 8 | raw_data_y[0]) / 32768 * data_range
            data_z = (raw_data_z[1] << 8 | raw_data_z[0]) / 32768 * data_range
            
            if data_x >= data_range:
                data_x -= 2 * data_range
            if data_y >= data_range:
                data_y -= 2 * data_range
            if data_z >= data_range:
                data_z -= 2 * data_range
            
            return (data_x, data_y, data_z)
        
    
    def get_data_quaterions(self, package_head: int, data_range: int):
        """
        get quaterions
        what is the range of quaterions?
        """
        # package_head = 0x51
        package_heads = (package_head, package_head+1, package_head+2, package_head+3)
        
        
        try:
            raw_data_q0 = self.i2c.read_i2c_block_data(self.addr, package_heads[0], 2)
            raw_data_q1 = self.i2c.read_i2c_block_data(self.addr, package_heads[1], 2)
            raw_data_q2 = self.i2c.read_i2c_block_data(self.addr, package_heads[2], 2)
            raw_data_q3 = self.i2c.read_i2c_block_data(self.addr, package_heads[3], 2)
            
        except IOError:
            print("ReadError:Quaterions")
            return (0,0,0)
        
        else:
            # data_range = 1
            data_q0 = (raw_data_q0[1] << 8 | raw_data_q0[0]) / 32768 * data_range
            data_q1 = (raw_data_q1[1] << 8 | raw_data_q1[0]) / 32768 * data_range
            data_q2 = (raw_data_q2[1] << 8 | raw_data_q2[0]) / 32768 * data_range
            data_q3 = (raw_data_q3[1] << 8 | raw_data_q3[0]) / 32768 * data_range
            
            
            if data_q0 >= data_range:
                data_q0 -= 2 * data_range
            if data_q1 >= data_range:
                data_q1 -= 2 * data_range
            if data_q2 >= data_range:
                data_q2 -= 2 * data_range
            if data_q3 >= data_range:
                data_q3 -= 2 * data_range
            
            return (data_q0, data_q1, data_q2, data_q3)
            
            
if __name__ == "__main__":
    
    import time
    
    imu_interface = ImuInterface_I2C(addr=0x50, i2cbus=0)
    for _ in range(100):
        data = imu_interface.get_data_xyz(package_head= imu_interface.buffer["acceleration"], data_range= imu_interface.data_range["acceleration"])
        print(data)
        time.sleep(1)