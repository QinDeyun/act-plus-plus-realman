"""
a demo record the demo

Zezhi Liu 

2025.03.24

"""

from Robotic_Arm.rm_robot_interface import *
import time
import numpy as np
from datetime import datetime

# 实例化RoboticArm类
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
# 创建机械臂连接，打印连接id
handle = arm.rm_create_robot_arm("192.168.1.18", 8080)

# 机械臂移动到合适的位置（抓紧水壶）
arm.rm_movej([-9.262, 58.873, -85.67, 114.3, -40.46, 57.919, 314.559], 5, 0, 0, 1)

time.sleep(3)

Record_pose = []
Record_joints = []
Record_force = []

print(time.time())
# 初始的位置
arm.rm_movel([-0.023, -0.227, 0.321, -1.477, -1.143, 1.587])

# 抬高4厘米
arm.rm_movel([-0.023, -0.18, 0.321, -1.477, -1.143, 1.587])

# 开始倒水
arm.rm_movel([0.006, -0.155, 0.462, 1.791, -0.855, -1.832])

print(time.time())

# for i in range(100):
#     _, current_info= arm.rm_get_current_arm_state()

#     current_pose = current_info['pose']

#     print(current_pose)
    
#     # print(current_info)
#     current_joints =current_info['joint']

#     print(current_joints)

#     _, Force = arm.rm_get_force_data()

#     force_data = Force['force_data'][:]
#     print(force_data)

#     Record_pose.append(current_pose)
#     Record_joints.append(current_joints)
#     Record_force.append(force_data)

#     time.sleep(0.05)

# time.sleep(2)

# with open('DEMO_OPEN_RR_DOOR_RECORD_joints11.txt', 'w') as file:
#     for item in Record_joints:
#         file.write("%s\n" % item)

arm.rm_delete_robot_arm()