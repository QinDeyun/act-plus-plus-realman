"""
a demo record the demo

Zezhi Liu 

2025.03.24

"""

from Robotic_Arm.rm_robot_interface import *
import time
import numpy as np
from datetime import datetime
import threading

import json
import scipy.signal as scsi

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

import h5py

def record_data():

    data_dict = {
        '/observations/qpos': [],
        '/action': [],
        '/observations/images/camera_1': [],
    }



    i = 0
    while True:
        if i != 0:
            _, current_info= arm.rm_get_current_arm_state()
            current_joints =current_info['joint']
            data_dict['/action'].append(current_info['joint'])
        global stop_thread
        if stop_thread:




            print("nums: ", i)
            cv2.destroyAllWindows()
            pipeline.stop()
            arm.rm_delete_robot_arm()

            max_timesteps = i

            # HDF5
            t0 = time.time()
            dataset_path = os.path.join("/home/zezhi/QinDeyun/act/act-plus-plus-main/data_collect_demo", f'episode_{4}')
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                # root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                _ = image.create_dataset("camera_1", (max_timesteps, 480, 640, 3), dtype='uint8',
                                        chunks=(1, 480, 640, 3), )
                # compression='gzip',compression_opts=2,)
                # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
                qpos = obs.create_dataset('qpos', (max_timesteps, 7))
                qvel = obs.create_dataset('qvel', (max_timesteps, 7))
                action = root.create_dataset('action', (max_timesteps, 7))

                for name, array in data_dict.items():
                    root[name][...] = array
            print(f'Saving: {time.time() - t0:.1f} secs\n')

            
            break

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow('color', color_image)
        # rgb_file_path = os.path.join(color_path, 'color'+str(data_number+i)+'.png')
        # cv2.imwrite(rgb_file_path, color_image)
        # print('color saved', rgb_file_path)
        # Visualize the captured color image
        # plt.imshow(color_image)
        # plt.title(f"Frame {i}")
        # plt.axis('off')
        # plt.show()
        # plt.close()


        _, current_info= arm.rm_get_current_arm_state()
        current_joints =current_info['joint']
        print(current_joints)
        Record_joints.append(current_joints)
        i += 1

        data_dict['/observations/qpos'].append(current_info['joint'])
        data_dict['/observations/images/camera_1'].append(color_image)

        time.sleep(0.05)



# 实例化RoboticArm类
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
# 创建机械臂连接，打印连接id
handle = arm.rm_create_robot_arm("192.168.1.18", 8080)

# 初始化 RealSense 相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

# 高4厘米
arm.rm_movel([-0.023, -0.18, 0.321, -1.85, -1.143, 1.587], 5, 0, 0, 1)
# 机械臂移动到合适的位置（抓紧水壶）
arm.rm_movej([-9.262, 58.873, -85.67, 114.3, -40.46, 57.919, 314.559], 5, 0, 0, 1)

time.sleep(3)

Record_pose = []
Record_joints = []
Record_force = []

a = time.time()
print(a)

stop_thread = False
thread = threading.Thread(target = record_data)
thread.start()

# 初始的位置
arm.rm_movel([-0.023, -0.227, 0.321, -1.477, -1.143, 1.587], 5, 0, 0, 1)

# 抬高4厘米
arm.rm_movel([-0.023, -0.18, 0.321, -1.477, -1.143, 1.587], 5, 0, 0, 1)

target_x = 0.126
target_y = -0.145
target_z = 0.56
# 移动到对应位置
arm.rm_movel([target_x, target_y, target_z - 0.12, -1.477, -1.143, 1.587], 5, 0, 0, 1)

# 开始倒水
arm.rm_movel([target_x, target_y, target_z, 1.791, -0.855, -1.832], 5, 0, 0, 1)

print(time.time() - a)

stop_thread = True

# time.sleep(2)

# with open('DEMO_OPEN_RR_DOOR_RECORD_joints11.txt', 'w') as file:
#     for item in Record_joints:
#         file.write("%s\n" % item)

