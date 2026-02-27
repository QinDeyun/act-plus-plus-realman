"""Demo script for recording a episode with RGB frames + robot joints.

Author: Deyun Qin
Date: 2025-03-24
"""

from Robotic_Arm.rm_robot_interface import *

import os
import queue
import threading
import time

import cv2
import h5py
import numpy as np
import pyrealsense2 as rs


# Output configuration
COLOR_DIR = "/home/zezhi/QinDeyun/act/act-plus-plus-main/data_collect_demo/color"
DATASET_DIR = "/home/zezhi/QinDeyun/act/act-plus-plus-main/data_collect_demo"
EPISODE_ID = 0


# Queues for producer/consumer threads
frame_queue = queue.Queue()
action_queue = queue.Queue()
pos_queue = queue.Queue()


# In-memory buffers
data_dict = {
    "/observations/qpos": [],
    "/action": [],
    "/observations/images/camera_rgb": [],
}


start_collect = False
stop_threads = False
stop_threads_2 = False


def get_frames():
    # Initialize RealSense pipeline.
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    i = 0

    try:
        while True:
            if start_collect:
                if i != 0:
                    _, current_info = arm.rm_get_current_arm_state()
                    action_queue.put(current_info["joint"])

                if stop_threads:
                    break

                # Wait for the next RGB frame.
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                # Push the frame to the queue for the consumer thread.
                frame_queue.put(color_frame)

                _, current_info = arm.rm_get_current_arm_state()
                current_joints = current_info["joint"]
                print(current_joints)
                pos_queue.put(current_joints)

                time.sleep(0.02)  # Add a small delay to reduce CPU usage.

                i += 1
            
    
    finally:
        pipeline.stop()


def process_frames():
    i = 0
    while True:
        # Consume frames and states from queues.
        if not frame_queue.empty():
            color_frame = frame_queue.get()
            if color_frame:
                # Convert frame to numpy array.
                img = np.asanyarray(color_frame.get_data())

                # Visualize RGB image.
                cv2.imshow("RGB", img)

                rgb_file_path = os.path.join(COLOR_DIR, f"color{i}.png")
                cv2.imwrite(rgb_file_path, img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                print("Frame processed:", img.shape)
                i += 1

        if not pos_queue.empty():
            pos = pos_queue.get()
            if pos:
                data_dict["/observations/qpos"].append(pos)
        if not action_queue.empty():
            action = action_queue.get()
            if action:
                data_dict["/action"].append(action)

        if stop_threads_2:
            cv2.destroyAllWindows()
            break
        
        time.sleep(0.02)

# Instantiate RoboticArm.
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
# Create robot connection.
handle = arm.rm_create_robot_arm("192.168.1.18", 8080)

# Move to initial position from the end of the previous episode
# Move up 4 cm.
arm.rm_movel([-0.023, -0.18, 0.321, -1.85, -1.143, 1.587], 5, 0, 0, 1)
# Move the arm to a suitable pose.
arm.rm_movej([-9.262, 58.873, -85.67, 114.3, -40.46, 57.919, 314.559], 5, 0, 0, 1)

# Start frame producer thread.
frame_thread = threading.Thread(target=get_frames)
frame_thread.daemon = True
frame_thread.start()

# Start consumer thread.
process_thread = threading.Thread(target=process_frames)
process_thread.daemon = True
process_thread.start()

# Wait for a few seconds to warm-up the camera.
time.sleep(3)
start_collect = True

a = time.time()
print(a)

# initial pose
arm.rm_movel([-0.023, -0.227, 0.321, -1.477, -1.143, 1.587], 5, 0, 0, 1)

# waypoint
arm.rm_movel([-0.023, -0.18, 0.321, -1.477, -1.143, 1.587], 5, 0, 0, 1)

# Manually measured cup position
target_x = 0.126
target_y = -0.135
target_z = 0.56

# Move to target pose.
arm.rm_movel([target_x, target_y, target_z - 0.12, -1.477, -1.143, 1.587], 5, 0, 0, 1)

# Start pouring.
arm.rm_movel([target_x, target_y, target_z, 1.791, -0.855, -1.832], 5, 0, 0, 1)

print(time.time() - a)

stop_threads = True
frame_thread.join()

time.sleep(2)
stop_threads_2 = True
frame_thread.join()

arm.rm_delete_robot_arm()

max_timesteps = -1

# Print collected array shapes.
for key, value in data_dict.items():
    if isinstance(value, list) and len(value) > 0:
        print(f"{key}: {np.array(value).shape}")
        max_timesteps = np.array(value).shape[0]
    else:
        print(f"{key}: Empty or not a list")

for j in range(max_timesteps):
    rgb_file_path = os.path.join(COLOR_DIR, f"color{j}.png")
    rgb = cv2.imread(rgb_file_path)
    data_dict["/observations/images/camera_rgb"].append(rgb)

print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
# Print collected array shapes again (after loading images).
for key, value in data_dict.items():
    if isinstance(value, list) and len(value) > 0:
        print(f"{key}: {np.array(value).shape}")
        max_timesteps = np.array(value).shape[0]
    else:
        print(f"{key}: Empty or not a list")

# Save as HDF5.
t0 = time.time()
dataset_path = os.path.join(DATASET_DIR, f"episode_{EPISODE_ID}")
with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
    # root.attrs['sim'] = True
    obs = root.create_group('observations')
    image = obs.create_group('images')
    _ = image.create_dataset(
        "camera_rgb",
        (max_timesteps, 480, 640, 3),
        dtype="uint8",
        chunks=(1, 480, 640, 3),
    )
    # compression='gzip',compression_opts=2,)
    # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
    qpos = obs.create_dataset('qpos', (max_timesteps, 7))
    qvel = obs.create_dataset('qvel', (max_timesteps, 7))
    action = root.create_dataset('action', (max_timesteps, 7))

    for name, array in data_dict.items():
        root[name][...] = array
print(f'Saving: {time.time() - t0:.1f} secs\n')


