"""
a demo for REPLAY THE DEMO!!!! AND SAVE THE DATA

Zezhi Liu 

2025.3.24

"""

from Robotic_Arm.rm_robot_interface import *
import time
import numpy as np
import json
import scipy.signal as scsi

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs



# ATTENTION HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!# 

# the number of the demo !

demo_number = 11
# data_number = 250 * demo_number
data_number = 100 * demo_number
target_points_number = 10000

# 实例化RoboticArm类
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = arm.rm_create_robot_arm("192.168.1.18", 8080)


def depth_to_pointcloud(depth_image, intrinsic, target_points=10000):
    # Create Open3D Image from depth map
    o3d_depth = o3d.geometry.Image(depth_image)

    # Get intrinsic parameters
    fx, fy, cx, cy = intrinsic.fx, intrinsic.fy, intrinsic.ppx, intrinsic.ppy

    # Create Open3D PinholeCameraIntrinsic object
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=depth_image.shape[1], height=depth_image.shape[0], fx=fx,
                                                      fy=fy, cx=cx, cy=cy)

    # Create Open3D PointCloud object from depth image and intrinsic parameters
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, o3d_intrinsic)

    points = np.asarray(pcd.points)
    WORK_SPACE = [
                    [-0.4, 0.1],
                    [-0.4, 0],
                    [0.5, 0.95]
                ]

    # crop
    points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
                                (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
                                (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]


    # Remove outliers using statistical outlier removal
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=500, std_ratio=2.0)
    points = np.asarray(point_cloud.select_by_index(ind).points)
    # o3d.visualization.draw_geometries([point_cloud])
    pcd.points = o3d.utility.Vector3dVector(points)


    # # Downsample or upsample the point cloud to have a consistent number of points
    # if len(pcd.points) > target_points:
    #     print("2000000000000")
    #     pcd = pcd.uniform_down_sample(int(len(pcd.points) / target_points))
    # elif len(pcd.points) < target_points:
    #     pcd = pcd.voxel_down_sample(voxel_size=0.01)
    #     while len(pcd.points) < target_points:
    #         pcd.points.extend(pcd.points[:target_points - len(pcd.points)])

    # sample points
    if len(pcd.points) >= target_points:
        idxs = np.random.choice(len(pcd.points), target_points, replace=False)
    else:
        idxs1 = np.arange(len(pcd.points))
        idxs2 = np.random.choice(len(pcd.points), target_points-len(pcd.points), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    points = np.asarray(pcd.points)
    points = points[idxs]
    pcd.points = o3d.utility.Vector3dVector(points)
    
    point_cloud.points = o3d.utility.Vector3dVector(pcd.points)
    # o3d.visualization.draw_geometries([point_cloud])
    print(len(point_cloud.points))
    print(np.asarray(pcd.points))
    # o3d.visualization.draw_geometries(pcd.points)
    return pcd


def save_pointcloud(pcd, file_name):
    o3d.io.write_point_cloud(file_name, pcd)

# 初始化 RealSense 相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

object_path = './data/open_RR_door_data' + str(demo_number)
if not os.path.exists(object_path):
    os.makedirs(object_path)

depth_path = './data/open_RR_door_data' + str(demo_number) + '/depth'
if not os.path.exists(depth_path):
    os.makedirs(depth_path)

color_path = './data/open_RR_door_data' + str(demo_number) + '/color'
if not os.path.exists(color_path):
    os.makedirs(color_path)


npy_path = './data/open_RR_door_data' + str(demo_number) + '/npy'
if not os.path.exists(npy_path):
    os.makedirs(npy_path)

pcd_path = './data/open_RR_door_data' + str(demo_number) + '/pcd'
if not os.path.exists(pcd_path):
    os.makedirs(pcd_path)
    
j1 = []
j2 = []
j3 = []
j4 = []
j5 = []
j6 = []
j7 = []

with open('DEMO_OPEN_RR_DOOR_RECORD_joints' + str(demo_number)+ '.txt', 'r') as file:
    data = file.readlines()

# 解析数据并计算末端位置
i = 0
for line in data:
    point_data = json.loads(line)
    print(i, point_data)
    j1.append(point_data[0])
    j2.append(point_data[1])
    j3.append(point_data[2])
    j4.append(point_data[3])
    j5.append(point_data[4])
    j6.append(point_data[5])
    j7.append(point_data[6])
    i += 1

print(len(j1))

#============== tart the replay of the robot =======================

start_joints = [j1[0], j2[0], j3[0], j4[0], j5[0], j6[0], j7[0]]
joints = start_joints

arm.rm_movej(start_joints, 5, 0, 0, 1)

time.sleep(2)

Record_pose = []
Record_joints = []
Record_force = []

for i in range(len(j1)):
    #================== move =====================
    joints[0] = j1[i]
    joints[1] = j2[i]
    joints[2] = j3[i]
    joints[3] = j4[i]
    joints[4] = j5[i]
    joints[5] = j6[i]
    joints[6] = j7[i]
    print(i)
    # the canfd function can not work when the step delta positoin exceed 0.002
    arm.rm_movej_canfd(joints,False)

    #================== joints =====================
    _, current_info= arm.rm_get_current_arm_state()

    print(arm.rm_get_arm_all_state())


    print(current_info)
    current_pose = current_info['pose']

    print(current_pose)
    
    # print(current_info)
    current_joints =current_info['joint']

    print(current_joints)

    _, Force = arm.rm_get_force_data()

    force_data = Force['force_data'][:]
    print(force_data)

    Record_pose.append(current_pose)
    Record_joints.append(current_joints)
    Record_force.append(force_data)

    #====================== points clouds =====================

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    if not aligned_depth_frame:
        continue

    depth_frame = frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame:
        continue

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    pc = depth_to_pointcloud(depth_image, depth_intrinsics)

    # cv2.imshow('RealSense', color_image)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.008), cv2.COLORMAP_JET)
    # cv2.imshow('depth_color', depth_colormap)

    rgb_file_path = os.path.join(color_path, 'color'+str(data_number+i)+'.png')
    cv2.imwrite(rgb_file_path, color_image)
    print('color saved', rgb_file_path)

    # 保存深度图像
    # depth_file_path = os.path.join(object_path, 'depth{:d}.png'.format(counter))
    depth_file_path = os.path.join(depth_path, 'depth'+str(data_number+i)+'.png')
    cv2.imwrite(depth_file_path, depth_image)
    print('depth saved', depth_file_path)

    # 将点云保存为 pcd 文件
    # pcd_file_path = os.path.join(object_path, 'point_cloud{:d}.pcd'.format(counter))
    pcd_file_path = os.path.join(pcd_path, 'pcd'+str(data_number+i)+'.pcd')
    save_pointcloud(pc, pcd_file_path)
    print('pc saved', pcd_file_path)

    # 将点云保存为 npy 文件
    npy_file_path = os.path.join(npy_path, 'npy'+str(data_number+i)+'.npy')
    np.save(npy_file_path, np.asarray(pc.points))
    print('pc saved as npy', npy_file_path)
    print("!!!!!!!!!")
    print(np.asarray(pc.points))

    time.sleep(0.01)

time.sleep(2)

joints_path = './data/open_RR_door_data' + str(demo_number) + '/joints'
if not os.path.exists(joints_path):
    os.makedirs(joints_path)

pose_path = './data/open_RR_door_data' + str(demo_number) + '/pose'
if not os.path.exists(pose_path):
    os.makedirs(pose_path)

force_path = './data/open_RR_door_data' + str(demo_number) + '/force'
if not os.path.exists(force_path):
    os.makedirs(force_path)

with open(pose_path + '/DEMO_OPEN_RR_DOOR_RECORD_pose.txt', 'w') as file:
    for item in Record_pose:
        file.write("%s\n" % item)

with open(joints_path + '/DEMO_OPEN_RR_DOOR_RECORD_joints.txt', 'w') as file:
    for item in Record_joints:
        file.write("%s\n" % item)

with open(force_path + '/DEMO_OPEN_RR_DOOR_RECORD_force.txt', 'w') as file:
    for item in Record_force:
        file.write("%s\n" % item)

pipeline.stop()
arm.rm_delete_robot_arm()