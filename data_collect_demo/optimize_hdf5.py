import os
import numpy as np
import cv2
import h5py
import argparse

import matplotlib.pyplot as plt
from constants import DT

import IPython
e = IPython.embed

def pad_or_trim_to_length(data, target_length, padding_value=0):
    """
    Pad or trim the data to the target length.
    """
    current_length = len(data)
    if current_length > target_length:
        # Trim the data
        indices = np.random.choice(current_length, target_length, replace=False)
        data = [data[i] for i in sorted(indices)]
    elif current_length < target_length:
        raise ValueError(f"Data length {current_length} is less than target length {target_length}.")
    return data

def load_hdf5(dataset_dir, dataset_name, target_length=None):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        compressed = root.attrs.get('compress', False)
        qpos = root['/observations/qpos'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        if compressed:
            compress_len = root['/compress_len'][()]

    if compressed:
        for cam_id, cam_name in enumerate(image_dict.keys()):
            padded_compressed_image_list = image_dict[cam_name]
            image_list = []
            for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list):
                image_len = int(compress_len[cam_id, frame_id])
                compressed_image = padded_compressed_image
                image = cv2.imdecode(compressed_image, 1)
                image_list.append(image)
            image_dict[cam_name] = image_list

    # If target_length is specified, pad or trim the data
    if target_length is not None:
        qpos = pad_or_trim_to_length(qpos, target_length)
        action = pad_or_trim_to_length(action, target_length)
        for cam_name in image_dict.keys():
            image_dict[cam_name] = pad_or_trim_to_length(image_dict[cam_name], target_length)

    return qpos, action, image_dict

def main(args):
    dataset_dir = args['dataset_dir']
    for i in range(80):
        # episode_idx = args['episode_idx']
        episode_idx = i
        ismirror = args['ismirror']
        target_length = args.get('target_length', None)

        if ismirror:
            dataset_name = f'mirror_episode_{episode_idx}'
        else:
            dataset_name = f'episode_{episode_idx}'

        qpos, action, image_dict = load_hdf5(dataset_dir, dataset_name, target_length)
        print('hdf5 loaded!!')
        print(f'qpos shape: {np.array(qpos).shape}')
        print(f'action shape: {np.array(action).shape}')
        # print(f'image_dict shape: {image_dict}')
        for cam_name in image_dict.keys():
            print(f'{cam_name} shape: {np.array(image_dict[cam_name]).shape}')
        
        # Save the data back to the original HDF5 file
        output_path = os.path.join(dataset_dir + '/optimized', dataset_name + '.hdf5')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        max_timesteps = len(qpos)  # Assuming qpos length represents the number of timesteps
        with h5py.File(output_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            obs = root.create_group('observations')
            image_group = obs.create_group('images')
            for cam_name, images in image_dict.items():
                image_group.create_dataset(cam_name, data=np.array(images), dtype='uint8',
                            chunks=(1, 480, 640, 3),
                            compression="gzip")
            obs.create_dataset('qpos', data=np.array(qpos), compression="gzip")
            obs.create_dataset('qvel', (max_timesteps, 7))
            root.create_dataset('action', data=np.array(action), compression="gzip")
        print(f'Data saved to {output_path}')

        DT = 0.02

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    parser.add_argument('--ismirror', action='store_true')
    parser.add_argument('--target_length', action='store', type=int, help='Target length for uniformity.', required=False)
    main(vars(parser.parse_args()))