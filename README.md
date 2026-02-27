# act-plus-plus-realman

This repository extends **[act-plus-plus](https://github.com/MarkFzp/act-plus-plus)** with real-world support for **[Realman robotic arms](https://www.realman-robotics.cn/products/rmo)**, providing an end-to-end pipeline for **real-robot data collection → dataset post-processing → policy training (ACT / Diffusion / VINN) → real-robot deployment/inference**.

> Note: This repo contains both simulation-related code and real-robot code. The real-robot stack depends on the Realman arm SDK (`Robotic_Arm.rm_robot_interface`) and Intel RealSense (`pyrealsense2`), which you must install locally.

## What’s added / changed

- **Real-robot dataset collection**: added [data_collect_demo/](data_collect_demo/) (scripts + example data)
  - [data_collect_demo/record_episode_realman_real-world.py](data_collect_demo/record_episode_realman_real-world.py): records RealSense RGB + robot joint state and writes `episode_*.hdf5`
  - [data_collect_demo/optimize_hdf5.py](data_collect_demo/optimize_hdf5.py): re-saves episodes with gzip compression into an `optimized/` folder for training
  - [data_collect_demo/data_example/](data_collect_demo/data_example/): a sample episode in HDF5 format
- **Task config**: adds the `pour_water` task in [aloha_scripts/constants.py](aloha_scripts/constants.py)
  - Defines dataset path, episode length, camera names, train split ratio, etc.
  - Note: `pour_water` uses only the desktop camera `camera_rgb` (no wrist camera)
- **Real-robot deployment**: [imitate_episodes_deploy.py](imitate_episodes_deploy.py) includes RealSense + Realman online inference and sends joint commands via `rm_movej_canfd`

## Installation

### 1) Conda environment

Create the environment from [conda_env.yaml](conda_env.yaml) (Python 3.9 + PyTorch 2.0 + MuJoCo, etc.):

```bash
conda env create -f conda_env.yaml
conda activate aloha
```

Optional: install this repo in editable mode (for imports):

```bash
pip install -e .
```

### 2) Real-robot dependencies (optional)

To run real-robot collection/deployment, you need external dependencies (not shipped in this repo):

- Realman arm Python SDK that provides `Robotic_Arm.rm_robot_interface`
- Intel RealSense SDK + Python bindings (`pyrealsense2`)

If you only do offline training/evaluation (no robot connected), you can skip these.

## Quickstart (pour_water)

### Step 0: Configure dataset root and task

1) Update `DATA_DIR` in [aloha_scripts/constants.py](aloha_scripts/constants.py) to your local dataset root.

Current default:

```python
DATA_DIR = os.path.expanduser('~/QinDeyun/act/act-plus-plus-main/data_collect_demo')
```

2) Verify the `pour_water` config (camera names must match HDF5 keys exactly):

```python
'pour_water':{
    'dataset_dir': DATA_DIR + '/pour_water_data/optimized',
    'episode_len': 225,
    'camera_names': ['camera_rgb'],
    'train_ratio': 0.98,
},
```

### Step 1: Collect real-robot episodes (`episode_*.hdf5`)

Use [data_collect_demo/record_episode_realman_real-world.py](data_collect_demo/record_episode_realman_real-world.py).

What it does:

- Starts a RealSense RGB stream (default `640x480@30fps`)
- Reads robot joint state at ~`50Hz` (`rm_get_current_arm_state()`)
- Writes an HDF5 episode containing:
  - `/observations/qpos` (7-dim joint state)
  - `/action` (the script records “previous-step joints”, 7-dim)
  - `/observations/images/camera_rgb` (RGB frames, `uint8`)

You must edit the script’s top-level configuration for your setup:

- `COLOR_DIR`: directory for per-frame `png` dumps
- `DATASET_DIR`: output directory for `episode_*.hdf5`
- `EPISODE_ID`: episode index
- `arm.rm_create_robot_arm("192.168.1.18", 8080)`: robot IP/port

### Step 2: Post-process episodes into `optimized/` (gzip)

[data_collect_demo/optimize_hdf5.py](data_collect_demo/optimize_hdf5.py) reads `episode_*.hdf5` and re-saves them with gzip compression into `dataset_dir/optimized/`.

Example:

```bash
python data_collect_demo/optimize_hdf5.py --dataset_dir /path/to/pour_water_data
```

Output example:

```
/path/to/pour_water_data/optimized/episode_0.hdf5
```

### Step 3: Train (ACT / Diffusion / VINN)

Training entrypoint: [imitate_episodes.py](imitate_episodes.py). Example ACT command:

```bash
python imitate_episodes.py \
  --task_name pour_water \
  --ckpt_dir ckpt_dir_4_25 \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 50 \
  --hidden_dim 512 \
  --batch_size 128 \
  --dim_feedforward 3200 \
  --num_steps 20000 \
  --lr 8e-5 \
  --seed 0 \
  --eval_every 30000
```

Key outputs:

- `ckpt_dir_4_25/policy_best.ckpt`: best checkpoint
- `ckpt_dir_4_25/dataset_stats.pkl`: normalization stats used for deployment

> WandB: training calls `wandb.init(...)` by default. To avoid online logging, set `WANDB_MODE=offline` or comment out WandB calls.

### Step 4: Real-robot deployment (online camera + joint commands)

Deployment entrypoint: [imitate_episodes_deploy.py](imitate_episodes_deploy.py). Use `--eval` to run `eval_bc(...)`.

Example:

```bash
python imitate_episodes_deploy.py \
  --task_name pour_water \
  --ckpt_dir ckpt_dir_4_25 \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 50 \
  --hidden_dim 512 \
  --batch_size 128 \
  --dim_feedforward 3200 \
  --num_steps 20000 \
  --lr 8e-5 \
  --seed 0 \
  --eval
```

High-level logic:

- A RealSense thread continuously pushes the *latest* RGB frame into a queue
- A robot thread continuously pushes the *latest* joint state into a queue
- Each control step: read latest observation → policy forward → `post_process` (unnormalize) → send `arm.rm_movej_canfd(target_qpos, ...)`

You still need to adapt these to your hardware:

- Robot connection: `arm.rm_create_robot_arm("192.168.1.18", 8080)`
- Command API: currently uses `rm_movej_canfd(target_qpos, True, 0, 0, 0)` (test with speed limits + E-stop ready)

> Note: the deployment script contains an `IPython.embed()` in real-robot mode (interactive pause). Exit the shell to proceed, or remove/comment it if you want fully automatic runs.

## Dataset format (HDF5)

Training/deployment expects these keys (see `EpisodicDataset` in [utils.py](utils.py)):

- `/observations/qpos`: `(T, 7)`
- `/observations/qvel`: `(T, 7)` (some scripts write placeholders)
- `/action`: `(T, 7)`
- `/observations/images/<camera_name>`: e.g. `camera_rgb`, typically `(T, 480, 640, 3)`, `uint8`

`<camera_name>` must match the `camera_names` list in [aloha_scripts/constants.py](aloha_scripts/constants.py).

## Visualization / sanity checks

Visualize or load an episode:

```bash
python visualize_episodes.py --dataset_dir /path/to/pour_water_data/optimized --episode_idx 0
```

Example training loss curve:

![loss](static/loss_curve.png)

Real-robot demo video:

- [pour_water_demo.mov](static/pour_water_demo.mov)

## Repository layout (partial)

- [data_collect_demo/](data_collect_demo/): Realman real-robot data collection and processing
- [aloha_scripts/](aloha_scripts/): task configs and robot-related utilities
- [imitate_episodes.py](imitate_episodes.py): training entrypoint (behavior cloning)
- [imitate_episodes_deploy.py](imitate_episodes_deploy.py): real-robot deployment/inference entrypoint
- [utils.py](utils.py): HDF5 loading, normalization stats, dataloaders

## Troubleshooting

- **Dataset not found / path errors**: check `DATA_DIR` and `TASK_CONFIGS[task_name]['dataset_dir']` in [aloha_scripts/constants.py](aloha_scripts/constants.py).
- **Camera name mismatch**: your HDF5 must contain `/observations/images/<key>` for each entry in `camera_names` (e.g. `camera_rgb`).
- **Cannot import `Robotic_Arm`**: it’s an external SDK not included in this repo; install it and ensure it’s on your Python path.
- **Deployment seems stuck**: check whether [imitate_episodes_deploy.py](imitate_episodes_deploy.py) is waiting in `IPython.embed()`.

## Acknowledgements

This repository is built on top of the ideas and code structure from [act-plus-plus](https://github.com/MarkFzp/act-plus-plus). Thanks to the original authors and community.
