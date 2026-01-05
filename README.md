# Quadcam Calibration Toolbox

A minimal, automated toolkit to calibrate a 4-camera fisheye system (e.g., **OAK-FFC-4P**) and generate rectification maps for stereo depth estimation of the OmniNxt system. This repo was inspired by this [repo](https://github.com/UAV-Swarm/tools-quarterKalibr).

This toolkit streamlines the process into two steps:

1. **Physical Calibration**
   Records ROS 2 data and uses **TartanCalib (Docker)** to solve intrinsics and extrinsics.

2. **Map Generation**
   Analytically generates virtual stereo cameras, computes rectification, merges lookup tables, and visualizes results for verification.

> [!IMPORTANT]
> **Lens Stability is Critical:** Ensure all lenses are firmly seated in their mounts with zero play. Even slight lateral wobble (left/right movement) can translate to multiple pixels of error, severely degrading calibration accuracy and depth estimation quality.

---

## Prerequisites

### 1. Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. System Dependencies

**ROS 2 (Humble recommended)**
Source your environment:

```bash
source /opt/ros/humble/setup.bash
```

**Docker**
Required for running the TartanCalib calibration solver.

---

## Step 0: Start Camera Driver

Before starting the calibration scripts, launch the camera driver [`oak_ffc_4p_driver`](https://github.com/lis-epfl/oak_ffc_4p_driver_ros2) to publish the required image topics. Make sure that your drone and your PC where you're doing the calibration have the same `ROS_DOMAIN_ID` to be able to subscribe to the image topic. You need to launch:

```bash
ros2 launch oak_ffc_4p_driver_ros2 oak_ffc_4p_driver_calibration.launch
```

---

## Step 1: Record & Calibrate

Run `calibrate_physical.py` to record a calibration sequence and solve for the physical parameters of the rig:

```bash
python3 calibrate_physical.py 
```

### Interactive Recording

A window will open showing the camera feed (4-image grid).

**Press `SPACE`** to advance through stages:

The order or recording goes like this for easy setup: mono0, pair01, mono1, pair12, mono2, pair23, mono3, pair30.

**Press `Q`** to finish recording.
The script will automatically launch Docker to run the solver.

![Calibration Visualization](imgs/calibration.gif)

After the calibration is done, check the pdf files in `calibration_output/cam_*` and make sure the reprojection error is less then 1.5 pixels. If not, you can recalibrate the camera/pair in question using the `--only` argument (see arguments below).

### Arguments

- `--out` – Output directory (default: `calibration_output`)
- `--topic` – ROS 2 image topic (default: `/oak_ffc_4p_driver_node/compressed`)
- `--uncompressed` – Use if topic publishes raw `sensor_msgs/Image`
- `--only` – Calibrate specific targets only (e.g., `--only mono0,pair01`)
- `--skip-recording` – Re-calibrate using existing bags only

---

## Step 2: Generate Maps & Verify

Run the map generation script:

```bash
python3 generate_maps.py --width 224 --height 224 --hfov 110
```

### Visual Verification

For each stereo pair, a window will show:

Top: **Raw input pair**
Bottom: **Rectified pair** with *green horizontal epipolar lines*

You should verify that:

- Green lines are perfectly horizontal
- Features align across left/right views

Press any key to continue to the next pair.

### Arguments

- `--calib-dir` – Input calibration folder (default: `calibration_output`)
- `--out-dir` – Output directory for final maps (default: `final_maps`)
- `--width, --height` – Final rectified resolution (default: 224x224)
- `--hfov` – Horizontal FoV for virtual stereo (default: 110°)

---

## Step 3: Integration

The `final_maps/` directory will contain all the configuration files required by the depth estimation node.

### Files Generated Per Stereo Pair

- `final_rectified_to_fisheye_map_{i}_{j}.yml` – OpenCV remap lookup tables
- `final_map_config_{i}_{j}.yaml` – Calibration metadata (intrinsics, baseline, poses)

### Deploy

Copy the contents of:

```
final_maps/
```

to your [depth estimation](https://github.com/lis-epfl/depth_estimation_ros2) package’s config directory, e.g.:

```
depth_estimation_ros2/config/final_maps_224_224
```

## TODO
Add calibration between cameras and IMU for VIO.
