# Quadcam Calibration Toolbox

A minimal, automated toolkit to calibrate a 4-camera fisheye system (e.g., **OAK-FFC-4P**) and generate rectification maps for stereo depth estimation of the OmniNxt system. This repo was inspired by this [repo](https://github.com/UAV-Swarm/tools-quarterKalibr).

This toolkit streamlines the process into two steps:

1. **Physical Calibration**
   Records ROS 2 data and uses **TartanCalib (Docker)** to solve intrinsics and extrinsics.

2. **Map Generation**
   Analytically generates virtual stereo cameras, computes rectification, merges lookup tables, and visualizes results for verification.

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

Before starting the calibration scripts, launch the camera driver [`oak_ffc_4p_driver`](https://github.com/lis-epfl/oak_ffc_4p_driver_ros2) to publish the required image topics. Make sure that your drone and your PC where you're doing the calibration have the same `ROS_DOMAIN_ID` to be able to subscribe to the image topic.

**Crucial:** Ensure your driver configuration (`cam_config.yaml` or equivalent parameters) matches the settings below.
`synс_master` must be set, and `compress_images: true` should be enabled for efficient recording.

```yaml
oak_ffc_4p_driver_node:
  ros__parameters:
    sync_master: "CAM_A"              # master clock: "CAM_A", "CAM_B", "CAM_C", "CAM_D"
    resolution: "800"
    fps: 10
    rgb: true
    auto_exposure_time: true
    exposure_time_us: 10000
    iso: 400
    image_info: true
    auto_awb: true
    awb_value: 4000
    sharpness_calibration_mode: false
    enable_upside_down: false
    max_size_fps: 10
    publish_cams_individually: false
    compress_images: true
    jpeg_quality: 90
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
python3 generate_maps.py --width 320 --height 240 --hfov 100
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
- `--width, --height` – Final rectified resolution (default: 320×240)
- `--hfov` – Horizontal FoV for virtual stereo (default: 100°)

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

to your depth estimation package’s config directory, e.g.:

```
depth_estimation_ros2/config/final_maps_240_320
```

## TODO
Add calibration between cameras and IMU for VIO.
