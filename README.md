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

## Step 0: Drone-Side Setup

These commands run **on the drone** (e.g. `ssh lis@nxtN.local`). Make sure your calibration PC and the drone share the same `ROS_DOMAIN_ID` so topics are visible from both sides.

### 0a. Start the camera driver (calibration mode)

Launch the calibration-specific driver, which publishes an assembled compressed image at `/oak_ffc_4p_driver_node/compressed` at 5 Hz with quality settings tuned for calibration:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_swarmnxt_ws/install/setup.bash
ros2 launch oak_ffc_4p_driver_ros2 oak_ffc_4p_driver_calibration.launch.py
```

### 0b. (Only if calibrating IMU) Start MicroXRCE-DDS Agent

Skip this section if you're not using `--record-imu`. The agent bridges PX4 ↔ ROS 2 over the FCU's serial link (`/dev/ttyTHS1` at 3 Mbaud) so the drone can publish `/fmu/out/sensor_combined`. The swarm-nxt Ansible playbooks install it as a `systemd --user` service, so the normal flow is:

```bash
systemctl --user status micro_xrce      # check whether it's already running
systemctl --user start  micro_xrce      # start if it isn't
```

The service runs (per `templates/micro_xrce_service.j2`):

```
chrt -f 98 MicroXRCEAgent serial --dev /dev/ttyTHS1 -b 3000000
```

(realtime priority 98, pinned to CPU 0).

**Fallback if the service isn't installed.** If `systemctl --user status micro_xrce` says "not loaded", run the agent manually in a terminal — `lis` is already in `dialout` so no `sudo` is needed:

```bash
MicroXRCEAgent serial --dev /dev/ttyTHS1 -b 3000000
```

Leave that terminal open while calibrating; `Ctrl+C` to stop.

**Verify the IMU stream is live and at the expected rate:**

```bash
ros2 topic hz /fmu/out/sensor_combined
```

The actual rate depends on the `rate_limit` field in PX4's `dds_topics.yaml` (see *XRCE-DDS Topic Configuration* in `swarm-nxt/docs/drone-setup.md`). With `rate_limit: 300.` the achievable rate is **333 Hz** (the agent rounds the poll interval down to 3 ms). If you instead see ~100 Hz, the FCU is on stock-rate firmware — reflash with the patched YAML before continuing, otherwise the IMU calibration will be rate-limited and translation observability will suffer.

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
- `--record-imu` – Also record `/fmu/out/sensor_combined` into every per-stage bag and run `kalibr_calibrate_imu_camera` per mono camera at the end (see *IMU-camera calibration* below)
- `--imu-topic` – PX4 IMU topic to subscribe to (default: `/fmu/out/sensor_combined`)
- `--imu-yaml` – Kalibr IMU YAML with Allan variance + topic + rate (default: `imu.yaml`)

---

## IMU-camera calibration

Pass `--record-imu` to extend the standard recording with synchronized IMU samples. Inside each per-stage bag (`cam_0_mono`, `pair_0_1`, …) you'll get an additional `/imu/data_raw` (`sensor_msgs/Imu`) topic in parallel with the camera frames. The PX4 `sensor_combined` message is converted in-process to `sensor_msgs/Imu` (FRD body frame) so Kalibr can consume it directly.

After intrinsics + extrinsics finish, the pipeline runs `kalibr_calibrate_imu_camera` once per mono bag (4 runs total). This gives 4 independent estimates of `T_cam_imu` that should agree once composed with the cam-cam extrinsics — a useful self-consistency check.

### Prerequisites

- **`px4_msgs` available in your ROS 2 environment.** Source the swarm-nxt workspace (`source ~/ros2_swarmnxt_ws/install/setup.bash`) before running, or the IMU subscriber will refuse to start with a clear error.
- **PX4 publishing `sensor_combined` at a usable rate.** Stock PX4 caps `/fmu/out/sensor_combined` at the uxrce_dds poll-loop ceiling (~100 Hz). Bump it via the `rate_limit: 300.` field on that topic in `dds_topics.yaml` and reflash. See `swarm-nxt/docs/drone-setup.md` → *XRCE-DDS Topic Configuration*.
- **`imu.yaml` with valid noise values.** A starter file lives in this repo, seeded from quarterKalibr's published values (same chip family). Replace with your own Allan variance result for fleet-grade calibration; otherwise expect Kalibr's reported uncertainties to be optimistic.
- **The motion you record must excite the IMU.** Camera-only calibration uses slow, careful motion. IMU calibration needs sustained 6-DOF excitation. Either include a short motion-rich phase per camera, or expect translation in `T_cam_imu` to be poorly constrained.

### Run

```bash
source ~/ros2_swarmnxt_ws/install/setup.bash   # for px4_msgs
python3 calibrate_physical.py --record-imu
```

### Outputs

Per camera, in `calibration_output/cam_{i}/`:
- `log1-camchain.yaml` – intrinsics (TartanCalib, unchanged)
- `*-camchain-imucam.yaml` – Kalibr's cam+IMU result, including `T_cam_imu` and the estimated time offset
- `*-results-imucam.txt` – residual report

Cross-check by composing `T_cam0_imu` × `T_cam1_cam0` and comparing to the independent `T_cam1_imu` estimate.

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

