#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import numpy as np
import cv2
import argparse
from pathlib import Path
try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    print("Please install scipy: pip install scipy")
    sys.exit(1)

# ROS2 Imports (Visual verification only)
try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image
except ImportError:
    pass

# ==========================================
# PART 1: MATH & PROJECTIONS
# ==========================================

def _create_y_rotation_matrix(angle_rad: float) -> np.ndarray:
    """Returns the rotation matrix that transforms Virtual Ray -> Physical Ray"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

def projectPoints_fisheye(pts3d, K, xi, D):
    n = pts3d.shape[0]
    k1, k2, p1, p2 = D[0], D[1], D[2], D[3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    s = K[0, 1]

    norm_Xc = np.linalg.norm(pts3d, axis=1)
    valid = norm_Xc > 1e-9

    pts2d = np.zeros((n, 2), dtype=np.float32)
    Xs = pts3d[valid] / norm_Xc[valid, None]
    denom = Xs[:, 2] + xi

    valid_denom = np.abs(denom) > 1e-9

    xu = Xs[valid_denom, 0] / denom[valid_denom]
    xv = Xs[valid_denom, 1] / denom[valid_denom]

    r2 = xu*xu + xv*xv
    r4 = r2*r2
    rad_dist = (1 + k1*r2 + k2*r4)

    xd = xu * rad_dist + 2*p1*xu*xv + p2*(r2 + 2*xu**2)
    yd = xv * rad_dist + p1*(r2 + 2*xv**2) + 2*p2*xu*xv

    u = fx * xd + s * yd + cx
    v = fy * yd + cy

    pts2d_valid = np.stack([u, v], axis=1)
    pts2d_final = np.zeros((n, 2), dtype=np.float32)
    full_mask = np.zeros(n, dtype=bool)
    full_mask[valid] = valid_denom
    pts2d_final[full_mask] = pts2d_valid

    return pts2d_final, full_mask

class VirtualStereoGen:
    def __init__(self, calib_dir, width, height, hfov):
        self.calib_dir = Path(calib_dir)
        self.W = width
        self.H = height
        self.HFOV = hfov
        # Store absolute poses of all physical cameras relative to Cam0
        self.cam_poses_in_cam0 = self._load_full_chain()

    def _load_full_chain(self):
        """Calculates T_cam_i_to_cam0 for all 4 cameras."""
        poses = {0: np.eye(4)}

        # We need the chain 0->1, 1->2, 2->3.
        # These are usually stored in the pair folders.
        pairs_to_trace = [(0,1), (1,2), (2,3)]

        current_pose = np.eye(4) # Pose of current cam in Cam0

        for i, next_i in pairs_to_trace:
            chain_path = self.calib_dir / f"cam_{i}_{next_i}" / "log1-camchain.yaml"
            if not chain_path.exists():
                print(f"[Warn] Chain {i}->{next_i} missing. Poses may be wrong.")
                poses[next_i] = np.eye(4)
                continue

            with open(chain_path) as f:
                d = yaml.safe_load(f)

            # T_cn_cnm1 is T_{current}_{prev}.
            # For pair 0-1, cam1 is current, cam0 is prev.
            # So T_1_0 = T_cn_cnm1.
            # Pose of 1 in 0 = T_1_0 ? No.
            # Point in 1 = T_1_0 * Point in 0? No.
            # Kalibr notation T_A_B usually transforms point in B to point in A.
            # We want Pose of i, which is T_world_i (Point in i -> Point in World/Cam0).
            # If T_1_0 transforms 0->1, then T_world_1 = T_world_0 * inv(T_1_0).
            # Let's assume standard Extrinsics: T in yaml is 4x4.

            T_next_curr = np.array(d['cam1']['T_cn_cnm1']) # T_{i+1}_{i}

            # Pose_{i+1} = Pose_{i} @ inv(T_{i+1}_{i})
            # Check: P_{i+1} = T_{i+1}_{i} * P_{i}
            # P_{world} = Pose_{i} * P_{i}
            # P_{world} = Pose_{i+1} * P_{i+1}
            # -> Pose_{i} * P_{i} = Pose_{i+1} * T_{i+1}_{i} * P_{i}
            # -> Pose_{i} = Pose_{i+1} * T_{i+1}_{i}
            # -> Pose_{i+1} = Pose_{i} * inv(T_{i+1}_{i})

            next_pose = current_pose @ np.linalg.inv(T_next_curr)
            poses[next_i] = next_pose
            current_pose = next_pose

        # For 3->0 loop closure check? Not strictly needed for open chain.
        return poses

    def load_calib_data(self, i, j):
        chain_path = self.calib_dir / f"cam_{i}_{j}" / "log1-camchain.yaml"
        if not chain_path.exists(): return None

        with open(chain_path) as f: data = yaml.safe_load(f)

        def extract_cam(d):
            intr = d['intrinsics']
            dist = d['distortion_coeffs']
            return {
                'xi': intr[0], 'fx': intr[1], 'fy': intr[2], 'cx': intr[3], 'cy': intr[4],
                'D': np.array(dist),
                'K': np.array([[intr[1], 0, intr[3]], [0, intr[2], intr[4]], [0,0,1]])
            }

        camL = extract_cam(data['cam0'])
        camR = extract_cam(data['cam1'])
        # Relative transform for rectification logic
        T_rel = np.array(data['cam1']['T_cn_cnm1'])

        return camL, camR, T_rel

    def generate_virtual_maps(self, camL, camR, T_rel, i, j, rotation_deg=45.0):
        focal = self.W / (2 * np.tan(np.radians(self.HFOV / 2)))
        K_virt = np.array([[focal, 0, self.W/2], [0, focal, self.H/2], [0, 0, 1]])

        angle_rad = np.radians(rotation_deg)
        R_virt_to_phys_L = _create_y_rotation_matrix(angle_rad)  # +45
        R_virt_to_phys_R = _create_y_rotation_matrix(-angle_rad) # -45

        map_L_x, map_L_y = self._make_map(camL, R_virt_to_phys_L, focal)
        map_R_x, map_R_y = self._make_map(camR, R_virt_to_phys_R, focal)

        # Analytic Extrinsics for Stereo Rectify
        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]

        R_virt_rel = R_virt_to_phys_R.T @ R_rel @ R_virt_to_phys_L
        t_virt_rel = R_virt_to_phys_R.T @ t_rel

        # --- CALCULATE GLOBAL POSE (T_rect_left_cam0) ---
        # 1. Get Pose of Physical Left Camera (i) in Cam0
        T_physL_cam0 = self.cam_poses_in_cam0.get(i, np.eye(4))

        # 2. Get Pose of Virtual Left in Physical Left
        # P_phys = R_v2p * P_virt  ->  T_virt_in_phys = [R_v2p | 0]
        T_virtL_physL = np.eye(4)
        T_virtL_physL[:3, :3] = R_virt_to_phys_L

        # 3. Chain: T_virtL_cam0 = T_physL_cam0 * T_virtL_physL
        T_virtL_cam0 = T_physL_cam0 @ T_virtL_physL

        return {
            'map_L': (map_L_x, map_L_y),
            'map_R': (map_R_x, map_R_y),
            'K_virt': K_virt,
            'R_virt_rel': R_virt_rel,
            't_virt_rel': t_virt_rel,
            'res': (self.W, self.H),
            'T_virtL_cam0': T_virtL_cam0  # Pass this forward
        }

    def _make_map(self, omni, R, focal):
        grid_y, grid_x = np.indices((self.H, self.W))
        x_v = grid_x - self.W/2
        y_v = grid_y - self.H/2
        z_v = np.full_like(x_v, focal)
        pts_virt = np.stack([x_v.flatten(), y_v.flatten(), z_v.flatten()], axis=1)
        pts_phys = pts_virt @ R.T
        pts2d, valid_mask = projectPoints_fisheye(pts_phys, omni['K'], omni['xi'], omni['D'])

        map_x = pts2d[:, 0].reshape(self.H, self.W).astype(np.float32)
        map_y = pts2d[:, 1].reshape(self.H, self.W).astype(np.float32)
        return map_x, map_y

# ==========================================
# PART 2: RECTIFICATION & MERGE
# ==========================================

def compute_rectification(virt_data):
    K = virt_data['K_virt']
    D = np.zeros(5)
    size = virt_data['res']
    R = virt_data['R_virt_rel']
    t = virt_data['t_virt_rel']

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, D, K, D, size, R, t,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    m1l, m2l = cv2.initUndistortRectifyMap(K, D, R1, P1, size, cv2.CV_32FC1)
    m1r, m2r = cv2.initUndistortRectifyMap(K, D, R2, P2, size, cv2.CV_32FC1)

    # Calculate Final Pose of Rectified Left Camera in Cam0
    # T_rect_in_virt = inv(R1) (Pure rotation)
    # T_rect_cam0 = T_virt_cam0 * T_rect_in_virt
    # R1 transforms Unrect->Rect. So Rect->Unrect is R1.T

    T_rect_in_virt = np.eye(4)
    T_rect_in_virt[:3, :3] = R1.T

    T_virtL_cam0 = virt_data['T_virtL_cam0']
    T_rectL_cam0 = T_virtL_cam0 @ T_rect_in_virt

    return m1l, m2l, m1r, m2r, P1, P2, Q, T_rectL_cam0

def combine_maps(map_fisheye_to_virt, map_rect_to_virt):
    map_virt_x, map_virt_y = map_fisheye_to_virt
    map_rect_x, map_rect_y = map_rect_to_virt
    final_x = cv2.remap(map_virt_x, map_rect_x, map_rect_y, cv2.INTER_LINEAR)
    final_y = cv2.remap(map_virt_y, map_rect_x, map_rect_y, cv2.INTER_LINEAR)
    return final_x, final_y

# ==========================================
# PART 3: VISUALIZATION
# ==========================================

def visualize_comparison(imgL, imgR, mapL, mapR, pair_name):
    if imgL is None or imgR is None: return

    rectL = cv2.remap(imgL, mapL[0], mapL[1], cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, mapR[0], mapR[1], cv2.INTER_LINEAR)

    after_view = np.hstack([rectL, rectR])
    h_after, w_after = after_view.shape[:2]
    for y in range(0, h_after, 25):
        cv2.line(after_view, (0, y), (w_after, y), (0, 255, 0), 1)

    before_view = np.hstack([imgL, imgR])
    h_before, w_before = before_view.shape[:2]

    if w_before != w_after:
        scale = w_after / w_before
        before_view = cv2.resize(before_view, (0,0), fx=scale, fy=scale)

    full_viz = np.vstack([before_view, after_view])

    # Force 2x Scale
    full_viz = cv2.resize(full_viz, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

    cv2.putText(full_viz, f"BEFORE (Raw) - {pair_name}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    cv2.putText(full_viz, "AFTER (Rectified)", (20, full_viz.shape[0]//2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    win_name = f"Compare {pair_name}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, full_viz.shape[1], full_viz.shape[0])
    cv2.imshow(win_name, full_viz)
    cv2.waitKey(0)
    cv2.destroyWindow(win_name)

def grab_ros_image(bag_dir, i, j):
    p = bag_dir / f"pair_{i}_{j}"
    if not p.exists(): return None, None
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(p), storage_id='sqlite3'), ConverterOptions('', ''))
    topics = {f"/cam_{i}/image_raw", f"/cam_{j}/image_raw"}
    imgs = {}
    while reader.has_next() and len(imgs) < 2:
        top, data, _ = reader.read_next()
        if top in topics and top not in imgs:
            msg = deserialize_message(data, Image)
            shape = (msg.height, msg.width, 3) if msg.encoding=='bgr8' else (msg.height, msg.width)
            arr = np.frombuffer(msg.data, np.uint8).reshape(shape)
            if len(arr.shape)==2: arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            imgs[top] = arr
    return imgs.get(f"/cam_{i}/image_raw"), imgs.get(f"/cam_{j}/image_raw")

# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-dir", default="calibration_output")
    parser.add_argument("--out-dir", default="final_maps")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--hfov", type=float, default=110.0)
    args = parser.parse_args()

    calib_path = Path(args.calib_dir)
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    gen = VirtualStereoGen(calib_path, args.width, args.height, args.hfov)
    pairs = [(0,1), (1,2), (2,3), (3,0)]

    print("\n[Init] Loaded global camera poses:")
    for cid, pose in gen.cam_poses_in_cam0.items():
        print(f"  Cam {cid}: {pose[:3,3]}")

    for (i, j) in pairs:
        print(f"\nProcessing Pair {i}-{j}...")
        data = gen.load_calib_data(i, j)
        if not data:
            print(f"Skipping {i}-{j}")
            continue
        camL, camR, T_rel = data

        virt_data = gen.generate_virtual_maps(camL, camR, T_rel, i, j, rotation_deg=45.0)
        m1l, m2l, m1r, m2r, P1, P2, Q, T_rectL_cam0 = compute_rectification(virt_data)

        final_L = combine_maps(virt_data['map_L'], (m1l, m2l))
        final_R = combine_maps(virt_data['map_R'], (m1r, m2r))

        pair_str = f"{i}_{j}"
        baseline = abs(P2[0,3] / P2[0,0])

        # Save Maps (Legacy format)
        fs = cv2.FileStorage(str(out_path / f"final_rectified_to_fisheye_map_{pair_str}.yml"), cv2.FILE_STORAGE_WRITE)
        fs.write("map_left_x", final_L[0]); fs.write("map_left_y", final_L[1])
        fs.write("map_right_x", final_R[0]); fs.write("map_right_y", final_R[1])
        fs.release()

        # Save Config with CORRECT POSE
        with open(out_path / f"final_map_config_{pair_str}.yaml", 'w') as f:
            yaml.dump({
                'pair': pair_str,
                'final_resolution': [args.width, args.height],
                'baseline_meters': float(baseline),
                'K_rect_left': P1[:3,:3].tolist(),
                'T_rect_left_cam0': T_rectL_cam0.tolist()
            }, f)

        # Visualize
        imgL, imgR = grab_ros_image(calib_path / "ros2_bags", i, j)
        visualize_comparison(imgL, imgR, final_L, final_R, pair_str)

    print("\nProcessing Complete.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
