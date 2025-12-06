#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shlex
import yaml
import shutil
import subprocess
import concurrent.futures
import queue
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Set

import numpy as np
import cv2

# ROS 2 imports
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CompressedImage
from builtin_interfaces.msg import Time

# ROS 2 / messages for bag writing
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image

try:
    from cv_bridge import CvBridge
    _BRIDGE = CvBridge()
except Exception:
    _BRIDGE = None


# ---------- CONFIG ----------
DOCKER_IMAGE_TAG = "mortyl0834/omnitartancalib:quad_cam"
MODEL = "omni-radtan"
PAIR_LIST = [(0, 1), (1, 2), (2, 3), (3, 0)]
# Default topic names
DEFAULT_COMPRESSED_TOPIC = "/oak_ffc_4p_driver_node/compressed"
DEFAULT_RAW_TOPIC = "/oak_ffc_4p_driver_node/image_raw"
# ----------------------------


@dataclass
class WriterBundle:
    writer: SequentialWriter
    topics: Dict[str, TopicMetadata]


@dataclass
class RecordingTarget:
    """Parsed recording target from --only argument"""
    mono_cams: Set[int] = field(default_factory=set)
    pairs: Set[Tuple[int, int]] = field(default_factory=set)

    @classmethod
    def parse(cls, only_str: Optional[str]) -> 'RecordingTarget':
        target = cls()
        if not only_str:
            target.mono_cams = {0, 1, 2, 3}
            target.pairs = set(PAIR_LIST)
            return target

        parts = only_str.split(',')
        for part in parts:
            part = part.strip().lower()
            if part.startswith('mono'):
                try:
                    cam_id = int(part[4:])
                    if 0 <= cam_id <= 3:
                        target.mono_cams.add(cam_id)
                except ValueError: pass
            elif part.startswith('pair'):
                try:
                    pair_str = part[4:]
                    if len(pair_str) == 2:
                        i, j = int(pair_str[0]), int(pair_str[1])
                        sorted_pair = tuple(sorted((i, j)))
                        for p_orig in PAIR_LIST:
                            if tuple(sorted(p_orig)) == sorted_pair:
                                target.pairs.add(p_orig)
                                break
                except ValueError: pass
        return target

    def is_empty(self) -> bool:
        return len(self.mono_cams) == 0 and len(self.pairs) == 0

    def get_recording_stages(self) -> List[Tuple[str, any]]:
        stages = []
        if self.mono_cams == {0, 1, 2, 3} and self.pairs == set(PAIR_LIST):
            stages.append(('MONO', 0)); stages.append(('PAIR', (0, 1)))
            stages.append(('MONO', 1)); stages.append(('PAIR', (1, 2)))
            stages.append(('MONO', 2)); stages.append(('PAIR', (2, 3)))
            stages.append(('MONO', 3)); stages.append(('PAIR', (3, 0)))
        else:
            for cam_id in sorted(list(self.mono_cams)): stages.append(('MONO', cam_id))
            for pair in PAIR_LIST:
                if pair in self.pairs: stages.append(('PAIR', pair))
        return stages


@dataclass
class CalibrationTarget:
    mono_cams: Set[int] = field(default_factory=set)
    pairs: Set[Tuple[int, int]] = field(default_factory=set)
    required_mono_cams: Set[int] = field(default_factory=set)

    @classmethod
    def from_recording_target(cls, rec_target: RecordingTarget) -> 'CalibrationTarget':
        if rec_target.is_empty():
            cal_target = cls(mono_cams={0, 1, 2, 3}, pairs=set(PAIR_LIST))
            cal_target.required_mono_cams = {0, 1, 2, 3}
            return cal_target

        cal_target = cls()
        cal_target.mono_cams.update(rec_target.mono_cams)
        cal_target.pairs.update(rec_target.pairs)
        for cam_id in rec_target.mono_cams:
            for pair in PAIR_LIST:
                if cam_id in pair: cal_target.pairs.add(pair)
        for i, j in cal_target.pairs:
            cal_target.required_mono_cams.add(i); cal_target.required_mono_cams.add(j)
        return cal_target


@dataclass
class CalibPaths:
    out_dir: Path
    ros2_dir: Path
    ros1_dir: Path
    mono_bags_ros2: Dict[int, Path] = field(default_factory=dict)
    mono_bags_ros1: Dict[int, Path] = field(default_factory=dict)
    pair_bags_ros2: Dict[Tuple[int, int], Path] = field(default_factory=dict)
    pair_bags_ros1: Dict[Tuple[int, int], Path] = field(default_factory=dict)


@dataclass
class FrameData:
    frames_cv: List[np.ndarray]
    stamp: Time
    timestamp_ns: int


class BufferedRosBagWriter:
    def __init__(self, buffer_size: int = 200):
        self.buffer_size = buffer_size
        self.write_queue = queue.Queue(maxsize=buffer_size)
        self.writer_thread = None
        self.stop_event = threading.Event()
        self.dropped_frames = 0
        self.written_frames = 0

    def start(self):
        self.stop_event.clear()
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.writer_thread: self.writer_thread.join(timeout=10)

    def queue_frame(self, frame_data: FrameData, writers: dict, recording_type: str, recording_data):
        try:
            self.write_queue.put_nowait((frame_data, writers, recording_type, recording_data))
        except queue.Full:
            self.dropped_frames += 1

    def _writer_loop(self):
        while not self.stop_event.is_set() or not self.write_queue.empty():
            try:
                item = self.write_queue.get(timeout=0.1)
                if item is None: continue
                frame_data, writers, recording_type, recording_data = item

                if recording_type == 'MONO':
                    ci = recording_data
                    img_msg = _ros_image_from_numpy(frame_data.frames_cv[ci], frame_data.stamp)
                    writers[ci].writer.write(f"/cam_{ci}/image_raw", _serialize(img_msg), frame_data.timestamp_ns)
                elif recording_type == 'PAIR':
                    i, j = recording_data
                    img_i = _ros_image_from_numpy(frame_data.frames_cv[i], frame_data.stamp)
                    img_j = _ros_image_from_numpy(frame_data.frames_cv[j], frame_data.stamp)
                    writers[(i, j)].writer.write(f"/cam_{i}/image_raw", _serialize(img_i), frame_data.timestamp_ns)
                    writers[(i, j)].writer.write(f"/cam_{j}/image_raw", _serialize(img_j), frame_data.timestamp_ns)
                self.written_frames += 1
            except queue.Empty: continue
            except Exception as e: print(f"\n[ERROR] Writer exception: {e}")


class StackedImageSubscriber(Node):
    def __init__(self, topic_name: str, is_compressed: bool):
        super().__init__('stacked_image_subscriber')
        self.frame_queue = queue.Queue(maxsize=2)
        if is_compressed:
            self.create_subscription(CompressedImage, topic_name, self.compressed_image_callback, 10)
        else:
            self.create_subscription(RosImage, topic_name, self.raw_image_callback, 10)

    def compressed_image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_image is not None: self._process_and_queue_frame(cv_image, msg.header.stamp)

    def raw_image_callback(self, msg: RosImage):
        if _BRIDGE:
            cv_image = _BRIDGE.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        else:
            cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if cv_image is not None: self._process_and_queue_frame(cv_image, msg.header.stamp)

    def _process_and_queue_frame(self, cv_image: np.ndarray, stamp: Time):
        height, width = cv_image.shape[:2]
        cam_width = width // 4
        frames = [cv_image[:, (i * cam_width):((i + 1) * cam_width)].copy() for i in range(4)]
        t_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        if self.frame_queue.full():
            try: self.frame_queue.get_nowait()
            except queue.Empty: pass
        self.frame_queue.put_nowait((frames, stamp, t_ns))

    def get_new_frame(self):
        try: return self.frame_queue.get_nowait()
        except queue.Empty: return None


def _ros_image_from_numpy(img_bgr: np.ndarray, stamp) -> Image:
    if _BRIDGE:
        msg = _BRIDGE.cv2_to_imgmsg(img_bgr, encoding='bgr8')
        msg.header.stamp = stamp
        return msg
    msg = Image()
    msg.header.stamp = stamp
    msg.height, msg.width = img_bgr.shape[:2]
    msg.encoding = "bgr8"
    msg.step = int(msg.width * 3)
    msg.data = img_bgr.tobytes()
    return msg


def _start_ros2_writer(uri: Path, topics: List[Tuple[str, str]]) -> WriterBundle:
    writer = SequentialWriter()
    writer.open(StorageOptions(uri=str(uri), storage_id="sqlite3"), ConverterOptions("", ""))
    tm = {}
    for name, tstr in topics:
        meta = TopicMetadata(name=name, type=tstr, serialization_format="cdr")
        writer.create_topic(meta)
        tm[name] = meta
    return WriterBundle(writer=writer, topics=tm)


def _serialize(msg) -> bytes:
    return serialize_message(msg)


def _create_viz_grid(crops: List[np.ndarray], n_kept_mono: List[int], n_kept_pair: Dict,
                     status_text: str, recording_cams: List[int], buffer_usage: str = "") -> np.ndarray:
    viz_crops = []
    for i, crop in enumerate(crops):
        viz = crop.copy()
        h, w = viz.shape[:2]
        # Draw label background
        cv2.rectangle(viz, (0, 0), (w, 30), (0, 0, 0), -1)

        # Draw green border if this camera is currently recording
        if i in recording_cams:
            thickness = 10
            # Start from y=30 (below label)
            cv2.rectangle(viz, (thickness//2, 30 + thickness//2),
                          (w - thickness//2, h - thickness//2),
                          (0, 255, 0), thickness)

        # Draw text
        cv2.putText(viz, f"Cam {i}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(viz, f"Saved: {n_kept_mono[i]}", (w - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        viz_crops.append(viz)

    row1 = np.hstack([viz_crops[0], viz_crops[1]])
    row2 = np.hstack([viz_crops[2], viz_crops[3]])
    grid = np.vstack([row1, row2])
    h, w = grid.shape[:2]
    info_bar = np.zeros((60, w, 3), dtype=np.uint8)

    cv2.putText(info_bar, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    pair_stats_text = "Pair frames: " + " | ".join([f"P{i}{j}:{n_kept_pair.get((i, j), 0)}" for i, j in PAIR_LIST])
    cv2.putText(info_bar, pair_stats_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if buffer_usage:
        (text_w, _), _ = cv2.getTextSize(buffer_usage, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(info_bar, buffer_usage, (w - text_w - 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return np.vstack([info_bar, grid])


def _docker_run(cmd: str, mounts: Dict[str, str], gui: bool = False, image: str = DOCKER_IMAGE_TAG):
    try:
        subprocess.run(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        subprocess.check_call(["docker", "pull", image])
    base = ["docker", "run", "--rm"]
    if gui: base += ["-e", "DISPLAY", "-e", "QT_X11_NO_MITSHM=1", "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw"]
    for host, cont in mounts.items(): base += ["-v", f"{host}:{cont}"]
    full = base + [image, "/bin/bash", "-lc", f"source /catkin_ws/devel/setup.bash && {cmd}"]
    print(" ".join(shlex.quote(c) for c in full))
    subprocess.run(full, check=True)


def _run_single_intrinsic_calib(bag_path: Path, target_yaml: Path, cam_out_dir: Path, cam_id: int) -> Optional[dict]:
    if not bag_path or not bag_path.exists(): return None
    try:
        cam_out_dir.mkdir(exist_ok=True)
        cmd = (f"export KALIBR_MANUAL_FOCAL_LENGTH_INIT=1 && "
               f"yes 255 | rosrun kalibr tartan_calibrate --bag /data/input.bag --target /data/target.yaml "
               f"--topics /cam_{cam_id}/image_raw --models {MODEL} --save_dir /data/out --dont-show-report")
        mounts = {str(bag_path.absolute()): "/data/input.bag:ro", str(target_yaml.absolute()): "/data/target.yaml:ro", str(cam_out_dir.absolute()): "/data/out"}
        _docker_run(cmd, mounts, gui=False)
        with open(cam_out_dir / "log1-camchain.yaml", "r") as f: return yaml.safe_load(f)["cam0"]
    except Exception as e:
        print(f"  [ERROR] Mono {cam_id}: {e}")
        return None


def _run_single_extrinsic_calib(bag_path: Path, target_yaml: Path, pair_out_dir: Path, cam_i_intr: dict, cam_j_intr: dict, pair: Tuple[int, int]):
    i, j = pair
    try:
        if pair_out_dir.exists(): shutil.rmtree(pair_out_dir)
        pair_out_dir.mkdir(parents=True)
        intr_path = pair_out_dir / "intrinsic.yaml"
        with open(intr_path, "w") as f: yaml.safe_dump({"cam0": cam_i_intr, "cam1": cam_j_intr}, f)
        cmd = (f"export KALIBR_MANUAL_FOCAL_LENGTH_INIT=1 && "
               f"rosrun kalibr tartan_calibrate --bag /data/input.bag --target /data/target.yaml "
               f"--topics /cam_{i}/image_raw /cam_{j}/image_raw --models {MODEL} {MODEL} "
               f"--save_dir /data/out --dont-show-report --intrinsic-prarameters /data/intrinsics.yaml")
        mounts = {str(bag_path.absolute()): "/data/input.bag:ro", str(target_yaml.absolute()): "/data/target.yaml:ro", str(intr_path.absolute()): "/data/intrinsics.yaml:ro", str(pair_out_dir.absolute()): "/data/out"}
        _docker_run(cmd, mounts, gui=False)
    except Exception as e:
        print(f"  [ERROR] Pair {i}-{j}: {e}")


class MultiCamCalibrator:
    def __init__(self, out_dir: Path, target_yaml: Path):
        self.out = CalibPaths(out_dir=out_dir.resolve(), ros2_dir=out_dir.resolve()/"ros2_bags", ros1_dir=out_dir.resolve()/"ros1_bags")
        self.target_yaml = target_yaml.resolve()
        self.intrinsics: Dict[int, dict] = {}

    def discover_ros2_bags(self) -> bool:
        if not self.out.ros2_dir.exists(): return False
        for ci in range(4):
            src = self.out.ros2_dir / f"cam_{ci}_mono"
            if src.exists(): self.out.mono_bags_ros2[ci] = src
        for pair in PAIR_LIST:
            src = self.out.ros2_dir / f"pair_{pair[0]}_{pair[1]}"
            if src.exists(): self.out.pair_bags_ros2[pair] = src
        return bool(self.out.mono_bags_ros2 or self.out.pair_bags_ros2)

    def _preserve_existing_bags(self, recording_target: RecordingTarget):
        if not self.out.ros2_dir.exists(): return
        for ci in range(4):
            if ci not in recording_target.mono_cams:
                src = self.out.ros2_dir / f"cam_{ci}_mono"
                if src.exists(): self.out.mono_bags_ros2[ci] = src
        for pair in PAIR_LIST:
            if pair not in recording_target.pairs:
                src = self.out.ros2_dir / f"pair_{pair[0]}_{pair[1]}"
                if src.exists(): self.out.pair_bags_ros2[pair] = src

    def load_existing_intrinsics(self, calib_target: CalibrationTarget):
        for cam_id in (calib_target.required_mono_cams - calib_target.mono_cams):
            p = self.out.out_dir / f"cam_{cam_id}" / "log1-camchain.yaml"
            if p.exists():
                with open(p, "r") as f: self.intrinsics[cam_id] = yaml.safe_load(f)["cam0"]

    def collect_from_ros2_topic(self, recording_target: RecordingTarget, topic_name: str, is_compressed: bool):
        if not recording_target.is_empty(): self._preserve_existing_bags(recording_target)
        if self.out.ros2_dir.exists():
            for child in self.out.ros2_dir.iterdir():
                # Only delete directories that are IN the recording target
                is_target = False
                if child.name.startswith("cam_") and "mono" in child.name:
                    try:
                        if int(child.name.split('_')[1]) in recording_target.mono_cams: is_target=True
                    except: pass
                elif child.name.startswith("pair_"):
                    try:
                        p = child.name.split('_')
                        if (int(p[1]), int(p[2])) in recording_target.pairs: is_target=True
                    except: pass

                if recording_target.is_empty() or is_target:
                    if child.is_dir(): shutil.rmtree(child)

        self.out.ros2_dir.mkdir(parents=True, exist_ok=True)

        stages = recording_target.get_recording_stages()
        if not stages: return

        if not rclpy.ok(): rclpy.init()
        subscriber = StackedImageSubscriber(topic_name, is_compressed)
        executor = SingleThreadedExecutor()
        executor.add_node(subscriber)
        threading.Thread(target=executor.spin, daemon=True).start()

        mono_writers = {ci: _start_ros2_writer(self.out.ros2_dir/f"cam_{ci}_mono", [(f"/cam_{ci}/image_raw", "sensor_msgs/msg/Image")])
                        for ci in recording_target.mono_cams}
        pair_writers = {p: _start_ros2_writer(self.out.ros2_dir/f"pair_{p[0]}_{p[1]}",
                        [(f"/cam_{p[0]}/image_raw", "sensor_msgs/msg/Image"), (f"/cam_{p[1]}/image_raw", "sensor_msgs/msg/Image")])
                        for p in recording_target.pairs}

        # Populate path dicts immediately
        for ci in mono_writers: self.out.mono_bags_ros2[ci] = self.out.ros2_dir/f"cam_{ci}_mono"
        for p in pair_writers: self.out.pair_bags_ros2[p] = self.out.ros2_dir/f"pair_{p[0]}_{p[1]}"

        buf_writer = BufferedRosBagWriter()
        buf_writer.start()

        # REMOVE TOOLBAR: Use GUI_NORMAL
        cv2.namedWindow("Live Calibration Recording", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("Live Calibration Recording", 1600, 900)

        n_kept_mono = [0]*4
        n_kept_pair = {p:0 for p in PAIR_LIST}

        # START IDLE at -1
        idx = -1
        done = False

        print("[Visualization] Press 'SPACE' to advance recording stage, 'q' to quit.")

        try:
            while not done:
                frame_data = subscriber.get_new_frame()
                recording_cams = []

                # Logic for current stage
                if idx == -1:
                    status = "IDLE: Press SPACE to start recording."
                elif 0 <= idx < len(stages):
                    stype, sdata = stages[idx]
                    status = f"RECORDING {stype} {sdata}. SPACE to next."
                    if stype == 'MONO': recording_cams = [sdata]
                    else: recording_cams = list(sdata)

                    # Only write to buffer if we are in a valid recording stage (not Idle)
                    if frame_data:
                        obj = FrameData(*frame_data)
                        if stype == 'MONO':
                            buf_writer.queue_frame(obj, mono_writers, 'MONO', sdata)
                            n_kept_mono[sdata] += 1
                        else:
                            buf_writer.queue_frame(obj, pair_writers, 'PAIR', sdata)
                            n_kept_pair[sdata] += 1

                elif idx >= len(stages):
                    status = "DONE. Press 'q' to exit."

                if frame_data:
                    # Update buffer usage text
                    buffer_usage = f"Buffer: {buf_writer.write_queue.qsize()}/{buf_writer.buffer_size}"
                    if buf_writer.dropped_frames > 0:
                        buffer_usage += f" | Dropped: {buf_writer.dropped_frames}"

                    viz = _create_viz_grid(frame_data[0], n_kept_mono, n_kept_pair, status, recording_cams, buffer_usage)
                    cv2.imshow("Live Calibration Recording", viz)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'): done = True
                elif key == ord(' '): idx += 1

        finally:
            print("  [Info] Waiting for buffer to drain...")
            # 1. STOP the writer thread first.
            # This waits (joins) until the queue is empty and the thread finishes.
            buf_writer.stop()

            # 2. NOW it is safe to close the files to save metadata.
            # The background thread is dead, so no more writes will happen.
            all_writers = list(mono_writers.values()) + list(pair_writers.values())
            for bundle in all_writers:
                if hasattr(bundle.writer, 'close'):
                    bundle.writer.close()
                else:
                    # Fallback for older ROS 2 bindings
                    del bundle.writer

            # 3. Final Cleanup
            mono_writers.clear()
            pair_writers.clear()

            cv2.destroyAllWindows()
            executor.shutdown()
            subscriber.destroy_node()
            print("  [Info] Recording cleanup complete.")

    def convert_all_to_ros1(self, recording_target: RecordingTarget, force_reconvert: bool = True):
        print("\n[Step 2] Converting ROS2 -> ROS1...")
        self.out.ros1_dir.mkdir(parents=True, exist_ok=True)

        def convert(src, dst):
            if dst.exists(): dst.unlink()
            print(f"  Converting {src.name}...")
            subprocess.run(["rosbags-convert", "--src", str(src), "--dst", str(dst),
                            "--src-typestore", "ros2_humble", "--dst-typestore", "ros1_noetic"], check=True)

        # MONO
        for ci, src in self.out.mono_bags_ros2.items():
            dst = self.out.ros1_dir / f"{src.name}.bag"
            self.out.mono_bags_ros1[ci] = dst

            # [cite_start]Optimization: If targets exist and this isn't one, skip conversion [cite: 1]
            if not recording_target.is_empty() and ci not in recording_target.mono_cams:
                continue

            if (force_reconvert and ci in recording_target.mono_cams) or not dst.exists():
                convert(src, dst)

        # PAIR
        for pr, src in self.out.pair_bags_ros2.items():
            dst = self.out.ros1_dir / f"{src.name}.bag"
            self.out.pair_bags_ros1[pr] = dst

            # [cite_start]Optimization: If targets exist and this isn't one, skip conversion [cite: 1]
            if not recording_target.is_empty() and pr not in recording_target.pairs:
                continue

            if (force_reconvert and pr in recording_target.pairs) or not dst.exists():
                convert(src, dst)

    def run_calibration(self, target: CalibrationTarget, workers: int):
        # Intrinsics
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for ci in target.mono_cams:
                bag = self.out.mono_bags_ros1.get(ci)
                if bag and bag.exists():
                    futs[ex.submit(_run_single_intrinsic_calib, bag, self.target_yaml, self.out.out_dir/f"cam_{ci}", ci)] = ci
            for f in concurrent.futures.as_completed(futs):
                res = f.result()
                if res: self.intrinsics[futs[f]] = res

        # Extrinsics
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = []
            for i, j in target.pairs:
                if i in self.intrinsics and j in self.intrinsics:
                    bag = self.out.pair_bags_ros1.get((i, j))
                    if bag and bag.exists():
                        futs.append(ex.submit(_run_single_extrinsic_calib, bag, self.target_yaml,
                                            self.out.out_dir/f"cam_{i}_{j}", self.intrinsics[i], self.intrinsics[j], (i, j)))
            for f in concurrent.futures.as_completed(futs): f.result()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="calibration_output")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--skip-recording", action="store_true")
    ap.add_argument("--uncompressed", action="store_true")
    ap.add_argument("--topic", type=str, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    rec_target = RecordingTarget.parse(args.only)
    cal_target = CalibrationTarget.from_recording_target(rec_target)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target_yaml = Path("april_6x6.yaml")
    if not target_yaml.exists():
        with open(target_yaml, 'w') as f: f.write("target_type: 'aprilgrid'\ntagCols: 6\ntagRows: 6\ntagSize: 0.088\ntagSpacing: 0.3\n")

    calib = MultiCamCalibrator(out_dir, target_yaml)
    topic = args.topic if args.topic else (DEFAULT_RAW_TOPIC if args.uncompressed else DEFAULT_COMPRESSED_TOPIC)

    did_record = False
    if not args.skip_recording:
        calib.collect_from_ros2_topic(rec_target, topic, not args.uncompressed)
        did_record = True

    calib.discover_ros2_bags()
    calib.convert_all_to_ros1(rec_target, force_reconvert=did_record)

    calib.load_existing_intrinsics(cal_target)
    calib.run_calibration(cal_target, args.workers)

if __name__ == "__main__":
    main()
