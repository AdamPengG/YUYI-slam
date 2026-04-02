from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class OVOAsyncWorkerLegacyYOLO(Node):
    def __init__(self) -> None:
        super().__init__("ovo_async_worker_legacy_yolo")
        self._declare_parameters()
        self._load_parameters()

        self.current_process: Optional[subprocess.Popen] = None
        self.current_export_dir: Path | None = None
        self._stopping = False
        self._waiting_log_emitted = False
        self._debug_image_last_mtime_ns = -1
        self._debug_image_cached_msg: Optional[Image] = None
        self._bridge = CvBridge()
        self.debug_image_pub = self.create_publisher(Image, self.debug_image_topic, 10)

        self.timer = self.create_timer(self.poll_period_sec, self._poll)
        self.debug_timer = self.create_timer(self.debug_publish_period_sec, self._publish_debug_image)
        self.get_logger().info(
            "Legacy YOLO semantic observer worker ready. "
            f"scene={self.scene_name}, experiment={self.experiment_name}, poll={self.poll_period_sec}s"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "scene_name": "isaac_turtlebot3_livo2_online",
            "dataset_name": "Replica",
            "dataset_root": "/home/peng/isacc_slam/reference/OVO/data/input/Datasets/Replica",
            "run_root": "/home/peng/isacc_slam/runs/ovo_pose_keyframes",
            "export_dir_override": "",
            "ovo_root": "/home/peng/isacc_slam/reference/OVO",
            "ovo_config": "data/working/configs/ovo_livo2_yoloe26x.yaml",
            "experiment_name": "isaac_livo2_online_legacy_yolo",
            "artifact_path": "/home/peng/isacc_slam/reference/OVO/data/output/Replica/isaac_livo2_online_legacy_yolo/isaac_turtlebot3_livo2_online/semantic_snapshot.npz",
            "observer_script": "/home/peng/isacc_slam/src/onemap_semantic_mapper/scripts/run_semantic_observer_online_legacy_yolo.py",
            "poll_period_sec": 5.0,
            "min_keyframes": 1,
            "rerun_every_new_keyframes": 1,
            "clear_output_on_start": False,
            "resume_if_exists": True,
            "device": "cuda",
            "topk_labels": 3,
            "visibility_depth_tolerance_m": 0.08,
            "snapshot_voxel_size_m": 0.0,
            "snapshot_max_points": 1000000,
            "snapshot_mode": "registered_frame",
            "min_observation_points": 4,
            "min_mask_area": 24,
            "merge_centroid_radius_m": 0.75,
            "support_expansion_radius_m": 0.20,
            "class_set": "full",
            "near_ground_filter_height_m": 0.08,
            "near_ground_floor_percentile": 1.0,
            "assoc_score_min": 0.42,
            "reproj_iou_min": 0.08,
            "surface_hit_min": 0.10,
            "reproj_dilate_px": 3,
            "track_pending_hits": 1,
            "track_dormant_after_sec": 2.0,
            "track_delete_after_sec": 30.0,
            "new_track_min_points": 6,
            "fuse_voxel_size_m": 0.03,
            "support_expansion_max_points": 12000,
            "use_semantic_subset_projection": False,
            "debug_image_topic": "/semantic_debug_image",
            "debug_publish_period_sec": 0.25,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_parameters(self) -> None:
        self.scene_name = str(self.get_parameter("scene_name").value)
        self.dataset_name = str(self.get_parameter("dataset_name").value)
        self.dataset_root = Path(str(self.get_parameter("dataset_root").value)).expanduser()
        self.run_root = Path(str(self.get_parameter("run_root").value)).expanduser()
        export_dir_override = str(self.get_parameter("export_dir_override").value).strip()
        self.export_dir_override = Path(export_dir_override).expanduser() if export_dir_override else None
        self.ovo_root = Path(str(self.get_parameter("ovo_root").value)).expanduser()
        self.ovo_config = str(self.get_parameter("ovo_config").value)
        self.experiment_name = str(self.get_parameter("experiment_name").value)
        self.artifact_path = Path(str(self.get_parameter("artifact_path").value)).expanduser()
        self.observer_script = Path(str(self.get_parameter("observer_script").value)).expanduser()
        self.poll_period_sec = float(self.get_parameter("poll_period_sec").value)
        self.min_keyframes = int(self.get_parameter("min_keyframes").value)
        self.process_every_new_keyframes = int(self.get_parameter("rerun_every_new_keyframes").value)
        self.clear_output_on_start = bool(self.get_parameter("clear_output_on_start").value)
        self.resume_if_exists = bool(self.get_parameter("resume_if_exists").value)
        self.device = str(self.get_parameter("device").value)
        self.topk_labels = int(self.get_parameter("topk_labels").value)
        self.visibility_depth_tolerance_m = float(self.get_parameter("visibility_depth_tolerance_m").value)
        self.snapshot_voxel_size_m = float(self.get_parameter("snapshot_voxel_size_m").value)
        self.snapshot_max_points = int(self.get_parameter("snapshot_max_points").value)
        self.snapshot_mode = str(self.get_parameter("snapshot_mode").value)
        self.min_observation_points = int(self.get_parameter("min_observation_points").value)
        self.min_mask_area = int(self.get_parameter("min_mask_area").value)
        self.merge_centroid_radius_m = float(self.get_parameter("merge_centroid_radius_m").value)
        self.support_expansion_radius_m = float(self.get_parameter("support_expansion_radius_m").value)
        self.class_set = str(self.get_parameter("class_set").value)
        self.near_ground_filter_height_m = float(self.get_parameter("near_ground_filter_height_m").value)
        self.near_ground_floor_percentile = float(self.get_parameter("near_ground_floor_percentile").value)
        self.assoc_score_min = float(self.get_parameter("assoc_score_min").value)
        self.reproj_iou_min = float(self.get_parameter("reproj_iou_min").value)
        self.surface_hit_min = float(self.get_parameter("surface_hit_min").value)
        self.reproj_dilate_px = int(self.get_parameter("reproj_dilate_px").value)
        self.track_pending_hits = int(self.get_parameter("track_pending_hits").value)
        self.track_dormant_after_sec = float(self.get_parameter("track_dormant_after_sec").value)
        self.track_delete_after_sec = float(self.get_parameter("track_delete_after_sec").value)
        self.new_track_min_points = int(self.get_parameter("new_track_min_points").value)
        self.fuse_voxel_size_m = float(self.get_parameter("fuse_voxel_size_m").value)
        self.support_expansion_max_points = int(self.get_parameter("support_expansion_max_points").value)
        self.use_semantic_subset_projection = bool(self.get_parameter("use_semantic_subset_projection").value)
        self.debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        self.debug_publish_period_sec = float(self.get_parameter("debug_publish_period_sec").value)
        self.status_path = self.artifact_path.parent / "online_status.json"
        self.debug_image_path = self.artifact_path.parent / "semantic_debug_latest.png"

    def _scene_dir(self) -> Path:
        return self.dataset_root / self.scene_name

    def _resolve_export_dir(self) -> Path | None:
        if self.export_dir_override is not None:
            export_dir = self.export_dir_override
            if not export_dir.exists():
                return None
            if not (export_dir / "keyframe_packets.jsonl").exists():
                return None
            if not (export_dir / "local_cloud_packets.jsonl").exists():
                return None
            return export_dir
        if not self.run_root.exists():
            return None
        candidates = []
        for path in self.run_root.iterdir():
            if not path.is_dir():
                continue
            if self.scene_name not in path.name:
                continue
            candidates.append(path)
        if not candidates:
            return None
        latest = max(candidates, key=lambda item: item.stat().st_mtime_ns)
        export_dir = latest / "export"
        if not (export_dir / "keyframe_packets.jsonl").exists():
            return None
        if not (export_dir / "local_cloud_packets.jsonl").exists():
            return None
        return export_dir

    def _build_command(self, export_dir: Path) -> str:
        scene_dir = self._scene_dir()
        output_dir = self.artifact_path.parent
        cmd = [
            "cd",
            shlex.quote(str(self.ovo_root)),
            "&&",
            "source /home/peng/miniconda3/etc/profile.d/conda.sh",
            "&&",
            "conda activate ovo5090",
            "&&",
            "python",
            shlex.quote(str(self.observer_script)),
            "--scene-dir",
            shlex.quote(str(scene_dir)),
            "--export-dir",
            shlex.quote(str(export_dir)),
            "--output-dir",
            shlex.quote(str(output_dir)),
            "--ovo-root",
            shlex.quote(str(self.ovo_root)),
            "--ovo-config",
            shlex.quote(self.ovo_config),
            "--dataset-name",
            shlex.quote(self.dataset_name),
            "--scene-name",
            shlex.quote(self.scene_name),
            "--poll-period-sec",
            shlex.quote(str(self.poll_period_sec)),
            "--min-keyframes",
            shlex.quote(str(self.min_keyframes)),
            "--process-every-new-keyframes",
            shlex.quote(str(self.process_every_new_keyframes)),
            "--device",
            shlex.quote(self.device),
            "--topk-labels",
            shlex.quote(str(self.topk_labels)),
            "--visibility-depth-tolerance-m",
            shlex.quote(str(self.visibility_depth_tolerance_m)),
            "--snapshot-voxel-size-m",
            shlex.quote(str(self.snapshot_voxel_size_m)),
            "--snapshot-max-points",
            shlex.quote(str(self.snapshot_max_points)),
            "--snapshot-mode",
            shlex.quote(str(self.snapshot_mode)),
            "--min-observation-points",
            shlex.quote(str(self.min_observation_points)),
            "--min-mask-area",
            shlex.quote(str(self.min_mask_area)),
            "--merge-centroid-radius-m",
            shlex.quote(str(self.merge_centroid_radius_m)),
            "--support-expansion-radius-m",
            shlex.quote(str(self.support_expansion_radius_m)),
            "--class-set",
            shlex.quote(self.class_set),
            "--near-ground-filter-height-m",
            shlex.quote(str(self.near_ground_filter_height_m)),
            "--near-ground-floor-percentile",
            shlex.quote(str(self.near_ground_floor_percentile)),
            "--assoc-score-min",
            shlex.quote(str(self.assoc_score_min)),
            "--reproj-iou-min",
            shlex.quote(str(self.reproj_iou_min)),
            "--surface-hit-min",
            shlex.quote(str(self.surface_hit_min)),
            "--reproj-dilate-px",
            shlex.quote(str(self.reproj_dilate_px)),
            "--track-pending-hits",
            shlex.quote(str(self.track_pending_hits)),
            "--track-dormant-after-sec",
            shlex.quote(str(self.track_dormant_after_sec)),
            "--track-delete-after-sec",
            shlex.quote(str(self.track_delete_after_sec)),
            "--new-track-min-points",
            shlex.quote(str(self.new_track_min_points)),
            "--fuse-voxel-size-m",
            shlex.quote(str(self.fuse_voxel_size_m)),
            "--support-expansion-max-points",
            shlex.quote(str(self.support_expansion_max_points)),
        ]
        if self.use_semantic_subset_projection:
            cmd.append("--use-semantic-subset-projection")
        if self.clear_output_on_start:
            cmd.append("--clear-output")
        if self.resume_if_exists:
            cmd.append("--resume-if-exists")
        return " ".join(cmd)

    def _start_process(self, export_dir: Path) -> None:
        command = self._build_command(export_dir)
        self.current_export_dir = export_dir
        self.current_process = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            cwd=str(self.ovo_root),
            stdout=None,
            stderr=None,
            env=os.environ.copy(),
        )
        self.get_logger().info(
            "Started legacy YOLO semantic observer subprocess for "
            f"scene={self.scene_name}, export_dir={export_dir}."
        )

    def _poll(self) -> None:
        export_dir = self._resolve_export_dir()
        if self.current_process is None:
            if export_dir is None or not self._scene_dir().exists():
                if not self._waiting_log_emitted:
                    self.get_logger().info(
                        f"Waiting for scene/export manifests. scene_dir={self._scene_dir()}, run_root={self.run_root}"
                    )
                    self._waiting_log_emitted = True
                return
            self._waiting_log_emitted = False
            self._start_process(export_dir)
            return

        ret = self.current_process.poll()
        if ret is None:
            return

        if self._stopping:
            self.get_logger().info("Legacy YOLO semantic observer exited during shutdown.")
            self.current_process = None
            return

        if self.status_path.exists():
            try:
                status_text = self.status_path.read_text(encoding="utf-8")
                self.get_logger().warning(
                    f"Legacy YOLO semantic observer exited with code {ret}. Last status: {status_text}"
                )
            except OSError:
                self.get_logger().warning(f"Legacy YOLO semantic observer exited with code {ret}.")
        else:
            self.get_logger().warning(f"Legacy YOLO semantic observer exited with code {ret}. Restarting.")

        self.current_process = None
        self.current_export_dir = None

    def _publish_debug_image(self) -> None:
        if not self.debug_image_path.exists():
            return
        try:
            stat = self.debug_image_path.stat()
        except OSError:
            return
        if stat.st_mtime_ns != self._debug_image_last_mtime_ns:
            image_bgr = cv2.imread(str(self.debug_image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                return
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self._debug_image_cached_msg = self._bridge.cv2_to_imgmsg(image_rgb, encoding="rgb8")
            self._debug_image_last_mtime_ns = stat.st_mtime_ns
        if self._debug_image_cached_msg is None:
            return
        msg = self._debug_image_cached_msg
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_init"
        self.debug_image_pub.publish(msg)

    def destroy_node(self) -> bool:
        self._stopping = True
        if self.current_process is not None and self.current_process.poll() is None:
            self.get_logger().warning("Terminating active legacy YOLO semantic observer subprocess.")
            self.current_process.terminate()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = OVOAsyncWorkerLegacyYOLO()
    try:
        rclpy.spin(node)
    except rclpy.executors.ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
