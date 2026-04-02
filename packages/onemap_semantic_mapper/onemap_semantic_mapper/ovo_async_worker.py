from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node


class OVOAsyncWorker(Node):
    def __init__(self) -> None:
        super().__init__("ovo_async_worker")
        self._declare_parameters()
        self._load_parameters()

        self.current_process: Optional[subprocess.Popen] = None
        self.current_export_dir: Path | None = None
        self._stopping = False
        self._waiting_log_emitted = False
        self._startup_wall_time_ns = time.time_ns()

        self.timer = self.create_timer(self.poll_period_sec, self._poll)
        self.get_logger().info(
            "Semantic observer worker ready. "
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
            "ovo_config": "data/working/configs/ovo_livo2_vanilla.yaml",
            "experiment_name": "isaac_livo2_online_vanilla",
            "artifact_path": "/home/peng/isacc_slam/reference/OVO/data/output/Replica/isaac_livo2_online_vanilla/isaac_turtlebot3_livo2_online/semantic_snapshot.npz",
            "observer_script": "/home/peng/isacc_slam/src/onemap_semantic_mapper/scripts/run_semantic_observer_online.py",
            "poll_period_sec": 5.0,
            "min_keyframes": 5,
            "rerun_every_new_keyframes": 1,
            "clear_output_on_start": False,
            "resume_if_exists": True,
            "device": "cuda",
            "class_set": "full",
            "topk_labels": 3,
            "visibility_depth_tolerance_m": 0.03,
            "snapshot_voxel_size_m": 0.0,
            "snapshot_max_points": 5000000,
            "snapshot_include_unassigned_points": True,
            "export_stale_objects": True,
            "online_cleanup_object_points": False,
            "min_observation_points": 4,
            "min_mask_area": 16,
            "mask_erosion_px": 0,
            "merge_centroid_radius_m": 0.75,
            "observer_abstain_margin": 0.02,
            "observer_min_binding_score": 0.15,
            "semantic_keyframe_max_gap": 2,
            "semantic_novelty_thresh": 0.08,
            "tentative_min_obs": 2,
            "geometry_voxel_size_m": 0.02,
            "geometry_truncation_m": 0.08,
            "geometry_surface_band_m": 0.03,
            "direct_semantic_only": True,
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
        self.class_set = str(self.get_parameter("class_set").value)
        self.topk_labels = int(self.get_parameter("topk_labels").value)
        self.visibility_depth_tolerance_m = float(self.get_parameter("visibility_depth_tolerance_m").value)
        self.snapshot_voxel_size_m = float(self.get_parameter("snapshot_voxel_size_m").value)
        self.snapshot_max_points = int(self.get_parameter("snapshot_max_points").value)
        self.snapshot_include_unassigned_points = bool(self.get_parameter("snapshot_include_unassigned_points").value)
        self.export_stale_objects = bool(self.get_parameter("export_stale_objects").value)
        self.online_cleanup_object_points = bool(self.get_parameter("online_cleanup_object_points").value)
        self.min_observation_points = int(self.get_parameter("min_observation_points").value)
        self.min_mask_area = int(self.get_parameter("min_mask_area").value)
        self.mask_erosion_px = int(self.get_parameter("mask_erosion_px").value)
        self.merge_centroid_radius_m = float(self.get_parameter("merge_centroid_radius_m").value)
        self.observer_abstain_margin = float(self.get_parameter("observer_abstain_margin").value)
        self.observer_min_binding_score = float(self.get_parameter("observer_min_binding_score").value)
        self.semantic_keyframe_max_gap = int(self.get_parameter("semantic_keyframe_max_gap").value)
        self.semantic_novelty_thresh = float(self.get_parameter("semantic_novelty_thresh").value)
        self.tentative_min_obs = int(self.get_parameter("tentative_min_obs").value)
        self.geometry_voxel_size_m = float(self.get_parameter("geometry_voxel_size_m").value)
        self.geometry_truncation_m = float(self.get_parameter("geometry_truncation_m").value)
        self.geometry_surface_band_m = float(self.get_parameter("geometry_surface_band_m").value)
        self.direct_semantic_only = bool(self.get_parameter("direct_semantic_only").value)
        self.status_path = self.artifact_path.parent / "online_status.json"

    def _scene_dir(self) -> Path:
        return self.dataset_root / self.scene_name

    def _resolve_export_dir(self) -> Path | None:
        if self.export_dir_override is not None:
            export_dir = self.export_dir_override
            if not export_dir.exists():
                return None
            if not self._session_is_acceptable(export_dir):
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
        if not self._session_is_acceptable(export_dir):
            return None
        if not (export_dir / "keyframe_packets.jsonl").exists():
            return None
        if not (export_dir / "local_cloud_packets.jsonl").exists():
            return None
        return export_dir

    def _session_is_acceptable(self, export_dir: Path) -> bool:
        session_path = export_dir / "session.json"
        if self.resume_if_exists:
            return session_path.exists()
        if not session_path.exists():
            return False
        try:
            session_mtime_ns = session_path.stat().st_mtime_ns
            if session_mtime_ns >= self._startup_wall_time_ns:
                return True

            # Exporter creates session.json before keyframe/local-cloud packets begin
            # streaming, so a fresh run can legitimately have an older session file
            # than the worker startup time. Accept the export once packet manifests
            # or the export directory itself have been updated after worker startup.
            export_mtime_ns = export_dir.stat().st_mtime_ns
            if export_mtime_ns >= self._startup_wall_time_ns:
                return True

            for packet_name in ("keyframe_packets.jsonl", "local_cloud_packets.jsonl"):
                packet_path = export_dir / packet_name
                if packet_path.exists() and packet_path.stat().st_mtime_ns >= self._startup_wall_time_ns:
                    return True
            return False
        except OSError:
            return False

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
            "--class-set",
            shlex.quote(self.class_set),
            "--topk-labels",
            shlex.quote(str(self.topk_labels)),
            "--visibility-depth-tolerance-m",
            shlex.quote(str(self.visibility_depth_tolerance_m)),
            "--snapshot-voxel-size-m",
            shlex.quote(str(self.snapshot_voxel_size_m)),
            "--snapshot-max-points",
            shlex.quote(str(self.snapshot_max_points)),
            "--snapshot-include-unassigned-points",
            shlex.quote("1" if self.snapshot_include_unassigned_points else "0"),
            "--export-stale-objects",
            shlex.quote("1" if self.export_stale_objects else "0"),
            "--online-cleanup-object-points",
            shlex.quote("1" if self.online_cleanup_object_points else "0"),
            "--min-observation-points",
            shlex.quote(str(self.min_observation_points)),
            "--min-mask-area",
            shlex.quote(str(self.min_mask_area)),
            "--mask-erosion-px",
            shlex.quote(str(self.mask_erosion_px)),
            "--merge-centroid-radius-m",
            shlex.quote(str(self.merge_centroid_radius_m)),
            "--observer-abstain-margin",
            shlex.quote(str(self.observer_abstain_margin)),
            "--observer-min-binding-score",
            shlex.quote(str(self.observer_min_binding_score)),
            "--semantic-keyframe-max-gap",
            shlex.quote(str(self.semantic_keyframe_max_gap)),
            "--semantic-novelty-thresh",
            shlex.quote(str(self.semantic_novelty_thresh)),
            "--tentative-min-obs",
            shlex.quote(str(self.tentative_min_obs)),
            "--geometry-voxel-size-m",
            shlex.quote(str(self.geometry_voxel_size_m)),
            "--geometry-truncation-m",
            shlex.quote(str(self.geometry_truncation_m)),
            "--geometry-surface-band-m",
            shlex.quote(str(self.geometry_surface_band_m)),
            "--direct-semantic-only",
            shlex.quote("1" if self.direct_semantic_only else "0"),
        ]
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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        self.get_logger().info(
            "Started semantic observer subprocess for "
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
            self.get_logger().info("Semantic observer exited during shutdown.")
            self.current_process = None
            return

        if self.status_path.exists():
            try:
                status_text = self.status_path.read_text(encoding="utf-8")
                self.get_logger().warning(
                    f"Semantic observer exited with code {ret}. Last status: {status_text}"
                )
            except OSError:
                self.get_logger().warning(f"Semantic observer exited with code {ret}.")
        else:
            self.get_logger().warning(f"Semantic observer exited with code {ret}. Restarting.")

        self.current_process = None
        self.current_export_dir = None

    def destroy_node(self) -> bool:
        self._stopping = True
        if self.current_process is not None and self.current_process.poll() is None:
            self.get_logger().warning("Terminating active semantic observer subprocess.")
            self.current_process.terminate()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = OVOAsyncWorker()
    try:
        rclpy.spin(node)
    except rclpy.executors.ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
