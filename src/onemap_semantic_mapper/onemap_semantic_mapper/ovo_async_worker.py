from __future__ import annotations

import os
import shlex
import subprocess
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
        self._stopping = False

        self.timer = self.create_timer(self.poll_period_sec, self._poll)
        self.get_logger().info(
            "OVO async worker ready. "
            f"scene={self.scene_name}, experiment={self.experiment_name}, poll={self.poll_period_sec}s"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "scene_name": "isaac_turtlebot3_livo2_online",
            "dataset_name": "Replica",
            "ovo_root": "/home/peng/isacc_slam/reference/OVO",
            "ovo_config": "data/working/configs/ovo_livo2_vanilla.yaml",
            "experiment_name": "isaac_livo2_online_vanilla",
            "render_prefix": "online",
            "poll_period_sec": 5.0,
            "min_keyframes": 5,
            "rerun_every_new_keyframes": 1,
            "render_overview_every_new_keyframes": 5,
            "semantic_mode": "semantic",
            "snapshot_max_points": 250000,
            "clear_output_on_start": False,
            "resume_if_exists": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_parameters(self) -> None:
        self.scene_name = str(self.get_parameter("scene_name").value)
        self.dataset_name = str(self.get_parameter("dataset_name").value)
        self.ovo_root = Path(str(self.get_parameter("ovo_root").value)).expanduser()
        self.ovo_config = str(self.get_parameter("ovo_config").value)
        self.experiment_name = str(self.get_parameter("experiment_name").value)
        self.render_prefix = str(self.get_parameter("render_prefix").value)
        self.poll_period_sec = float(self.get_parameter("poll_period_sec").value)
        self.min_keyframes = int(self.get_parameter("min_keyframes").value)
        self.export_every_new_keyframes = int(self.get_parameter("rerun_every_new_keyframes").value)
        self.render_overview_every_new_keyframes = int(
            self.get_parameter("render_overview_every_new_keyframes").value
        )
        self.semantic_mode = str(self.get_parameter("semantic_mode").value)
        self.snapshot_max_points = int(self.get_parameter("snapshot_max_points").value)
        self.clear_output_on_start = bool(self.get_parameter("clear_output_on_start").value)
        self.resume_if_exists = bool(self.get_parameter("resume_if_exists").value)
        self.status_path = (
            self.ovo_root
            / "data"
            / "output"
            / self.dataset_name
            / self.experiment_name
            / self.scene_name
            / "online_status.json"
        )

    def _build_command(self) -> str:
        cmd = [
            "cd",
            shlex.quote(str(self.ovo_root)),
            "&&",
            "source /home/peng/miniconda3/etc/profile.d/conda.sh",
            "&&",
            "conda activate ovo5090",
            "&&",
            "python",
            "run_online_incremental.py",
            "--dataset_name",
            shlex.quote(self.dataset_name),
            "--scene_name",
            shlex.quote(self.scene_name),
            "--experiment_name",
            shlex.quote(self.experiment_name),
            "--ovo_config",
            shlex.quote(self.ovo_config),
            "--poll_period_sec",
            shlex.quote(str(self.poll_period_sec)),
            "--min_keyframes",
            shlex.quote(str(self.min_keyframes)),
            "--export_every_new_keyframes",
            shlex.quote(str(self.export_every_new_keyframes)),
            "--render_overview_every_new_keyframes",
            shlex.quote(str(self.render_overview_every_new_keyframes)),
            "--render_prefix",
            shlex.quote(self.render_prefix),
            "--semantic_mode",
            shlex.quote(self.semantic_mode),
            "--snapshot_max_points",
            shlex.quote(str(self.snapshot_max_points)),
        ]
        if self.clear_output_on_start:
            cmd.append("--clear_output")
        if self.resume_if_exists:
            cmd.append("--resume_if_exists")
        return " ".join(cmd)

    def _start_process(self) -> None:
        command = self._build_command()
        self.current_process = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            cwd=str(self.ovo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        self.get_logger().info(
            f"Started persistent online OVO worker for scene={self.scene_name}, experiment={self.experiment_name}."
        )

    def _poll(self) -> None:
        if self.current_process is None:
            self._start_process()
            return

        ret = self.current_process.poll()
        if ret is None:
            return

        if self._stopping:
            self.get_logger().info("OVO online worker exited during shutdown.")
            self.current_process = None
            return

        if self.status_path.exists():
            try:
                status_text = self.status_path.read_text(encoding="utf-8")
                self.get_logger().warning(
                    f"OVO online worker exited with code {ret}. Last status: {status_text}"
                )
            except OSError:
                self.get_logger().warning(f"OVO online worker exited with code {ret}.")
        else:
            self.get_logger().warning(f"OVO online worker exited with code {ret}. Restarting.")

        self.current_process = None

    def destroy_node(self) -> bool:
        self._stopping = True
        if self.current_process is not None and self.current_process.poll() is None:
            self.get_logger().warning("Terminating active online OVO subprocess.")
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
