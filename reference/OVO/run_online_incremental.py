from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
import yaml

from ovo.entities.logger import Logger
from ovo.entities.ovo import OVO
from ovo.slam.vanilla_mapper import VanillaMapper
from ovo.utils import gen_utils, io_utils
from render_run_overview import render_run_overview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent online OVO vanilla runner for live-exported keyframes.")
    parser.add_argument("--dataset_name", default="Replica")
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--ovo_config", default="data/working/configs/ovo_livo2_vanilla.yaml")
    parser.add_argument("--poll_period_sec", type=float, default=2.0)
    parser.add_argument("--min_keyframes", type=int, default=5)
    parser.add_argument("--export_every_new_keyframes", type=int, default=1)
    parser.add_argument("--render_overview_every_new_keyframes", type=int, default=5)
    parser.add_argument("--render_prefix", default="online")
    parser.add_argument("--semantic_mode", choices=["instance", "semantic"], default="semantic")
    parser.add_argument("--snapshot_max_points", type=int, default=250000)
    parser.add_argument("--run_once", action="store_true")
    parser.add_argument("--clear_output", action="store_true")
    parser.add_argument("--resume_if_exists", action="store_true")
    return parser.parse_args()


def build_config(dataset_name: str, scene_name: str, ovo_config_path: str) -> Dict[str, Any]:
    config = io_utils.load_config(ovo_config_path)
    map_module = config["slam"].get("slam_module", "vanilla")
    if map_module.startswith("orbslam"):
        map_module = "vanilla"

    config_slam = io_utils.load_config(
        os.path.join(config["slam"]["config_path"], map_module, f"{dataset_name.lower()}.yaml")
    )
    io_utils.update_recursive(config, config_slam)

    config_dataset = io_utils.load_config(f"data/working/configs/{dataset_name}/{dataset_name.lower()}.yaml")
    io_utils.update_recursive(config, config_dataset)

    scene_cfg_path = Path(f"data/working/configs/{dataset_name}/{scene_name}.yaml")
    if scene_cfg_path.exists():
        config_scene = io_utils.load_config(str(scene_cfg_path))
        io_utils.update_recursive(config, config_scene)

    config.setdefault("data", {})
    config["data"]["scene_name"] = scene_name
    config["data"]["input_path"] = f"data/input/Datasets/{dataset_name}/{scene_name}"
    config["use_wandb"] = False
    config.setdefault("vis", {})
    config["vis"]["stream"] = False
    config["vis"]["show_stream"] = False
    return config


def load_classes(dataset_name: str) -> list[str]:
    eval_info_path = Path(f"data/working/configs/{dataset_name}/eval_info.yaml")
    with eval_info_path.open("r", encoding="utf-8") as handle:
        info = yaml.safe_load(handle)
    return info.get("class_names_reduced", info["class_names"])


def load_replica_frame(scene_dir: Path, frame_id: int, cam_cfg: Dict[str, Any]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    image_path = scene_dir / "results" / f"frame{frame_id:06d}.jpg"
    depth_path = scene_dir / "results" / f"depth{frame_id:06d}.png"
    traj_path = scene_dir / "traj.txt"

    if not image_path.exists() or not depth_path.exists():
        raise FileNotFoundError(f"Missing keyframe files for frame {frame_id}: {image_path} / {depth_path}")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to load image {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise RuntimeError(f"Failed to load depth {depth_path}")
    depth_m = depth_raw.astype(np.float32) / float(cam_cfg["depth_scale"])

    with traj_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx == frame_id:
                pose = np.asarray([float(v) for v in line.split()], dtype=np.float32).reshape(4, 4)
                break
        else:
            raise IndexError(f"traj.txt does not contain pose for frame {frame_id}")

    return frame_id, image_rgb, depth_m, pose


def count_ready_frames(scene_dir: Path) -> int:
    results_dir = scene_dir / "results"
    if not results_dir.exists():
        return 0
    image_count = len(list(results_dir.glob("frame*.jpg")))
    depth_count = len(list(results_dir.glob("depth*.png")))
    traj_path = scene_dir / "traj.txt"
    pose_count = 0
    if traj_path.exists():
        with traj_path.open("r", encoding="utf-8") as handle:
            pose_count = sum(1 for _ in handle)
    return min(image_count, depth_count, pose_count)


def build_obj_to_class(ovo: OVO, classes: list[str]) -> tuple[dict[int, int], str | None]:
    try:
        instances_info = ovo.classify_instances(classes)
        object_ids = list(ovo.objects.keys())
        return {
            int(obj_id): int(instances_info["classes"][idx])
            for idx, obj_id in enumerate(object_ids)
        }, None
    except Exception as exc:
        return {}, str(exc)


def export_semantic_snapshot(
    output_path: Path,
    ovo: OVO,
    map_dict: Dict[str, Any],
    classes: list[str],
    mode: str,
) -> None:
    xyz = map_dict["xyz"].detach().cpu().numpy().astype(np.float32)
    obj_ids = map_dict["obj_ids"].squeeze(-1).detach().cpu().numpy().astype(np.int32)

    obj_to_class, classify_error = build_obj_to_class(ovo, classes)
    unique_obj_ids = sorted(int(v) for v in np.unique(obj_ids) if int(v) >= 0)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inst_palette = (plt.get_cmap("tab20b")(np.linspace(0, 1, max(20, len(unique_obj_ids) + 1)))[:, :3] * 255).astype(np.uint8)
    sem_palette = (plt.get_cmap("tab20")(np.linspace(0, 1, max(20, len(classes) + 1)))[:, :3] * 255).astype(np.uint8)

    rgb = np.full((xyz.shape[0], 3), 180, dtype=np.uint8)
    class_ids = np.full((xyz.shape[0],), -1, dtype=np.int32)

    obj_to_color_index = {obj_id: idx for idx, obj_id in enumerate(unique_obj_ids)}
    instance_centers = []
    instance_labels = []
    instance_ids = []

    for obj_id in unique_obj_ids:
        mask = obj_ids == obj_id
        class_id = int(obj_to_class.get(obj_id, -1))
        class_ids[mask] = class_id
        if mode == "semantic" and class_id >= 0:
            rgb[mask] = sem_palette[class_id % len(sem_palette)]
        else:
            rgb[mask] = inst_palette[obj_to_color_index[obj_id] % len(inst_palette)]

        center = xyz[mask].mean(axis=0).astype(np.float32)
        class_name = classes[class_id] if 0 <= class_id < len(classes) else "unknown"
        instance_centers.append(center)
        instance_labels.append(class_name)
        instance_ids.append(obj_id)

    output_npz = output_path / "semantic_snapshot.npz"
    output_summary = output_path / "semantic_snapshot_summary.txt"
    tmp_npz_path = output_npz.with_name(output_npz.name + ".tmp.npz")
    if tmp_npz_path.exists():
        tmp_npz_path.unlink()
    np.savez_compressed(
        tmp_npz_path,
        xyz=xyz,
        rgb=rgb,
        instance_id=obj_ids,
        class_id=class_ids,
        instance_ids=np.asarray(instance_ids, dtype=np.int32),
        instance_centers=np.asarray(instance_centers, dtype=np.float32) if instance_centers else np.zeros((0, 3), dtype=np.float32),
        instance_labels=np.asarray(instance_labels, dtype=object),
        class_names=np.asarray(classes, dtype=object),
        classify_error=np.asarray("" if classify_error is None else classify_error, dtype=object),
        mode=np.asarray(mode, dtype=object),
    )
    tmp_npz_path.replace(output_npz)

    lines = [
        f"num_points={xyz.shape[0]}",
        f"num_instances={len(unique_obj_ids)}",
        f"mode={mode}",
    ]
    if classify_error is not None:
        lines.append(f"classification_warning={classify_error}")
    for obj_id, class_name, center in zip(instance_ids[:50], instance_labels[:50], instance_centers[:50]):
        mask = obj_ids == obj_id
        lines.append(
            f"instance {obj_id}: class={class_name}, points={int(mask.sum())}, "
            f"center={[round(float(v), 4) for v in center.tolist()]}"
        )
    output_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


class OnlineOVORunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = build_config(args.dataset_name, args.scene_name, args.ovo_config)
        self.device = self.config.get("device", "cuda")
        self.scene_dir = Path(self.config["data"]["input_path"])
        self.output_path = Path("data/output") / args.dataset_name / args.experiment_name / args.scene_name
        self.output_path.mkdir(parents=True, exist_ok=True)

        if args.clear_output and self.output_path.exists():
            for child in self.output_path.iterdir():
                if child.is_file():
                    child.unlink()

        io_utils.save_dict_to_yaml(self.config, "config.yaml", directory=self.output_path)
        gen_utils.setup_seed(self.config["seed"])

        self.logger = Logger(self.output_path, os.getpid(), use_wandb=False)
        cam_cfg = self.config["cam"]
        self.cam_intrinsics = torch.tensor(
            [
                [cam_cfg["fx"], 0.0, cam_cfg["cx"]],
                [0.0, cam_cfg["fy"], cam_cfg["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.ovo = OVO(self.config["semantic"], self.logger, self.config["data"]["scene_name"], self.cam_intrinsics, device=self.device)
        self.mapper = VanillaMapper(self.config, self.cam_intrinsics)
        self.classes = load_classes(args.dataset_name)

        self.processed_frames = 0
        self.last_exported_frame_count = 0
        self.last_rendered_frame_count = 0

        if args.resume_if_exists and (self.output_path / "ovo_map.ckpt").exists():
            self._restore_state()

        self.status_path = self.output_path / "online_status.json"
        self._write_status("ready")

    def _restore_state(self) -> None:
        ckpt = torch.load(self.output_path / "ovo_map.ckpt", map_location=self.device, weights_only=False)
        self.mapper.set_map_dict(ckpt["map_params"])
        self.ovo.restore_dict(ckpt["ovo_map_params"], debug_info=self.config.get("debug", False))
        cam_path = self.output_path / "estimated_c2w.npy"
        if cam_path.exists():
            c2w = torch.load(cam_path, weights_only=False)
            self.mapper.set_cam_dict(c2w)
            if self.mapper.estimated_c2ws:
                self.processed_frames = max(int(k) for k in self.mapper.estimated_c2ws.keys()) + 1
                self.last_exported_frame_count = self.processed_frames
        print(f"Restored incremental OVO state from {self.output_path}, next_frame={self.processed_frames}")

    def _write_status(self, state: str, extra: Dict[str, Any] | None = None) -> None:
        payload = {
            "state": state,
            "scene_name": self.args.scene_name,
            "experiment_name": self.args.experiment_name,
            "processed_frames": self.processed_frames,
            "last_exported_frame_count": self.last_exported_frame_count,
            "output_path": str(self.output_path),
        }
        if extra:
            payload.update(extra)
        self.status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _save_representation(self) -> None:
        submap_ckpt = {
            "map_params": self.mapper.get_map_dict(),
            "ovo_map_params": self.ovo.capture_dict(debug_info=self.config.get("debug", False)),
        }
        io_utils.save_dict_to_ckpt(submap_ckpt, "ovo_map.ckpt", directory=self.output_path)
        if self.config["slam"].get("save_estimated_cam", False):
            c2w = self.mapper.get_cam_dict()
            torch.save(c2w, self.output_path / "estimated_c2w.npy")

    def _export_outputs(self, force_render: bool = False) -> None:
        self.ovo.complete_semantic_info()
        self._save_representation()
        export_semantic_snapshot(
            output_path=self.output_path,
            ovo=self.ovo,
            map_dict=self.mapper.get_map_dict(),
            classes=self.classes,
            mode=self.args.semantic_mode,
        )
        if force_render or (
            self.args.render_overview_every_new_keyframes > 0
            and self.processed_frames >= self.last_rendered_frame_count + self.args.render_overview_every_new_keyframes
        ):
            render_run_overview(
                self.output_path,
                max_points=self.args.snapshot_max_points,
                prefix=self.args.render_prefix,
            )
            self.last_rendered_frame_count = self.processed_frames
        self.last_exported_frame_count = self.processed_frames
        self._write_status(
            "exported",
            {
                "num_instances": len(self.ovo.objects),
                "num_points": int(self.mapper.pcd.shape[0]),
            },
        )

    def _process_frame(self, frame_id: int) -> None:
        frame_data = load_replica_frame(self.scene_dir, frame_id, self.config["cam"])
        self.mapper.track_camera(frame_data)
        estimated_c2w = self.mapper.get_c2w(frame_id)
        if estimated_c2w is None:
            return

        missing_depth = not (frame_data[2] > 0).any()
        if missing_depth:
            return

        if frame_id % self.config["mapping"].get("map_every", 1) == 0:
            self.mapper.map(frame_data, estimated_c2w)

        if frame_id % self.config["semantic"].get("segment_every", 1) == 0:
            with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                image = frame_data[1]
                scene_data = [frame_id, image, frame_data[2], ()]
                map_data = self.mapper.get_map()
                updated_points_ins_ids = self.ovo.detect_and_track_objects(scene_data, map_data, estimated_c2w)
                if updated_points_ins_ids is not None:
                    self.mapper.update_pcd_obj_ids(updated_points_ins_ids)
                self.ovo.compute_semantic_info()
                self.logger.log_memory_usage(frame_id)

        if frame_id > 0 and frame_id % 50 == 0:
            gc.collect()

    def run(self) -> None:
        self._write_status("running")
        while True:
            ready_frames = count_ready_frames(self.scene_dir)

            if ready_frames < self.args.min_keyframes:
                self._write_status("waiting_for_keyframes", {"ready_frames": ready_frames})
                if self.args.run_once:
                    time.sleep(self.args.poll_period_sec)
                    ready_frames = count_ready_frames(self.scene_dir)
                    if ready_frames < self.args.min_keyframes:
                        return
                else:
                    time.sleep(self.args.poll_period_sec)
                    continue

            if self.processed_frames < ready_frames:
                for frame_id in range(self.processed_frames, ready_frames):
                    self._process_frame(frame_id)
                    self.processed_frames = frame_id + 1
                    self._write_status(
                        "processing",
                        {
                            "ready_frames": ready_frames,
                            "num_instances": len(self.ovo.objects),
                            "num_points": int(self.mapper.pcd.shape[0]),
                        },
                    )

                if self.processed_frames >= self.last_exported_frame_count + self.args.export_every_new_keyframes:
                    self._export_outputs()
            elif self.args.run_once:
                if self.processed_frames > self.last_exported_frame_count:
                    self._export_outputs(force_render=True)
                return
            else:
                time.sleep(self.args.poll_period_sec)


def main() -> None:
    args = parse_args()
    runner = OnlineOVORunner(args)
    runner.run()


if __name__ == "__main__":
    main()
