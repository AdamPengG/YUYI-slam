from __future__ import annotations

import colorsys
import json
import os
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from rclpy.executors import ExternalShutdownException

from .data_types import ObservationLink
from .final_consolidation import write_ascii_ply
from .livo2_ovo_keyframe_exporter import Livo2OVOKeyframeExporter
from .semantic_observer import MaskObservation, SemanticObserver
from .visibility_projector import VisibleProjection


@dataclass
class InspectionJob:
    frame_index: int
    image_rgb: np.ndarray
    depth_m: np.ndarray
    c2w: np.ndarray
    xyz_world: np.ndarray


class Livo2OVOSemanticInspectionExporter(Livo2OVOKeyframeExporter):
    def __init__(self) -> None:
        super().__init__(node_name="livo2_ovo_semantic_inspection_exporter")
        self._declare_inspection_parameters()
        self._load_inspection_parameters()
        self._prepare_inspection_dir()
        self._write_inspection_manifest()

        self._class_names = self._load_class_names()
        self._class_lookup = {label: idx for idx, label in enumerate(self._class_names)}
        self._semantic_observer = SemanticObserver(
            min_mask_area=self.inspect_min_mask_area,
            min_hit_points=self.inspect_min_observation_points,
            mask_erosion_px=self.inspect_mask_erosion_px,
            abstain_margin=self.inspect_observer_abstain_margin,
            min_binding_score=self.inspect_observer_min_binding_score,
        )

        self._inspection_queue: queue.Queue[InspectionJob | None] = queue.Queue()
        self._inspection_stop = threading.Event()
        self._inspection_worker = threading.Thread(
            target=self._inspection_worker_loop,
            name="semantic-inspection-exporter",
            daemon=True,
        )
        self._inspection_worker.start()
        self.get_logger().info(
            "Semantic inspection exporter ready. "
            f"inspection_dir={self.inspect_dir}, helper={self.inspect_helper_script}"
        )

    def _declare_inspection_parameters(self) -> None:
        defaults = {
            "inspection.output_root": "/home/peng/isacc_slam/runs/semantic_inspection",
            "inspection.session_name": "",
            "inspection.overwrite_session": True,
            "inspection.ovo_root": "/home/peng/isacc_slam/reference/OVO",
            "inspection.ovo_config": "data/working/configs/ovo_livo2_vanilla.yaml",
            "inspection.dataset_name": "Replica",
            "inspection.class_set": "full",
            "inspection.topk_labels": 3,
            "inspection.device": "cuda",
            "inspection.conda_activate": "/home/peng/miniconda3/etc/profile.d/conda.sh",
            "inspection.conda_env": "ovo5090",
            "inspection.helper_script": "/home/peng/isacc_slam/src/onemap_semantic_mapper/scripts/run_semantic_inspection_infer.py",
            "inspection.depth_tolerance_m": 0.03,
            "inspection.abstain_min_score": 0.15,
            "inspection.abstain_min_margin": 0.05,
            "inspection.min_mask_area": 64,
            "inspection.min_observation_points": 12,
            "inspection.mask_erosion_px": 2,
            "inspection.observer_abstain_margin": 0.15,
            "inspection.observer_min_binding_score": 0.45,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_inspection_parameters(self) -> None:
        self.inspect_output_root = Path(str(self.get_parameter("inspection.output_root").value)).expanduser()
        session_name = str(self.get_parameter("inspection.session_name").value).strip()
        self.inspect_session_name = session_name if session_name else self.scene_name
        self.inspect_overwrite_session = bool(self.get_parameter("inspection.overwrite_session").value)
        self.inspect_ovo_root = Path(str(self.get_parameter("inspection.ovo_root").value)).expanduser()
        self.inspect_ovo_config = str(self.get_parameter("inspection.ovo_config").value)
        self.inspect_dataset_name = str(self.get_parameter("inspection.dataset_name").value)
        self.inspect_class_set = str(self.get_parameter("inspection.class_set").value)
        self.inspect_topk_labels = int(self.get_parameter("inspection.topk_labels").value)
        self.inspect_device = str(self.get_parameter("inspection.device").value)
        self.inspect_conda_activate = Path(str(self.get_parameter("inspection.conda_activate").value)).expanduser()
        self.inspect_conda_env = str(self.get_parameter("inspection.conda_env").value)
        self.inspect_helper_script = Path(str(self.get_parameter("inspection.helper_script").value)).expanduser()
        self.inspect_depth_tolerance_m = float(self.get_parameter("inspection.depth_tolerance_m").value)
        self.inspect_abstain_min_score = float(self.get_parameter("inspection.abstain_min_score").value)
        self.inspect_abstain_min_margin = float(self.get_parameter("inspection.abstain_min_margin").value)
        self.inspect_min_mask_area = int(self.get_parameter("inspection.min_mask_area").value)
        self.inspect_min_observation_points = int(self.get_parameter("inspection.min_observation_points").value)
        self.inspect_mask_erosion_px = int(self.get_parameter("inspection.mask_erosion_px").value)
        self.inspect_observer_abstain_margin = float(
            self.get_parameter("inspection.observer_abstain_margin").value
        )
        self.inspect_observer_min_binding_score = float(
            self.get_parameter("inspection.observer_min_binding_score").value
        )
        self.inspect_dir = self.inspect_output_root / self.inspect_session_name

    def _prepare_inspection_dir(self) -> None:
        self.inspect_output_root.mkdir(parents=True, exist_ok=True)
        if self.inspect_overwrite_session and self.inspect_dir.exists():
            shutil.rmtree(self.inspect_dir)
        self.inspect_dir.mkdir(parents=True, exist_ok=True)

    def _write_inspection_manifest(self) -> None:
        payload = {
            "scene_name": self.scene_name,
            "inspection_session_name": self.inspect_session_name,
            "inspection_dir": str(self.inspect_dir),
            "same_keyframes_as_exporter": True,
            "outputs": [
                "{idx}.ply",
                "{idx}_clip.ply",
                "{idx}_clip_paper.ply",
                "{idx}_ovo.ply",
                "{idx}_sam.png",
                "{idx}_clip.png",
                "{idx}_clip_textregion.png",
                "{idx}_clip_paper.png",
                "{idx}_sam_clip.png",
                "{idx}_sam_clip_textregion.png",
                "{idx}_sam_clip_paper.png",
                "{idx}_meta.json",
            ],
            "class_set": self.inspect_class_set,
            "dataset_name": self.inspect_dataset_name,
            "depth_tolerance_m": self.inspect_depth_tolerance_m,
            "abstain_min_score": self.inspect_abstain_min_score,
            "abstain_min_margin": self.inspect_abstain_min_margin,
        }
        (self.inspect_dir / "session_manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _load_class_names(self) -> list[str]:
        eval_info_path = self.inspect_ovo_root / "data" / "working" / "configs" / self.inspect_dataset_name / "eval_info.yaml"
        with eval_info_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if self.inspect_class_set == "reduced":
            return list(payload.get("class_names_reduced", payload["class_names"]))
        classes = [str(name) for name in payload["class_names"]]
        return [name for name in classes if name and name != "0"]

    def _save_frame(
        self,
        image_rgb: np.ndarray,
        depth_m: np.ndarray,
        c2w: np.ndarray,
        image_stamp: float,
        depth_stamp: float,
        pose_stamp: float,
        local_submap,
    ):
        image_name, depth_name, local_cloud_packet = super()._save_frame(
            image_rgb=image_rgb,
            depth_m=depth_m,
            c2w=c2w,
            image_stamp=image_stamp,
            depth_stamp=depth_stamp,
            pose_stamp=pose_stamp,
            local_submap=local_submap,
        )
        job = InspectionJob(
            frame_index=int(self._frame_index - 1),
            image_rgb=image_rgb.copy(),
            depth_m=depth_m.copy(),
            c2w=c2w.copy(),
            xyz_world=local_submap.xyz.astype(np.float32, copy=True),
        )
        self._inspection_queue.put(job)
        return image_name, depth_name, local_cloud_packet

    def _inspection_worker_loop(self) -> None:
        while not self._inspection_stop.is_set():
            try:
                job = self._inspection_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                self._inspection_queue.task_done()
                break
            try:
                self._process_inspection_job(job)
            except Exception as exc:  # pragma: no cover - defensive logging in worker thread
                self.get_logger().error(f"Inspection export failed for frame {job.frame_index}: {exc}")
            finally:
                self._inspection_queue.task_done()

    def _process_inspection_job(self, job: InspectionJob) -> None:
        prefix = str(job.frame_index)
        rgb_path = self.inspect_dir / f"{prefix}_rgb.png"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(job.image_rgb, cv2.COLOR_RGB2BGR))

        raw_rgb = np.full((job.xyz_world.shape[0], 3), 220, dtype=np.uint8)
        unknown_ids = np.full((job.xyz_world.shape[0],), -1, dtype=np.int32)
        write_ascii_ply(
            self.inspect_dir / f"{prefix}.ply",
            job.xyz_world.astype(np.float32, copy=False),
            raw_rgb,
            np.full((job.xyz_world.shape[0],), -1, dtype=np.int32),
            unknown_ids,
        )

        helper_stdout = self._run_helper(rgb_path=rgb_path, prefix=prefix)
        if helper_stdout:
            (self.inspect_dir / f"{prefix}_helper.log").write_text(helper_stdout, encoding="utf-8")

        masks_json_path = self.inspect_dir / f"{prefix}_masks.json"
        masks_npz_path = self.inspect_dir / f"{prefix}_masks.npz"
        masks_payload = self._load_masks_payload(masks_json_path, masks_npz_path)

        clip_class_ids, clip_rgb, clip_stats = self._assign_clip_semantics(
            xyz_world=job.xyz_world,
            c2w=job.c2w,
            depth_m=job.depth_m,
            masks_payload=masks_payload,
            variant="textregion",
        )
        write_ascii_ply(
            self.inspect_dir / f"{prefix}_clip.ply",
            job.xyz_world.astype(np.float32, copy=False),
            clip_rgb,
            np.full((job.xyz_world.shape[0],), -1, dtype=np.int32),
            clip_class_ids,
        )
        clip_paper_class_ids, clip_paper_rgb, clip_paper_stats = self._assign_clip_semantics(
            xyz_world=job.xyz_world,
            c2w=job.c2w,
            depth_m=job.depth_m,
            masks_payload=masks_payload,
            variant="paper_fusion",
        )
        write_ascii_ply(
            self.inspect_dir / f"{prefix}_clip_paper.ply",
            job.xyz_world.astype(np.float32, copy=False),
            clip_paper_rgb,
            np.full((job.xyz_world.shape[0],), -1, dtype=np.int32),
            clip_paper_class_ids,
        )

        ovo_class_ids, ovo_rgb, ovo_stats = self._assign_ovo_semantics(
            xyz_world=job.xyz_world,
            c2w=job.c2w,
            depth_m=job.depth_m,
            masks_payload=masks_payload,
        )
        write_ascii_ply(
            self.inspect_dir / f"{prefix}_ovo.ply",
            job.xyz_world.astype(np.float32, copy=False),
            ovo_rgb,
            np.full((job.xyz_world.shape[0],), -1, dtype=np.int32),
            ovo_class_ids,
        )

        meta = {
            "frame_index": int(job.frame_index),
            "point_count": int(job.xyz_world.shape[0]),
            "rgb_path": str(rgb_path),
            "sam_image_path": str(self.inspect_dir / f"{prefix}_sam.png"),
            "clip_image_path": str(self.inspect_dir / f"{prefix}_clip.png"),
            "clip_textregion_image_path": str(self.inspect_dir / f"{prefix}_clip_textregion.png"),
            "clip_paper_image_path": str(self.inspect_dir / f"{prefix}_clip_paper.png"),
            "sam_clip_image_path": str(self.inspect_dir / f"{prefix}_sam_clip.png"),
            "sam_clip_textregion_image_path": str(self.inspect_dir / f"{prefix}_sam_clip_textregion.png"),
            "sam_clip_paper_image_path": str(self.inspect_dir / f"{prefix}_sam_clip_paper.png"),
            "clip_ply_path": str(self.inspect_dir / f"{prefix}_clip.ply"),
            "clip_paper_ply_path": str(self.inspect_dir / f"{prefix}_clip_paper.ply"),
            "ovo_ply_path": str(self.inspect_dir / f"{prefix}_ovo.ply"),
            "clip": clip_stats,
            "clip_paper": clip_paper_stats,
            "ovo": ovo_stats,
        }
        (self.inspect_dir / f"{prefix}_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _run_helper(self, rgb_path: Path, prefix: str) -> str:
        command = (
            f"source {self._shell_quote(str(self.inspect_conda_activate))} && "
            f"conda activate {self._shell_quote(self.inspect_conda_env)} && "
            f"python {self._shell_quote(str(self.inspect_helper_script))} "
            f"--image-path {self._shell_quote(str(rgb_path))} "
            f"--output-dir {self._shell_quote(str(self.inspect_dir))} "
            f"--prefix {self._shell_quote(prefix)} "
            f"--ovo-root {self._shell_quote(str(self.inspect_ovo_root))} "
            f"--ovo-config {self._shell_quote(self.inspect_ovo_config)} "
            f"--dataset-name {self._shell_quote(self.inspect_dataset_name)} "
            f"--scene-name {self._shell_quote(self.scene_name)} "
            f"--class-set {self._shell_quote(self.inspect_class_set)} "
            f"--topk-labels {self.inspect_topk_labels} "
            f"--abstain-min-score {self.inspect_abstain_min_score} "
            f"--abstain-min-margin {self.inspect_abstain_min_margin} "
            f"--device {self._shell_quote(self.inspect_device)}"
        )
        env = os.environ.copy()
        env.setdefault("WANDB_DISABLED", "true")
        env.setdefault("WANDB_MODE", "disabled")
        result = subprocess.run(
            ["/bin/bash", "-lc", command],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(self.inspect_ovo_root),
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "semantic inspection helper failed: "
                f"code={result.returncode}, stdout={result.stdout}, stderr={result.stderr}"
            )
        return result.stdout.strip()

    def _shell_quote(self, value: str) -> str:
        return "'" + value.replace("'", "'\"'\"'") + "'"

    def _load_masks_payload(self, masks_json_path: Path, masks_npz_path: Path) -> dict[str, Any]:
        payload = json.loads(masks_json_path.read_text(encoding="utf-8")) if masks_json_path.exists() else {"masks": []}
        binary_masks = np.zeros((0, self.cam_height, self.cam_width), dtype=bool)
        if masks_npz_path.exists():
            with np.load(masks_npz_path) as data:
                binary_masks = np.asarray(data["binary_masks"], dtype=bool)
        payload["binary_masks"] = binary_masks
        return payload

    def _project_full_cloud(self, xyz_world: np.ndarray, c2w: np.ndarray, depth_m: np.ndarray) -> VisibleProjection:
        if xyz_world.size == 0:
            return VisibleProjection(
                visible_point_indices=np.zeros((0,), dtype=np.int32),
                projected_uv=np.zeros((0, 2), dtype=np.int32),
                projected_depth=np.zeros((0,), dtype=np.float32),
                depth_residual=np.zeros((0,), dtype=np.float32),
                zbuffer_rank=np.zeros((0,), dtype=np.int32),
                distance_to_depth_edge=np.zeros((0,), dtype=np.float32),
                visibility_score=np.zeros((0,), dtype=np.float32),
                quality_score=np.zeros((0,), dtype=np.float32),
                point_id_buffer=None,
                zbuffer_depth_buffer=None,
            )

        w2c = np.linalg.inv(c2w)
        xyz_h = np.hstack((xyz_world.astype(np.float32, copy=False), np.ones((xyz_world.shape[0], 1), dtype=np.float32)))
        xyz_cam = (w2c @ xyz_h.T).T[:, :3]
        z = xyz_cam[:, 2]
        in_front = z > 0.05
        if not np.any(in_front):
            return VisibleProjection(
                visible_point_indices=np.zeros((0,), dtype=np.int32),
                projected_uv=np.zeros((0, 2), dtype=np.int32),
                projected_depth=np.zeros((0,), dtype=np.float32),
                depth_residual=np.zeros((0,), dtype=np.float32),
                zbuffer_rank=np.zeros((0,), dtype=np.int32),
                distance_to_depth_edge=np.zeros((0,), dtype=np.float32),
                visibility_score=np.zeros((0,), dtype=np.float32),
                quality_score=np.zeros((0,), dtype=np.float32),
                point_id_buffer=None,
                zbuffer_depth_buffer=None,
            )

        xyz_cam = xyz_cam[in_front]
        point_indices = np.nonzero(in_front)[0]
        u = np.rint((xyz_cam[:, 0] * self.fx / xyz_cam[:, 2]) + self.cx).astype(np.int32)
        v = np.rint((xyz_cam[:, 1] * self.fy / xyz_cam[:, 2]) + self.cy).astype(np.int32)
        in_image = (u >= 0) & (u < self.cam_width) & (v >= 0) & (v < self.cam_height)
        if not np.any(in_image):
            return VisibleProjection(
                visible_point_indices=np.zeros((0,), dtype=np.int32),
                projected_uv=np.zeros((0, 2), dtype=np.int32),
                projected_depth=np.zeros((0,), dtype=np.float32),
                depth_residual=np.zeros((0,), dtype=np.float32),
                zbuffer_rank=np.zeros((0,), dtype=np.int32),
                distance_to_depth_edge=np.zeros((0,), dtype=np.float32),
                visibility_score=np.zeros((0,), dtype=np.float32),
                quality_score=np.zeros((0,), dtype=np.float32),
                point_id_buffer=None,
                zbuffer_depth_buffer=None,
            )

        point_indices = point_indices[in_image]
        u = u[in_image]
        v = v[in_image]
        z = xyz_cam[in_image, 2]
        pixel_id = v.astype(np.int64) * self.cam_width + u.astype(np.int64)
        order = np.lexsort((z, pixel_id))
        point_indices = point_indices[order]
        u = u[order]
        v = v[order]
        z = z[order]
        pixel_id = pixel_id[order]

        zbuffer_rank = np.zeros_like(z, dtype=np.int32)
        for idx in range(1, pixel_id.shape[0]):
            if pixel_id[idx] == pixel_id[idx - 1]:
                zbuffer_rank[idx] = zbuffer_rank[idx - 1] + 1
        _, unique_indices = np.unique(pixel_id, return_index=True)
        point_indices = point_indices[unique_indices]
        u = u[unique_indices]
        v = v[unique_indices]
        z = z[unique_indices]
        zbuffer_rank = zbuffer_rank[unique_indices]

        sampled_depth = depth_m[v, u]
        finite = np.isfinite(sampled_depth) & (sampled_depth > 0.05)
        point_indices = point_indices[finite]
        u = u[finite]
        v = v[finite]
        z = z[finite]
        zbuffer_rank = zbuffer_rank[finite]
        sampled_depth = sampled_depth[finite]
        depth_residual = sampled_depth - z
        keep = np.abs(depth_residual) <= self.inspect_depth_tolerance_m
        point_indices = point_indices[keep]
        u = u[keep]
        v = v[keep]
        z = z[keep]
        zbuffer_rank = zbuffer_rank[keep]
        depth_residual = depth_residual[keep]

        if point_indices.size == 0:
            return VisibleProjection(
                visible_point_indices=np.zeros((0,), dtype=np.int32),
                projected_uv=np.zeros((0, 2), dtype=np.int32),
                projected_depth=np.zeros((0,), dtype=np.float32),
                depth_residual=np.zeros((0,), dtype=np.float32),
                zbuffer_rank=np.zeros((0,), dtype=np.int32),
                distance_to_depth_edge=np.zeros((0,), dtype=np.float32),
                visibility_score=np.zeros((0,), dtype=np.float32),
                quality_score=np.zeros((0,), dtype=np.float32),
                point_id_buffer=None,
                zbuffer_depth_buffer=None,
            )

        depth_edge = self._depth_edge_distance(depth_m)
        distance_to_depth_edge = depth_edge[v, u]
        visibility_score = np.clip(
            1.0 - (np.abs(depth_residual) / max(self.inspect_depth_tolerance_m, 1e-4)),
            0.0,
            1.0,
        ).astype(np.float32)
        edge_score = np.clip(distance_to_depth_edge / 4.0, 0.0, 1.0)
        rank_score = np.clip(1.0 - (zbuffer_rank.astype(np.float32) * 0.25), 0.0, 1.0)
        quality_score = np.clip((visibility_score * 0.65) + (edge_score * 0.25) + (rank_score * 0.10), 0.0, 1.0)

        point_id_buffer = np.full((self.cam_height, self.cam_width), -1, dtype=np.int32)
        zbuffer_depth_buffer = np.full((self.cam_height, self.cam_width), np.inf, dtype=np.float32)
        point_id_buffer[v, u] = point_indices.astype(np.int32, copy=False)
        zbuffer_depth_buffer[v, u] = z.astype(np.float32, copy=False)
        return VisibleProjection(
            visible_point_indices=point_indices.astype(np.int32),
            projected_uv=np.column_stack((u, v)).astype(np.int32),
            projected_depth=z.astype(np.float32),
            depth_residual=depth_residual.astype(np.float32),
            zbuffer_rank=zbuffer_rank.astype(np.int32),
            distance_to_depth_edge=distance_to_depth_edge.astype(np.float32),
            visibility_score=visibility_score.astype(np.float32),
            quality_score=quality_score.astype(np.float32),
            point_id_buffer=point_id_buffer,
            zbuffer_depth_buffer=zbuffer_depth_buffer,
        )

    def _depth_edge_distance(self, depth_m: np.ndarray) -> np.ndarray:
        valid = np.isfinite(depth_m) & (depth_m > 0.05)
        if not np.any(valid):
            return np.zeros_like(depth_m, dtype=np.float32)
        depth_filled = depth_m.astype(np.float32, copy=True)
        depth_filled[~valid] = 0.0
        grad_x = cv2.Sobel(depth_filled, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_filled, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))
        edge_mask = (~valid) | (grad_mag > max(self.inspect_depth_tolerance_m * 1.5, 0.03))
        safe_region = (~edge_mask).astype(np.uint8)
        return cv2.distanceTransform(safe_region, cv2.DIST_L2, 3).astype(np.float32)

    def _assign_clip_semantics(
        self,
        xyz_world: np.ndarray,
        c2w: np.ndarray,
        depth_m: np.ndarray,
        masks_payload: dict[str, Any],
        variant: str = "textregion",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        class_ids = np.full((xyz_world.shape[0],), -1, dtype=np.int32)
        projection = self._project_full_cloud(xyz_world, c2w, depth_m)
        masks = np.asarray(masks_payload.get("binary_masks", np.zeros((0, 0, 0), dtype=bool)), dtype=bool)
        mask_entries = masks_payload.get("masks", [])
        if projection.visible_point_indices.size > 0 and masks.shape[0] > 0:
            u = projection.projected_uv[:, 0]
            v = projection.projected_uv[:, 1]
            point_scores = np.full((projection.visible_point_indices.shape[0],), -np.inf, dtype=np.float32)
            point_class_ids = np.full((projection.visible_point_indices.shape[0],), -1, dtype=np.int32)
            for mask_idx, entry in enumerate(mask_entries):
                if mask_idx >= masks.shape[0]:
                    break
                hits = masks[mask_idx, v, u]
                if not np.any(hits):
                    continue
                variant_entry = entry.get("variants", {}).get(variant, {})
                semantic_scores = variant_entry.get("semantic_scores", entry.get("semantic_scores", []))
                semantic_labels = variant_entry.get("semantic_label_candidates", entry.get("semantic_label_candidates", []))
                if not semantic_labels:
                    continue
                if not bool(variant_entry.get("accepted", entry.get("accepted", False))):
                    continue
                score = float(semantic_scores[0]) if semantic_scores else 0.0
                assigned_label = str(variant_entry.get("assigned_label", semantic_labels[0]))
                class_id = self._class_lookup.get(assigned_label, -1) if assigned_label != "abstain" else -1
                if class_id < 0:
                    continue
                update = hits & (score > point_scores)
                point_scores[update] = score
                point_class_ids[update] = class_id
            class_ids[projection.visible_point_indices] = point_class_ids
        rgb = self._class_ids_to_rgb(class_ids)
        return class_ids, rgb, self._label_stats(class_ids)

    def _assign_ovo_semantics(
        self,
        xyz_world: np.ndarray,
        c2w: np.ndarray,
        depth_m: np.ndarray,
        masks_payload: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        projection = self._project_full_cloud(xyz_world, c2w, depth_m)
        class_ids = np.full((xyz_world.shape[0],), -1, dtype=np.int32)
        if projection.visible_point_indices.size == 0:
            rgb = self._class_ids_to_rgb(class_ids)
            return class_ids, rgb, self._label_stats(class_ids)

        masks: list[MaskObservation] = []
        for mask_idx, entry in enumerate(masks_payload.get("masks", [])):
            binary_masks = np.asarray(masks_payload.get("binary_masks"), dtype=bool)
            if mask_idx >= binary_masks.shape[0]:
                break
            masks.append(
                MaskObservation(
                    mask_id=int(entry.get("mask_id", mask_idx)),
                    binary_mask=binary_masks[mask_idx],
                    bbox_xyxy=[int(v) for v in entry.get("bbox_xyxy", [0, 0, 0, 0])],
                    semantic_label_candidates=[str(v) for v in entry.get("semantic_label_candidates", [])],
                    semantic_scores=[float(v) for v in entry.get("semantic_scores", [])],
                )
            )
        observations = self._semantic_observer.observe(
            keyframe_id=0,
            local_cloud_id="inspection",
            projection=projection,
            masks=masks,
            existing_point_object_ids=None,
        )
        if observations:
            point_scores = np.full((xyz_world.shape[0],), -np.inf, dtype=np.float32)
            for observation in observations:
                if not observation.semantic_label_candidates:
                    continue
                label = str(observation.semantic_label_candidates[0])
                class_id = self._class_lookup.get(label, -1)
                score = float(observation.semantic_scores[0]) if observation.semantic_scores else 0.0
                score *= max(float(observation.quality_score), 1e-3)
                point_indices = np.asarray(observation.point_indices, dtype=np.int32)
                valid = (point_indices >= 0) & (point_indices < xyz_world.shape[0])
                point_indices = point_indices[valid]
                update = score > point_scores[point_indices]
                point_scores[point_indices[update]] = score
                class_ids[point_indices[update]] = class_id
        rgb = self._class_ids_to_rgb(class_ids)
        return class_ids, rgb, self._label_stats(class_ids)

    def _class_ids_to_rgb(self, class_ids: np.ndarray) -> np.ndarray:
        rgb = np.full((class_ids.shape[0], 3), 180, dtype=np.uint8)
        for class_id in np.unique(class_ids):
            mask = class_ids == class_id
            rgb[mask] = self._semantic_color(int(class_id))
        return rgb

    def _semantic_color(self, class_id: int) -> np.ndarray:
        if class_id < 0:
            return np.array([180, 180, 180], dtype=np.uint8)
        hue = (class_id * 0.137) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        return np.asarray([int(255 * channel) for channel in rgb], dtype=np.uint8)

    def _label_stats(self, class_ids: np.ndarray) -> dict[str, Any]:
        labels: dict[str, int] = {}
        labeled = class_ids[class_ids >= 0]
        unique_ids, counts = np.unique(labeled, return_counts=True) if labeled.size > 0 else ([], [])
        for class_id, count in zip(unique_ids, counts):
            label = self._class_names[int(class_id)] if int(class_id) < len(self._class_names) else str(class_id)
            labels[str(label)] = int(count)
        return {
            "labeled_point_count": int(labeled.size),
            "label_counts": labels,
        }

    def destroy_node(self) -> bool:
        self._inspection_stop.set()
        self._inspection_queue.put(None)
        if self._inspection_worker.is_alive():
            self._inspection_worker.join(timeout=5.0)
        return super().destroy_node()


def main(args=None) -> None:
    import rclpy

    rclpy.init(args=args)
    node = Livo2OVOSemanticInspectionExporter()
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
