from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .semantic_observer import MaskObservation

STRUCTURAL_LABELS = {
    "wall",
    "floor",
    "ceiling",
    "door",
    "window",
    "blinds",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class YOLOE26XAdapter:
    def __init__(
        self,
        semantic_cfg: dict[str, Any],
        class_names: list[str],
        scene_name: str,
        device: str,
        topk_labels: int,
    ) -> None:
        self.semantic_cfg = semantic_cfg
        self.class_names = list(class_names)
        self.scene_name = scene_name
        self.device = device
        self.topk_labels = max(int(topk_labels), 1)

        yoloe_cfg = dict(semantic_cfg.get("yoloe", {}))
        self.yoloe_cfg = yoloe_cfg
        self.repo_root = Path(yoloe_cfg.get("repo_root", _repo_root() / "reference" / "YOLOE_official")).expanduser()
        self.helper_script = Path(
            yoloe_cfg.get(
                "helper_script",
                _repo_root() / "src" / "onemap_semantic_mapper" / "scripts" / "run_yoloe26x_region_infer.py",
            )
        ).expanduser()
        self.conda_env = str(yoloe_cfg.get("conda_env", "yoloe_env")).strip()
        self.python_bin = str(yoloe_cfg.get("python_bin", "")).strip()
        self.cuda_visible_devices = str(yoloe_cfg.get("cuda_visible_devices", "1")).strip()
        self.model_path = str(yoloe_cfg.get("model_path", "yoloe-26x-seg.pt")).strip()
        self.conf_thresh = float(yoloe_cfg.get("conf_thresh", 0.10))
        self.iou_thresh = float(yoloe_cfg.get("iou_thresh", 0.50))
        self.max_det = int(yoloe_cfg.get("max_det", 100))
        self.helper_device = str(yoloe_cfg.get("device", device)).strip()

    def _build_python_command(self) -> list[str]:
        if self.python_bin:
            return [self.python_bin, str(self.helper_script)]
        env_python = Path(f"/home/peng/miniconda3/envs/{self.conda_env}/bin/python")
        if env_python.exists():
            return [str(env_python), str(self.helper_script)]
        conda_exe = shutil.which("conda") or "/home/peng/miniconda3/bin/conda"
        return [conda_exe, "run", "-n", self.conda_env, "python", str(self.helper_script)]

    def _run_helper(self, image_rgb: np.ndarray) -> tuple[dict[str, Any], np.ndarray | None]:
        if not self.helper_script.exists():
            raise RuntimeError(f"YOLOE helper script not found: {self.helper_script}")

        with tempfile.TemporaryDirectory(prefix="yoloe_frontend_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            image_path = temp_dir / "frame.png"
            class_names_path = temp_dir / "class_names.json"
            output_json = temp_dir / "yoloe_result.json"
            output_masks = temp_dir / "yoloe_masks.npz"

            cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            class_names_path.write_text(
                json.dumps({"class_names": self.class_names}, ensure_ascii=False),
                encoding="utf-8",
            )

            cmd = self._build_python_command() + [
                "--image-path",
                str(image_path),
                "--class-names-json",
                str(class_names_path),
                "--output-json",
                str(output_json),
                "--output-masks-npz",
                str(output_masks),
                "--model-path",
                self.model_path,
                "--device",
                self.helper_device,
                "--conf-thresh",
                str(self.conf_thresh),
                "--iou-thresh",
                str(self.iou_thresh),
                "--max-det",
                str(self.max_det),
                "--topk",
                str(self.topk_labels),
                "--repo-root",
                str(self.repo_root),
            ]
            env = dict(os.environ)
            if self.cuda_visible_devices:
                env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices

            completed = subprocess.run(
                cmd,
                cwd=str(self.repo_root if self.repo_root.exists() else _repo_root()),
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if completed.returncode != 0:
                stderr = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(f"YOLOE helper failed: {stderr}")
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            masks = None
            if output_masks.exists():
                with np.load(output_masks, allow_pickle=False) as data:
                    if "masks" in data:
                        masks = data["masks"].astype(bool, copy=False)
            return payload, masks

    def build_mask_observations(self, image_rgb: np.ndarray, frame_id: int) -> list[MaskObservation]:
        helper_payload, masks = self._run_helper(image_rgb)
        results = helper_payload.get("results", [])
        if not isinstance(results, list) or not results or masks is None or masks.shape[0] == 0:
            return []

        observations: list[MaskObservation] = []
        count = min(len(results), masks.shape[0])
        for idx in range(count):
            item = results[idx]
            binary_mask = masks[idx]
            ys, xs = np.nonzero(binary_mask)
            if xs.size == 0:
                continue
            bbox_xyxy = item.get("bbox_xyxy") or [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            label_candidates = [str(v) for v in item.get("semantic_label_candidates", [])][: self.topk_labels]
            label_scores = [float(v) for v in item.get("semantic_scores", [])][: len(label_candidates)]
            if not label_candidates:
                continue
            mask_area = int(binary_mask.sum())
            image_area = int(binary_mask.shape[0] * binary_mask.shape[1])
            top_label = str(label_candidates[0]).strip().lower()
            structural = [
                (str(label), float(score))
                for label, score in zip(label_candidates, label_scores, strict=False)
                if str(label).strip().lower() in STRUCTURAL_LABELS
            ]
            non_structural = [
                (str(label), float(score))
                for label, score in zip(label_candidates, label_scores, strict=False)
                if str(label).strip().lower() not in STRUCTURAL_LABELS
            ]
            if top_label in STRUCTURAL_LABELS:
                bbox_w = max(int(bbox_xyxy[2]) - int(bbox_xyxy[0]), 0)
                bbox_h = max(int(bbox_xyxy[3]) - int(bbox_xyxy[1]), 0)
                large_structural = (
                    mask_area >= int(image_area * 0.22)
                    or bbox_w >= int(binary_mask.shape[1] * 0.65)
                    or bbox_h >= int(binary_mask.shape[0] * 0.65)
                )
                if large_structural:
                    observation_kind = "stuff"
                    filtered = structural
                else:
                    if non_structural:
                        observation_kind = "thing"
                        filtered = non_structural
                    else:
                        continue
            else:
                observation_kind = "thing"
                filtered = non_structural
            if not filtered:
                continue
            label_candidates = [label for label, _score in filtered][: self.topk_labels]
            label_scores = [float(score) for _label, score in filtered][: len(label_candidates)]
            observations.append(
                MaskObservation(
                    mask_id=int(item.get("mask_id", idx)),
                    binary_mask=binary_mask,
                    bbox_xyxy=[int(v) for v in bbox_xyxy],
                    semantic_label_candidates=label_candidates,
                    semantic_scores=label_scores,
                    semantic_embedding=None,
                    semantic_embedding_variant="yoloe26x",
                    yolo_label_candidates=label_candidates,
                    yolo_scores=label_scores,
                    detection_score=float(label_scores[0]) if label_scores else 0.0,
                    observation_kind=str(observation_kind),
                    view_quality=float(label_scores[0]) if label_scores else 0.0,
                )
            )
        return observations

    def rank_descriptor(self, descriptor: np.ndarray | list[float], topk: int | None = None) -> list[tuple[str, float]]:
        return []

    def classify_descriptor(self, descriptor: np.ndarray | list[float]) -> tuple[str, float]:
        return "unknown", 0.0
