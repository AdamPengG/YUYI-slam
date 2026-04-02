from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .semantic_observer import MaskObservation


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class _ClipDescriptorBackend:
    def __init__(
        self,
        clip_cfg: dict[str, Any],
        class_names: list[str],
        templates: list[str],
        device: str,
        topk_labels: int,
    ) -> None:
        from ovo.entities.clip_generator import CLIPGenerator

        self.device = device
        self.class_names = class_names
        self.topk_labels = max(int(topk_labels), 1)
        self.clip_generator = CLIPGenerator(clip_cfg, device=device)
        self.embedding_variant = str(clip_cfg.get("embed_type", "unknown"))
        self.templates = templates
        self.text_embeddings = self._build_text_embeddings()

    def _build_text_embeddings(self) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        for class_name in self.class_names:
            phrases = [template.format(class_name) for template in self.templates]
            embed = self.clip_generator.get_txt_embedding(phrases).mean(0, keepdim=True).float()
            embed = torch.nn.functional.normalize(embed, p=2, dim=-1)
            embeddings.append(embed.squeeze(0).cpu())
        if not embeddings:
            return torch.zeros((0, self.clip_generator.clip_dim), dtype=torch.float32)
        return torch.vstack(embeddings)

    def extract_region_embeddings(self, image_rgb: np.ndarray, binary_masks: np.ndarray) -> torch.Tensor:
        if binary_masks.size == 0:
            return torch.zeros((0, self.clip_generator.clip_dim), dtype=torch.float32, device=self.device)
        image_tensor = torch.from_numpy(image_rgb.transpose((2, 0, 1))).to(self.device)
        masks_tensor = torch.from_numpy(binary_masks.astype(np.uint8, copy=False)).to(self.device)
        return self.clip_generator.extract_clip(image_tensor, masks_tensor, return_all=False)

    def rank_embeddings(
        self,
        embeddings: torch.Tensor,
        topk: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.numel() == 0:
            return (
                torch.zeros((0, 0), dtype=torch.float32, device=embeddings.device),
                torch.zeros((0, 0), dtype=torch.long, device=embeddings.device),
            )
        similarities = self.clip_generator.get_similarity(
            self.text_embeddings.to(embeddings.device, dtype=embeddings.dtype),
            embeddings,
            *self.clip_generator.similarity_args,
        )
        k = max(1, min(int(topk or self.topk_labels), similarities.shape[1]))
        return torch.topk(similarities, k=k, dim=1)

    def rank_descriptor(self, descriptor: np.ndarray | list[float], topk: int | None = None) -> list[tuple[str, float]]:
        if descriptor is None:
            return []
        descriptor_tensor = torch.as_tensor(np.asarray(descriptor, dtype=np.float32), device=self.device)
        if descriptor_tensor.ndim == 1:
            descriptor_tensor = descriptor_tensor.unsqueeze(0)
        descriptor_tensor = torch.nn.functional.normalize(descriptor_tensor, p=2, dim=-1)
        top_scores, top_indices = self.rank_embeddings(descriptor_tensor, topk=topk)
        ranked: list[tuple[str, float]] = []
        for class_idx, score in zip(
            top_indices[0].detach().cpu().tolist(),
            top_scores[0].detach().cpu().tolist(),
            strict=False,
        ):
            if 0 <= int(class_idx) < len(self.class_names):
                ranked.append((self.class_names[int(class_idx)], float(score)))
        return ranked

    def classify_descriptor(self, descriptor: np.ndarray | list[float]) -> tuple[str, float]:
        ranked = self.rank_descriptor(descriptor, topk=1)
        if not ranked:
            return "unknown", 0.0
        return ranked[0]


class TextRegionPerceptionAdapter:
    def __init__(
        self,
        semantic_cfg: dict[str, Any],
        class_names: list[str],
        scene_name: str,
        device: str,
        topk_labels: int,
    ) -> None:
        from ovo.entities.mask_generator import MaskGenerator

        templates = semantic_cfg.get("classify_templates", ["This is a photo of a {}"])
        if isinstance(templates, str):
            templates = [templates]
        self.mask_generator = MaskGenerator(semantic_cfg["sam"], scene_name=scene_name, device=device)
        self.descriptor_backend = _ClipDescriptorBackend(
            clip_cfg=semantic_cfg["clip"],
            class_names=class_names,
            templates=list(templates),
            device=device,
            topk_labels=topk_labels,
        )

    def build_mask_observations(self, image_rgb: np.ndarray, frame_id: int) -> list[MaskObservation]:
        seg_map, binary_maps = self.mask_generator.get_masks(image_rgb, frame_id)
        if seg_map.numel() == 0 or binary_maps.numel() == 0:
            return []

        clip_embeds = self.descriptor_backend.extract_region_embeddings(
            image_rgb, binary_maps.detach().cpu().numpy().astype(bool, copy=False)
        )
        if clip_embeds.numel() == 0:
            return []

        top_scores, top_indices = self.descriptor_backend.rank_embeddings(clip_embeds)
        binary_maps_np = binary_maps.detach().cpu().numpy().astype(bool, copy=False)

        observations: list[MaskObservation] = []
        for mask_idx, binary_mask in enumerate(binary_maps_np):
            ys, xs = np.nonzero(binary_mask)
            if xs.size == 0:
                continue
            bbox_xyxy = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            label_candidates = [
                self.descriptor_backend.class_names[int(class_idx)]
                for class_idx in top_indices[mask_idx].detach().cpu().tolist()
            ]
            label_scores = [float(score) for score in top_scores[mask_idx].detach().cpu().tolist()]
            observations.append(
                MaskObservation(
                    mask_id=int(mask_idx),
                    binary_mask=binary_mask,
                    bbox_xyxy=bbox_xyxy,
                    semantic_label_candidates=label_candidates,
                    semantic_scores=label_scores,
                    semantic_embedding=clip_embeds[mask_idx].detach().cpu().numpy().astype(np.float32, copy=True),
                    semantic_embedding_variant=self.descriptor_backend.embedding_variant,
                )
            )
        return observations

    def rank_descriptor(self, descriptor: np.ndarray | list[float], topk: int | None = None) -> list[tuple[str, float]]:
        return self.descriptor_backend.rank_descriptor(descriptor, topk=topk)

    def classify_descriptor(self, descriptor: np.ndarray | list[float]) -> tuple[str, float]:
        return self.descriptor_backend.classify_descriptor(descriptor)


class OVSAMAdapter:
    """Phase-1 OVSAM frontend.

    Keeps the current proposal source (SAM masks) and delegates region refinement /
    recognition to an external OVSAM helper runtime. Region embeddings for the
    downstream 3D observer are still computed with the current CLIP backend so the
    ObjectMemory path remains compatible.
    """

    def __init__(
        self,
        semantic_cfg: dict[str, Any],
        class_names: list[str],
        scene_name: str,
        device: str,
        topk_labels: int,
    ) -> None:
        from ovo.entities.mask_generator import MaskGenerator

        templates = semantic_cfg.get("classify_templates", ["This is a photo of a {}"])
        if isinstance(templates, str):
            templates = [templates]
        self.mask_generator = MaskGenerator(semantic_cfg["sam"], scene_name=scene_name, device=device)
        self.descriptor_backend = _ClipDescriptorBackend(
            clip_cfg=semantic_cfg["clip"],
            class_names=class_names,
            templates=list(templates),
            device=device,
            topk_labels=topk_labels,
        )
        self.scene_name = scene_name
        self.topk_labels = max(int(topk_labels), 1)

        ovsam_cfg = dict(semantic_cfg.get("ovsam", {}))
        self.ovsam_cfg = ovsam_cfg
        self.repo_root = Path(ovsam_cfg.get("repo_root", _repo_root() / "reference" / "OVSAM_official")).expanduser()
        self.weights_root = Path(ovsam_cfg.get("weights_root", self.repo_root / "models")).expanduser()
        self.mode = str(ovsam_cfg.get("mode", "prompt_region"))
        self.use_mask_bbox_as_prompt = bool(ovsam_cfg.get("use_mask_bbox_as_prompt", True))
        self.score_thresh = float(ovsam_cfg.get("score_thresh", 0.10))
        self.output_topk = max(int(ovsam_cfg.get("topk", self.topk_labels)), 1)
        self.use_refined_masks = bool(ovsam_cfg.get("use_refined_masks", True))
        self.cuda_visible_devices = str(ovsam_cfg.get("cuda_visible_devices", "1")).strip()
        self.helper_device = str(ovsam_cfg.get("device", "cuda")).strip()
        self.helper_script = Path(
            ovsam_cfg.get(
                "helper_script",
                _repo_root() / "src" / "onemap_semantic_mapper" / "scripts" / "run_ovsam_region_infer.py",
            )
        ).expanduser()
        self.conda_env = str(ovsam_cfg.get("conda_env", "ovsam_env")).strip()
        self.python_bin = str(ovsam_cfg.get("python_bin", "")).strip()
        self.config_path = str(
            ovsam_cfg.get(
                "config_path",
                self.repo_root / "seg" / "configs" / "ovsam" / "ovsam_coco_rn50x16_point.py",
            )
        )
        self.checkpoint_path = str(ovsam_cfg.get("checkpoint_path", "")).strip()

    def _build_python_command(self) -> list[str]:
        if self.python_bin:
            return [self.python_bin, str(self.helper_script)]
        conda_exe = shutil.which("conda") or "/home/peng/miniconda3/bin/conda"
        return [conda_exe, "run", "-n", self.conda_env, "python", str(self.helper_script)]

    def _run_helper(
        self,
        image_rgb: np.ndarray,
        proposals: list[dict[str, Any]],
        proposal_masks: np.ndarray,
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        if not self.helper_script.exists():
            raise RuntimeError(f"OVSAM helper script not found: {self.helper_script}")
        if not self.repo_root.exists():
            raise RuntimeError(f"OVSAM repo_root not found: {self.repo_root}")

        with tempfile.TemporaryDirectory(prefix="ovsam_frontend_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            image_path = temp_dir / "frame.png"
            proposals_path = temp_dir / "proposals.json"
            proposal_masks_path = temp_dir / "proposal_masks.npz"
            class_names_path = temp_dir / "class_names.json"
            output_json = temp_dir / "ovsam_result.json"
            output_masks = temp_dir / "ovsam_masks.npz"

            cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            proposals_path.write_text(json.dumps({"proposals": proposals}, ensure_ascii=False), encoding="utf-8")
            np.savez_compressed(proposal_masks_path, masks=proposal_masks.astype(bool, copy=False))
            class_names_path.write_text(
                json.dumps({"class_names": self.descriptor_backend.class_names}, ensure_ascii=False),
                encoding="utf-8",
            )

            cmd = self._build_python_command() + [
                "--repo-root",
                str(self.repo_root),
                "--weights-root",
                str(self.weights_root),
                "--config-path",
                str(self.config_path),
                "--checkpoint-path",
                str(self.checkpoint_path),
                "--image-path",
                str(image_path),
                "--proposals-json",
                str(proposals_path),
                "--proposal-masks-npz",
                str(proposal_masks_path),
                "--class-names-json",
                str(class_names_path),
                "--output-json",
                str(output_json),
                "--output-masks-npz",
                str(output_masks),
                "--mode",
                self.mode,
                "--topk",
                str(self.output_topk),
                "--score-thresh",
                str(self.score_thresh),
                "--device",
                self.helper_device,
            ]
            if self.use_mask_bbox_as_prompt:
                cmd.append("--use-mask-bbox-as-prompt")

            env = dict(os.environ)
            if self.cuda_visible_devices:
                env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
            completed = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if completed.returncode != 0:
                stderr = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(f"OVSAM helper failed: {stderr}")
            if not output_json.exists():
                raise RuntimeError("OVSAM helper finished without output json.")
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            refined_masks = None
            if output_masks.exists():
                with np.load(output_masks, allow_pickle=False) as data:
                    if "masks" in data:
                        refined_masks = data["masks"].astype(bool, copy=False)
            return payload, refined_masks

    def build_mask_observations(self, image_rgb: np.ndarray, frame_id: int) -> list[MaskObservation]:
        seg_map, binary_maps = self.mask_generator.get_masks(image_rgb, frame_id)
        if seg_map.numel() == 0 or binary_maps.numel() == 0:
            return []

        proposal_masks = binary_maps.detach().cpu().numpy().astype(bool, copy=False)
        proposals: list[dict[str, Any]] = []
        for mask_idx, binary_mask in enumerate(proposal_masks):
            ys, xs = np.nonzero(binary_mask)
            if xs.size == 0:
                continue
            bbox_xyxy = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            centroid_xy = [float(xs.mean()), float(ys.mean())]
            proposals.append(
                {
                    "proposal_id": int(mask_idx),
                    "mask_id": int(mask_idx),
                    "bbox_xyxy": bbox_xyxy,
                    "centroid_xy": centroid_xy,
                    "area": int(binary_mask.sum()),
                }
            )
        if not proposals:
            return []

        helper_payload, refined_masks = self._run_helper(image_rgb, proposals, proposal_masks)
        results = helper_payload.get("results", [])
        if not isinstance(results, list) or not results:
            return []

        masks_for_output = proposal_masks
        if self.use_refined_masks and refined_masks is not None and refined_masks.shape[0] == len(proposal_masks):
            masks_for_output = refined_masks

        clip_embeds = self.descriptor_backend.extract_region_embeddings(image_rgb, masks_for_output)
        if clip_embeds.numel() == 0:
            return []

        observations: list[MaskObservation] = []
        for item in results:
            proposal_id = int(item.get("proposal_id", item.get("mask_id", -1)))
            if proposal_id < 0 or proposal_id >= masks_for_output.shape[0]:
                continue
            binary_mask = masks_for_output[proposal_id]
            ys, xs = np.nonzero(binary_mask)
            if xs.size == 0:
                continue
            bbox_xyxy = item.get("bbox_xyxy") or [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            label_candidates = [str(name) for name in item.get("semantic_label_candidates", [])][: self.output_topk]
            label_scores = [float(score) for score in item.get("semantic_scores", [])][: len(label_candidates)]
            if not label_candidates:
                continue
            observations.append(
                MaskObservation(
                    mask_id=int(item.get("mask_id", proposal_id)),
                    binary_mask=binary_mask,
                    bbox_xyxy=[int(v) for v in bbox_xyxy],
                    semantic_label_candidates=label_candidates,
                    semantic_scores=label_scores,
                    semantic_embedding=clip_embeds[proposal_id].detach().cpu().numpy().astype(np.float32, copy=True),
                    semantic_embedding_variant="ovsam",
                )
            )
        return observations

    def rank_descriptor(self, descriptor: np.ndarray | list[float], topk: int | None = None) -> list[tuple[str, float]]:
        return self.descriptor_backend.rank_descriptor(descriptor, topk=topk)

    def classify_descriptor(self, descriptor: np.ndarray | list[float]) -> tuple[str, float]:
        return self.descriptor_backend.classify_descriptor(descriptor)
