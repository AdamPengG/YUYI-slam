from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .semantic_observer import MaskObservation


class TextRegionExtractor:
    def __init__(
        self,
        semantic_cfg: dict[str, Any],
        class_names: list[str],
        device: str,
        topk_labels: int,
    ) -> None:
        from ovo.entities.clip_generator import CLIPGenerator

        self.device = device
        self.class_names = list(class_names)
        self.topk_labels = max(int(topk_labels), 1)
        templates = semantic_cfg.get("classify_templates", ["This is a photo of a {}"])
        if isinstance(templates, str):
            templates = [templates]
        self.templates = list(templates)
        self.clip_generator = CLIPGenerator(semantic_cfg["clip"], device=device)
        self.embedding_variant = str(semantic_cfg["clip"].get("embed_type", "unknown"))
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

    def enrich_mask_observations(
        self,
        image_rgb: np.ndarray,
        mask_observations: list[MaskObservation],
    ) -> list[MaskObservation]:
        if not mask_observations:
            return []
        binary_masks = np.stack([obs.binary_mask.astype(bool, copy=False) for obs in mask_observations], axis=0)
        clip_embeds = self.extract_region_embeddings(image_rgb, binary_masks)
        if clip_embeds.numel() == 0:
            return mask_observations
        top_scores, top_indices = self.rank_embeddings(clip_embeds)
        enriched: list[MaskObservation] = []
        for idx, observation in enumerate(mask_observations):
            label_candidates = [
                self.class_names[int(class_idx)]
                for class_idx in top_indices[idx].detach().cpu().tolist()
                if 0 <= int(class_idx) < len(self.class_names)
            ]
            label_scores = [float(score) for score in top_scores[idx].detach().cpu().tolist()][: len(label_candidates)]
            enriched.append(
                MaskObservation(
                    mask_id=int(observation.mask_id),
                    binary_mask=observation.binary_mask,
                    bbox_xyxy=[int(v) for v in observation.bbox_xyxy],
                    semantic_label_candidates=label_candidates,
                    semantic_scores=label_scores,
                    semantic_embedding=clip_embeds[idx].detach().cpu().numpy().astype(np.float32, copy=True),
                    semantic_embedding_variant=self.embedding_variant,
                    yolo_label_candidates=list(observation.yolo_label_candidates or observation.semantic_label_candidates),
                    yolo_scores=[float(v) for v in (observation.yolo_scores or observation.semantic_scores)],
                    detection_score=float(observation.detection_score),
                    observation_kind=str(observation.observation_kind),
                    view_quality=float(observation.view_quality),
                )
            )
        return enriched

    def rank_descriptor(self, descriptor: np.ndarray | list[float], topk: int | None = None) -> list[tuple[str, float]]:
        if descriptor is None:
            return []
        descriptor_tensor = torch.as_tensor(np.asarray(descriptor, dtype=np.float32), device=self.device)
        if descriptor_tensor.ndim == 1:
            descriptor_tensor = descriptor_tensor.unsqueeze(0)
        expected_dim = int(self.text_embeddings.shape[1]) if self.text_embeddings.ndim == 2 and self.text_embeddings.shape[0] > 0 else int(descriptor_tensor.shape[-1])
        if int(descriptor_tensor.shape[-1]) != expected_dim:
            return []
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
