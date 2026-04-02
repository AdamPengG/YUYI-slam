from __future__ import annotations

import argparse
import colorsys
import json
import os
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

from .data_types import KeyframePacket, LocalCloudPacket, SensorConfig
from .io.keyframe_manifest import load_keyframe_packets
from .io.local_cloud_manifest import load_local_cloud_packets
from .io.observation_manifest import append_observation_link
from .io.sensor_config import load_sensor_config
from .object_memory import ObjectMemoryStore
from .ovsam_adapter import OVSAMAdapter, TextRegionPerceptionAdapter
from .yoloe26x_adapter import YOLOE26XAdapter
from .semantic_observer import MaskObservation, SemanticObserver
from .stuff_region_extractor import StuffRegionExtractor
from .textregion_extractor import TextRegionExtractor
from .visibility_projector import VisibilityProjector
from .voxel_node_map import VoxelNodeMap


STRUCTURAL_LABELS = {
    "wall",
    "floor",
    "ceiling",
    "door",
    "window",
    "blinds",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online semantic observer driven by FAST-LIVO2 keyframe/local-cloud manifests."
    )
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ovo-root", required=True)
    parser.add_argument("--ovo-config", default="data/working/configs/ovo_livo2_vanilla.yaml")
    parser.add_argument("--dataset-name", default="Replica")
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--poll-period-sec", type=float, default=5.0)
    parser.add_argument("--min-keyframes", type=int, default=5)
    parser.add_argument("--process-every-new-keyframes", type=int, default=1)
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--resume-if-exists", action="store_true")
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk-labels", type=int, default=3)
    parser.add_argument("--class-set", choices=["full", "reduced"], default="full")
    parser.add_argument("--visibility-depth-tolerance-m", type=float, default=0.03)
    parser.add_argument("--snapshot-voxel-size-m", type=float, default=0.0)
    parser.add_argument("--snapshot-max-points", type=int, default=2000000)
    parser.add_argument("--snapshot-include-unassigned-points", type=int, default=1)
    parser.add_argument("--export-stale-objects", type=int, default=1)
    parser.add_argument("--online-cleanup-object-points", type=int, default=0)
    parser.add_argument("--min-observation-points", type=int, default=12)
    parser.add_argument("--min-mask-area", type=int, default=64)
    parser.add_argument("--mask-erosion-px", type=int, default=2)
    parser.add_argument("--merge-centroid-radius-m", type=float, default=0.75)
    parser.add_argument("--observer-abstain-margin", type=float, default=0.15)
    parser.add_argument("--observer-min-binding-score", type=float, default=0.45)
    parser.add_argument("--semantic-keyframe-max-gap", type=int, default=2)
    parser.add_argument("--semantic-novelty-thresh", type=float, default=0.08)
    parser.add_argument("--tentative-min-obs", type=int, default=2)
    parser.add_argument("--geometry-voxel-size-m", type=float, default=0.02)
    parser.add_argument("--geometry-truncation-m", type=float, default=0.08)
    parser.add_argument("--geometry-surface-band-m", type=float, default=0.03)
    parser.add_argument("--support-view-refinement-enabled", type=int, default=1)
    parser.add_argument("--support-view-max-frames", type=int, default=2)
    parser.add_argument("--support-view-min-translation-m", type=float, default=0.12)
    parser.add_argument("--support-view-max-translation-m", type=float, default=2.0)
    parser.add_argument("--support-view-max-time-gap-sec", type=float, default=8.0)
    parser.add_argument("--support-view-min-overlap-points", type=int, default=8)
    parser.add_argument("--support-view-min-descriptor-similarity", type=float, default=0.18)
    parser.add_argument("--direct-semantic-only", type=int, default=1)
    parser.add_argument("--direct-instance-cluster-radius-m", type=float, default=0.55)
    return parser.parse_args()


class OnlineSemanticAdapter:
    def __init__(
        self,
        semantic_cfg: dict[str, Any],
        class_names: list[str],
        scene_name: str,
        device: str,
        topk_labels: int,
    ) -> None:
        from ovo.entities.clip_generator import CLIPGenerator
        from ovo.entities.mask_generator import MaskGenerator

        self.device = device
        self.class_names = class_names
        self.topk_labels = max(int(topk_labels), 1)
        self.templates = semantic_cfg.get("classify_templates", ["This is a photo of a {}"])
        if isinstance(self.templates, str):
            self.templates = [self.templates]
        self.mask_generator = MaskGenerator(semantic_cfg["sam"], scene_name=scene_name, device=device)
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

    def build_mask_observations(self, image_rgb: np.ndarray, frame_id: int) -> list[MaskObservation]:
        seg_map, binary_maps = self.mask_generator.get_masks(image_rgb, frame_id)
        if seg_map.numel() == 0 or binary_maps.numel() == 0:
            return []

        image_tensor = torch.from_numpy(image_rgb.transpose((2, 0, 1))).to(self.device)
        clip_embeds = self.clip_generator.extract_clip(
            image_tensor, binary_maps, return_all=False
        )
        if clip_embeds.numel() == 0:
            return []

        similarities = self.clip_generator.get_similarity(
            self.text_embeddings.to(clip_embeds.device, dtype=clip_embeds.dtype),
            clip_embeds.to(clip_embeds.device),
            *self.clip_generator.similarity_args,
        )
        topk = min(self.topk_labels, similarities.shape[1])
        top_scores, top_indices = torch.topk(similarities, k=topk, dim=1)
        binary_maps_np = binary_maps.detach().cpu().numpy().astype(bool, copy=False)

        observations: list[MaskObservation] = []
        for mask_idx, binary_mask in enumerate(binary_maps_np):
            ys, xs = np.nonzero(binary_mask)
            if xs.size == 0:
                continue
            bbox_xyxy = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            label_candidates = [
                self.class_names[int(class_idx)] for class_idx in top_indices[mask_idx].detach().cpu().tolist()
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
                    semantic_embedding_variant=self.embedding_variant,
                )
            )
        return observations

    def rank_descriptor(self, descriptor: np.ndarray | list[float], topk: int | None = None) -> list[tuple[str, float]]:
        if descriptor is None:
            return []
        descriptor_tensor = torch.as_tensor(np.asarray(descriptor, dtype=np.float32), device=self.device)
        if descriptor_tensor.ndim == 1:
            descriptor_tensor = descriptor_tensor.unsqueeze(0)
        descriptor_tensor = torch.nn.functional.normalize(descriptor_tensor, p=2, dim=-1)
        similarities = self.clip_generator.get_similarity(
            self.text_embeddings.to(descriptor_tensor.device, dtype=descriptor_tensor.dtype),
            descriptor_tensor,
            *self.clip_generator.similarity_args,
        )
        if similarities.numel() == 0:
            return []
        k = int(topk or self.topk_labels)
        k = max(1, min(k, similarities.shape[1]))
        top_scores, top_indices = torch.topk(similarities, k=k, dim=1)
        ranked: list[tuple[str, float]] = []
        for class_idx, score in zip(
            top_indices[0].detach().cpu().tolist(),
            top_scores[0].detach().cpu().tolist(),
        ):
            if 0 <= int(class_idx) < len(self.class_names):
                ranked.append((self.class_names[int(class_idx)], float(score)))
        return ranked

    def classify_descriptor(self, descriptor: np.ndarray | list[float]) -> tuple[str, float]:
        ranked = self.rank_descriptor(descriptor, topk=1)
        if not ranked:
            return "unknown", 0.0
        return ranked[0]


class OnlineSemanticObserverRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.scene_dir = Path(args.scene_dir).expanduser().resolve()
        self.export_dir = Path(args.export_dir).expanduser().resolve()
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.ovo_root = Path(args.ovo_root).expanduser().resolve()
        self.scene_name = args.scene_name or self.scene_dir.name
        os.chdir(self.ovo_root)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / "observer_state.json"
        self.status_path = self.output_dir / "online_status.json"
        self.object_memory_path = self.output_dir / "object_memory.json"
        self.snapshot_path = self.output_dir / "semantic_snapshot.npz"
        self.snapshot_summary_path = self.output_dir / "semantic_snapshot_summary.txt"
        self.observation_manifest_path = self.export_dir / "observation_links.jsonl"
        self.per_mask_vote_debug_path = self.export_dir / "per_mask_vote_debug.jsonl"
        self.per_object_support_history_path = self.export_dir / "per_object_support_history.jsonl"
        self.pending_object_debug_path = self.export_dir / "pending_object_debug.jsonl"
        self.support_view_refinement_debug_path = self.export_dir / "support_view_refinement_debug.jsonl"
        self.keyframe_schedule_debug_path = self.export_dir / "keyframe_schedule_debug.jsonl"
        self.visibility_projector_debug_path = self.output_dir / "visibility_projector_debug.npz"
        self.voxel_map_path = self.output_dir / "voxel_node_map.npz"
        self.keyframe_lookup: dict[int, KeyframePacket] = {}

        self.processed_keyframe_ids: set[int] = set()
        self.total_observations = 0
        self.last_snapshot_processed = 0
        self.last_semantic_keyframe_id: int | None = None
        self.cloud_cache: dict[str, np.ndarray] = {}
        self.local_cloud_lookup: dict[str, LocalCloudPacket] = {}
        self.direct_semantic_votes: dict[str, dict[int, dict[str, float]]] = {}
        self.direct_instance_token_to_label: dict[str, str] = {}
        self.direct_semantic_only = bool(args.direct_semantic_only)
        self.direct_instance_cluster_radius_m = max(float(args.direct_instance_cluster_radius_m), 0.05)

        if args.clear_output and not args.resume_if_exists:
            self._clear_runtime_outputs()

        if args.resume_if_exists:
            self._load_state()

        semantic_cfg = self._load_ovo_config()
        self.class_names = self._load_classes(semantic_cfg)
        self.sensor_config = load_sensor_config(self.scene_dir / "sensor_config.yaml")
        self.projector = VisibilityProjector(depth_tolerance_m=args.visibility_depth_tolerance_m)
        self.observer = SemanticObserver(
            min_mask_area=args.min_mask_area,
            min_hit_points=args.min_observation_points,
            mask_erosion_px=args.mask_erosion_px,
            abstain_margin=args.observer_abstain_margin,
            min_binding_score=args.observer_min_binding_score,
        )
        self.object_memory = self._load_object_memory()
        self.voxel_map = self._load_voxel_map()
        self.semantic_frontend = str(semantic_cfg.get("frontend", "textregion_pe")).strip().lower()
        self.proposal_adapter = None
        self.descriptor_extractor = None
        self.semantic_ranker = None
        self.stuff_extractor = None
        self.object_only_mode = False
        if self.semantic_frontend == "ovsam":
            self.semantic_adapter = OVSAMAdapter(
                semantic_cfg=semantic_cfg,
                class_names=self.class_names,
                scene_name=self.scene_name,
                device=args.device,
                topk_labels=args.topk_labels,
            )
            self.semantic_ranker = self.semantic_adapter
        elif self.semantic_frontend == "yoloe26x":
            self.proposal_adapter = YOLOE26XAdapter(
                semantic_cfg=semantic_cfg,
                class_names=self.class_names,
                scene_name=self.scene_name,
                device=args.device,
                topk_labels=args.topk_labels,
            )
            self.descriptor_extractor = TextRegionExtractor(
                semantic_cfg=semantic_cfg,
                class_names=self.class_names,
                device=args.device,
                topk_labels=args.topk_labels,
            )
            self.object_only_mode = True
            self.stuff_extractor = None
            self.semantic_adapter = self.proposal_adapter
            self.semantic_ranker = self.descriptor_extractor
            self.semantic_frontend = "yoloe26x_textregion_pe"
        else:
            self.semantic_frontend = "textregion_pe"
            self.semantic_adapter = TextRegionPerceptionAdapter(
                semantic_cfg=semantic_cfg,
                class_names=self.class_names,
                scene_name=self.scene_name,
                device=args.device,
                topk_labels=args.topk_labels,
            )
            self.semantic_ranker = self.semantic_adapter
        self.support_view_refinement_enabled = bool(args.support_view_refinement_enabled)
        self.support_view_max_frames = max(int(args.support_view_max_frames), 0)
        self.support_view_min_translation_m = max(float(args.support_view_min_translation_m), 0.0)
        self.support_view_max_translation_m = max(float(args.support_view_max_translation_m), self.support_view_min_translation_m)
        self.support_view_max_time_gap_sec = max(float(args.support_view_max_time_gap_sec), 0.0)
        self.support_view_min_overlap_points = max(int(args.support_view_min_overlap_points), 1)
        self.support_view_min_descriptor_similarity = float(args.support_view_min_descriptor_similarity)
        self.export_stale_objects = bool(args.export_stale_objects)
        self.online_cleanup_object_points = bool(args.online_cleanup_object_points)
        self.snapshot_include_unassigned_points = bool(args.snapshot_include_unassigned_points)

    def _clear_runtime_outputs(self) -> None:
        for path in [
            self.state_path,
            self.status_path,
            self.object_memory_path,
            self.snapshot_path,
            self.snapshot_summary_path,
            self.observation_manifest_path,
            self.per_mask_vote_debug_path,
            self.per_object_support_history_path,
            self.pending_object_debug_path,
            self.support_view_refinement_debug_path,
            self.keyframe_schedule_debug_path,
            self.visibility_projector_debug_path,
            self.voxel_map_path,
        ]:
            if path.exists():
                path.unlink()

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.processed_keyframe_ids = {int(v) for v in payload.get("processed_keyframe_ids", [])}
        self.total_observations = int(payload.get("total_observations", 0))
        self.last_snapshot_processed = int(payload.get("last_snapshot_processed", 0))
        self.last_semantic_keyframe_id = (
            None if payload.get("last_semantic_keyframe_id") is None else int(payload.get("last_semantic_keyframe_id"))
        )

    def _save_state(self) -> None:
        payload = {
            "processed_keyframe_ids": sorted(self.processed_keyframe_ids),
            "total_observations": int(self.total_observations),
            "last_snapshot_processed": int(self.last_snapshot_processed),
            "last_semantic_keyframe_id": None if self.last_semantic_keyframe_id is None else int(self.last_semantic_keyframe_id),
            "semantic_frontend": self.semantic_frontend,
        }
        self.state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _load_object_memory(self) -> ObjectMemoryStore:
        if self.args.resume_if_exists and self.object_memory_path.exists():
            payload = json.loads(self.object_memory_path.read_text(encoding="utf-8"))
            return ObjectMemoryStore.from_dict(payload)
        return ObjectMemoryStore(
            min_points_per_observation=self.args.min_observation_points,
            min_mask_area=self.args.min_mask_area,
            merge_centroid_radius_m=self.args.merge_centroid_radius_m,
        )

    def _load_voxel_map(self) -> VoxelNodeMap:
        if self.args.resume_if_exists and self.voxel_map_path.exists():
            return VoxelNodeMap.load(self.voxel_map_path)
        return VoxelNodeMap(
            voxel_size_m=self.args.geometry_voxel_size_m,
            truncation_m=self.args.geometry_truncation_m,
            surface_band_m=self.args.geometry_surface_band_m,
        )

    def _save_object_memory(self) -> None:
        self.object_memory_path.write_text(
            json.dumps(self.object_memory.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _load_ovo_config(self) -> dict[str, Any]:
        from ovo.utils import io_utils

        ovo_config_path = Path(self.args.ovo_config)
        if not ovo_config_path.is_absolute():
            ovo_config_path = self.ovo_root / ovo_config_path
        config = io_utils.load_config(str(ovo_config_path))
        return dict(config["semantic"])

    def _load_classes(self, semantic_cfg: dict[str, Any] | None = None) -> list[str]:
        semantic_cfg = semantic_cfg or {}
        frontend = str(semantic_cfg.get("frontend", "textregion_pe")).strip().lower()
        if frontend == "yoloe26x":
            yoloe_cfg = dict(semantic_cfg.get("yoloe", {}))
            configured_classes = [str(name).strip() for name in yoloe_cfg.get("classes", [])]
            configured_classes = [name for name in configured_classes if name]
            if configured_classes:
                return configured_classes
        eval_info_path = self.ovo_root / "data" / "working" / "configs" / self.args.dataset_name / "eval_info.yaml"
        with eval_info_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if self.args.class_set == "reduced":
            classes = list(payload.get("class_names_reduced", payload["class_names"]))
        else:
            classes = [str(name) for name in payload["class_names"]]
            classes = [name for name in classes if name and name != "0"]
        return classes

    def _build_mask_observations(
        self,
        image_rgb: np.ndarray,
        depth_m: np.ndarray | None,
        frame_id: int,
        semantic_keyframe: bool = True,
    ) -> list[MaskObservation]:
        mask_observations = self.semantic_adapter.build_mask_observations(image_rgb, frame_id)
        if self.stuff_extractor is not None:
            thing_masks = [obs.binary_mask for obs in mask_observations if str(obs.observation_kind) == "thing"]
            mask_observations.extend(
                self.stuff_extractor.build_mask_observations(
                    image_rgb=image_rgb,
                    depth_m=depth_m,
                    thing_masks=thing_masks,
                    start_mask_id=(int(frame_id) * 1000) + 10_000,
                )
            )
        if self.descriptor_extractor is not None and semantic_keyframe and mask_observations:
            mask_observations = self.descriptor_extractor.enrich_mask_observations(image_rgb, mask_observations)
        if self.object_only_mode:
            mask_observations = [obs for obs in mask_observations if str(obs.observation_kind) == "thing"]
        return mask_observations

    def _semantic_node_pressure(self) -> float:
        if not self.object_memory.objects:
            return 1.0
        pending_like = 0
        for obj in self.object_memory.objects.values():
            if obj.state == "pending":
                pending_like += 1
                continue
            if float(getattr(obj, "unknown_score", 1.0)) >= 0.45:
                pending_like += 1
                continue
            if float(getattr(obj, "reject_score", 0.0)) >= 0.55:
                pending_like += 1
        return min(1.0, float(pending_like) / max(len(self.object_memory.objects), 1))

    def _is_semantic_keyframe(self, packet: KeyframePacket) -> tuple[bool, dict[str, float | int | str]]:
        reasons = {str(reason) for reason in packet.selection_reasons}
        pressure = self._semantic_node_pressure()
        if self.last_semantic_keyframe_id is None:
            return True, {"pressure": pressure, "gap": 0, "reason": "bootstrap"}
        keyframe_gap = int(packet.keyframe_id) - int(self.last_semantic_keyframe_id)
        novelty = pressure
        if "semantic_trigger" in reasons:
            novelty = max(novelty, 1.0)
        if "coverage_novelty" in reasons:
            novelty = max(novelty, 0.25)
        if keyframe_gap >= int(self.args.semantic_keyframe_max_gap):
            return True, {"pressure": pressure, "gap": keyframe_gap, "reason": "max_gap"}
        if novelty >= float(self.args.semantic_novelty_thresh):
            return True, {"pressure": pressure, "gap": keyframe_gap, "reason": "node_pressure"}
        return False, {"pressure": pressure, "gap": keyframe_gap, "reason": "geometry_only"}

    def run(self) -> None:
        while True:
            processed = self.process_once()
            if self.args.run_once:
                return
            if not processed:
                time.sleep(self.args.poll_period_sec)

    def process_once(self) -> bool:
        keyframes = load_keyframe_packets(self.export_dir / "keyframe_packets.jsonl")
        local_clouds = {
            packet.local_cloud_id: packet
            for packet in load_local_cloud_packets(self.export_dir / "local_cloud_packets.jsonl")
        }
        self.local_cloud_lookup = local_clouds
        self.keyframe_lookup = {packet.keyframe_id: packet for packet in keyframes}

        total_keyframes = len(keyframes)
        if total_keyframes < self.args.min_keyframes:
            self._write_status(total_keyframes, "waiting_for_keyframes", 0)
            return False

        new_packets = [packet for packet in keyframes if packet.keyframe_id not in self.processed_keyframe_ids]
        if not new_packets:
            self._write_status(total_keyframes, "idle", 0)
            return False

        processed_now = 0
        for packet in new_packets:
            local_cloud = local_clouds.get(packet.local_cloud_ref)
            if local_cloud is None:
                continue
            if not Path(packet.rgb_path).exists():
                continue
            if packet.depth_path is not None and not Path(packet.depth_path).exists():
                continue

            self._process_keyframe(packet, local_cloud)
            self.processed_keyframe_ids.add(packet.keyframe_id)
            processed_now += 1

            processed_total = len(self.processed_keyframe_ids)
            if processed_total - self.last_snapshot_processed >= self.args.process_every_new_keyframes:
                self._save_object_memory()
                self._save_state()
                self._export_snapshot()
                self.last_snapshot_processed = processed_total
                self._save_state()
                self._write_status(total_keyframes, "processing", processed_now)

        if processed_now == 0:
            self._write_status(total_keyframes, "waiting_for_local_clouds", 0)
            return False

        self._save_object_memory()
        self._save_state()

        processed_total = len(self.processed_keyframe_ids)
        if processed_total - self.last_snapshot_processed >= self.args.process_every_new_keyframes:
            self._export_snapshot()
            self.last_snapshot_processed = processed_total
            self._save_state()

        self._write_status(total_keyframes, "updated", processed_now)
        return True

    def _process_keyframe(self, packet: KeyframePacket, local_cloud: LocalCloudPacket) -> None:
        image_rgb = self._load_image(packet.rgb_path)
        depth_m = self._load_depth(packet.depth_path)
        projection = self.projector.project(packet, local_cloud, self.sensor_config, depth_m)
        if projection.visible_point_indices.size == 0:
            self._integrate_geometry_keyframe(packet, local_cloud)
            return
        self._write_projection_debug(packet, local_cloud, projection)

        semantic_keyframe, schedule_meta = self._is_semantic_keyframe(packet)
        self._append_jsonl(
            self.keyframe_schedule_debug_path,
            {
                "keyframe_id": int(packet.keyframe_id),
                "stamp_sec": float(packet.stamp_sec),
                "is_geometry_keyframe": True,
                "is_semantic_keyframe": bool(semantic_keyframe),
                "schedule_reason": str(schedule_meta["reason"]),
                "semantic_pressure": float(schedule_meta["pressure"]),
                "semantic_gap": int(schedule_meta["gap"]),
                "selection_reasons": [str(v) for v in packet.selection_reasons],
            },
        )
        if semantic_keyframe:
            self.last_semantic_keyframe_id = int(packet.keyframe_id)

        mask_observations = self._build_mask_observations(
            image_rgb,
            depth_m,
            packet.keyframe_id,
            semantic_keyframe=semantic_keyframe,
        )
        if not mask_observations:
            self._integrate_geometry_keyframe(packet, local_cloud)
            return

        observations = self.observer.observe(
            keyframe_id=packet.keyframe_id,
            local_cloud_id=local_cloud.local_cloud_id,
            projection=projection,
            masks=mask_observations,
            existing_point_object_ids=self.object_memory.point_object_ids_for_local_cloud(
                local_cloud.local_cloud_id, local_cloud.point_count
            ),
            depth_m=depth_m,
        )
        if not observations:
            self._integrate_geometry_keyframe(packet, local_cloud)
            return

        if self.object_only_mode:
            self._accumulate_direct_semantic_labels(local_cloud, observations)

        for observation in observations:
            append_observation_link(self.observation_manifest_path, observation)
            self.total_observations += 1
        self._write_observation_debug(packet, local_cloud, observations)
        if self.object_only_mode and self.direct_semantic_only:
            self._integrate_geometry_keyframe(packet, local_cloud)
            return

        update_result = self.object_memory.update(
            observations=observations,
            keyframe_stamp_sec=packet.stamp_sec,
            keyframe_id=packet.keyframe_id,
            local_cloud=local_cloud,
            visible_point_indices=projection.visible_point_indices.tolist(),
        )
        self._refine_unstable_objects_with_support_views(packet, observations)
        self.object_memory.refresh_node_semantics(self.semantic_ranker.rank_descriptor, STRUCTURAL_LABELS)
        if not self.object_only_mode:
            self.object_memory.auto_split_nodes(min_component_points=max(self.args.tentative_min_obs * 4, 12))
        self._integrate_geometry_keyframe(packet, local_cloud)
        self._write_object_support_debug(packet, update_result.updated_object_ids + update_result.created_object_ids)
        self._write_pending_object_debug(packet)

    def _load_image(self, image_path: str) -> np.ndarray:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to load image {image_path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _load_depth(self, depth_path: str | None) -> np.ndarray | None:
        if depth_path is None:
            return None
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise RuntimeError(f"Failed to load depth {depth_path}")
        depth_scale = float(self.sensor_config.intrinsics.get("depth_scale", 1000.0))
        return depth_raw.astype(np.float32) / max(depth_scale, 1e-6)

    def _load_cloud_xyz(self, local_cloud: LocalCloudPacket) -> np.ndarray:
        cached = self.cloud_cache.get(local_cloud.local_cloud_id)
        if cached is not None:
            return cached
        with closing(np.load(Path(local_cloud.cloud_path))) as data:
            xyz = np.asarray(data["xyz"], dtype=np.float32)
        self.cloud_cache[local_cloud.local_cloud_id] = xyz
        return xyz

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_projection_debug(
        self,
        packet: KeyframePacket,
        local_cloud: LocalCloudPacket,
        projection,
    ) -> None:
        np.savez_compressed(
            self.visibility_projector_debug_path,
            keyframe_id=np.asarray(packet.keyframe_id, dtype=np.int32),
            local_cloud_id=np.asarray(local_cloud.local_cloud_id, dtype=object),
            visible_point_indices=projection.visible_point_indices.astype(np.int32, copy=False),
            projected_uv=projection.projected_uv.astype(np.int32, copy=False),
            projected_depth=projection.projected_depth.astype(np.float32, copy=False),
            depth_residual=projection.depth_residual.astype(np.float32, copy=False),
            zbuffer_rank=projection.zbuffer_rank.astype(np.int32, copy=False),
            distance_to_depth_edge=projection.distance_to_depth_edge.astype(np.float32, copy=False),
            visibility_score=projection.visibility_score.astype(np.float32, copy=False),
            quality_score=projection.quality_score.astype(np.float32, copy=False),
            point_id_buffer=np.asarray(
                projection.point_id_buffer if projection.point_id_buffer is not None else np.zeros((0, 0), dtype=np.int32),
                dtype=np.int32,
            ),
            zbuffer_depth_buffer=np.asarray(
                projection.zbuffer_depth_buffer
                if projection.zbuffer_depth_buffer is not None
                else np.zeros((0, 0), dtype=np.float32),
                dtype=np.float32,
            ),
        )

    def _write_observation_debug(
        self,
        packet: KeyframePacket,
        local_cloud: LocalCloudPacket,
        observations: list,
    ) -> None:
        for observation in observations:
            ranked_scores = sorted(
                observation.candidate_object_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            top1 = ranked_scores[0] if ranked_scores else None
            top2 = ranked_scores[1] if len(ranked_scores) > 1 else None
            self._append_jsonl(
                self.per_mask_vote_debug_path,
                {
                    "keyframe_id": int(packet.keyframe_id),
                    "stamp_sec": float(packet.stamp_sec),
                    "local_cloud_id": local_cloud.local_cloud_id,
                    "mask_id": int(observation.mask_id),
                    "mask_area": int(observation.mask_area),
                    "bbox_xyxy": [int(v) for v in observation.bbox_xyxy],
                    "semantic_label_candidates": list(observation.semantic_label_candidates),
                    "semantic_scores": [float(v) for v in observation.semantic_scores],
                    "yolo_label_candidates": list(observation.yolo_label_candidates),
                    "yolo_scores": [float(v) for v in observation.yolo_scores],
                    "detection_score": float(observation.detection_score),
                    "observation_kind": str(observation.observation_kind),
                    "vote_count": int(observation.vote_count),
                    "quality_score": float(observation.quality_score),
                    "visibility_score": float(observation.visibility_score),
                    "foreground_depth_median": observation.foreground_depth_median,
                    "foreground_depth_p10": observation.foreground_depth_p10,
                    "candidate_object_id": observation.candidate_object_id,
                    "candidate_object_scores": {
                        str(key): float(value) for key, value in observation.candidate_object_scores.items()
                    },
                    "top1_object": None if top1 is None else str(top1[0]),
                    "top1_score": None if top1 is None else float(top1[1]),
                    "top2_object": None if top2 is None else str(top2[0]),
                    "top2_score": None if top2 is None else float(top2[1]),
                    "abstained": bool(observation.abstained),
                    "abstain_reason": str(observation.abstain_reason),
                },
            )

    def _write_object_support_debug(self, packet: KeyframePacket, object_ids: list[str]) -> None:
        seen: set[str] = set()
        for object_id in object_ids:
            if object_id in seen:
                continue
            seen.add(object_id)
            obj = self.object_memory.objects.get(str(object_id))
            if obj is None:
                continue
            support_count = sum(len(indices) for _cloud_id, indices in obj.point_support_refs)
            self._append_jsonl(
                self.per_object_support_history_path,
                {
                    "keyframe_id": int(packet.keyframe_id),
                    "stamp_sec": float(packet.stamp_sec),
                    "object_id": str(object_id),
                    "state": str(obj.state),
                    "support_count": int(support_count),
                    "observation_count": int(obj.observation_count),
                    "stability_score": float(obj.stability_score),
                    "completeness_score": float(obj.completeness_score),
                    "negative_evidence_score": float(obj.negative_evidence_score),
                    "top_label": self._object_label(obj),
                    "node_type": str(getattr(obj, "node_type", "thing")),
                    "node_status": str(getattr(obj, "node_status", "tentative")),
                    "descriptor_label": self._object_descriptor_label(obj)[0],
                    "descriptor_conf": self._object_descriptor_label(obj)[1],
                    "descriptor_margin": self._object_descriptor_label(obj)[2],
                    "posterior": {str(k): float(v) for k, v in getattr(obj, "posterior", {}).items()},
                    "unknown_score": float(getattr(obj, "unknown_score", 1.0)),
                    "reject_score": float(getattr(obj, "reject_score", 0.0)),
                    "label_votes": {str(k): float(v) for k, v in obj.label_votes.items()},
                    "merge_reason": str(obj.merge_reason),
                    "label_compatibility_score": float(obj.label_compatibility_score),
                    "support_overlap_score": float(obj.support_overlap_score),
                },
            )

    def _write_pending_object_debug(self, packet: KeyframePacket) -> None:
        for object_id, obj in self.object_memory.objects.items():
            if obj.state != "pending":
                continue
            support_count = sum(len(indices) for _cloud_id, indices in obj.point_support_refs)
            self._append_jsonl(
                self.pending_object_debug_path,
                {
                    "keyframe_id": int(packet.keyframe_id),
                    "stamp_sec": float(packet.stamp_sec),
                    "object_id": str(object_id),
                    "support_count": int(support_count),
                    "observation_count": int(obj.observation_count),
                    "promotion_evidence": float(obj.promotion_evidence),
                    "negative_evidence_score": float(obj.negative_evidence_score),
                    "top_label": self._object_label(obj),
                    "node_type": str(getattr(obj, "node_type", "thing")),
                    "node_status": str(getattr(obj, "node_status", "tentative")),
                    "descriptor_label": self._object_descriptor_label(obj)[0],
                    "descriptor_conf": self._object_descriptor_label(obj)[1],
                    "descriptor_margin": self._object_descriptor_label(obj)[2],
                    "posterior": {str(k): float(v) for k, v in getattr(obj, "posterior", {}).items()},
                    "unknown_score": float(getattr(obj, "unknown_score", 1.0)),
                    "reject_score": float(getattr(obj, "reject_score", 0.0)),
                    "label_votes": {str(k): float(v) for k, v in obj.label_votes.items()},
                },
            )

    def _refine_unstable_objects_with_support_views(self, anchor_packet: KeyframePacket, observations: list) -> None:
        if not self.support_view_refinement_enabled or self.support_view_max_frames <= 0:
            return
        for observation in observations:
            object_id = observation.candidate_object_id
            if not object_id:
                continue
            obj = self.object_memory.objects.get(str(object_id))
            if obj is None:
                continue
            if not self._needs_support_view_refinement(observation, obj):
                continue
            support_packets = self._select_support_view_packets(anchor_packet)
            if not support_packets:
                continue
            for support_packet in support_packets:
                self._apply_support_view_refinement(anchor_packet, observation, obj, support_packet)

    def _needs_support_view_refinement(self, observation, obj) -> bool:
        if obj.state == "pending":
            return True
        descriptor_label, descriptor_conf, descriptor_margin = self._object_descriptor_label(obj)
        if descriptor_label == "unknown":
            return True
        if descriptor_conf < 0.08 or descriptor_margin < 0.01:
            return True
        if observation.abstained:
            return True
        if len(observation.semantic_scores) >= 2:
            if float(observation.semantic_scores[0]) - float(observation.semantic_scores[1]) < 0.03:
                return True
        support_count = sum(len(indices) for _local_cloud_id, indices in obj.point_support_refs)
        if support_count < max(self.args.min_observation_points * 2, 8):
            return True
        if obj.completeness_score < 0.15:
            return True
        return False

    def _select_support_view_packets(self, anchor_packet: KeyframePacket) -> list[KeyframePacket]:
        anchor_pose = np.asarray(anchor_packet.t_world_cam, dtype=np.float32).reshape(4, 4)
        anchor_translation = anchor_pose[:3, 3]
        candidates: list[tuple[float, float, KeyframePacket]] = []
        for keyframe_id in sorted(self.processed_keyframe_ids):
            if keyframe_id == anchor_packet.keyframe_id:
                continue
            packet = self.keyframe_lookup.get(int(keyframe_id))
            if packet is None:
                continue
            if packet.pose_alignment == "missing":
                continue
            if abs(float(packet.stamp_sec) - float(anchor_packet.stamp_sec)) > self.support_view_max_time_gap_sec:
                continue
            support_pose = np.asarray(packet.t_world_cam, dtype=np.float32).reshape(4, 4)
            translation = float(np.linalg.norm(support_pose[:3, 3] - anchor_translation))
            if translation < self.support_view_min_translation_m or translation > self.support_view_max_translation_m:
                continue
            time_gap = abs(float(packet.stamp_sec) - float(anchor_packet.stamp_sec))
            candidates.append((time_gap, translation, packet))
        candidates.sort(key=lambda item: (item[0], -item[1]))
        return [item[2] for item in candidates[: self.support_view_max_frames]]

    def _apply_support_view_refinement(self, anchor_packet: KeyframePacket, observation, obj, support_packet: KeyframePacket) -> None:
        support_cloud = self.local_cloud_lookup.get(support_packet.local_cloud_ref)
        if support_cloud is None:
            return
        image_rgb = self._load_image(support_packet.rgb_path)
        depth_m = self._load_depth(support_packet.depth_path)
        projection = self.projector.project(support_packet, support_cloud, self.sensor_config, depth_m)
        if projection.visible_point_indices.size == 0:
            return
        xyz = self._load_cloud_xyz(support_cloud)
        support_xyz = xyz[projection.visible_point_indices]
        center = np.asarray(obj.centroid_world, dtype=np.float32)
        bbox = np.asarray(obj.bbox_world, dtype=np.float32)
        radius = float(np.clip(np.linalg.norm(bbox[3:] - bbox[:3]) * 0.65, 0.20, 1.00))
        near_mask = np.linalg.norm(support_xyz - center[None, :], axis=1) <= radius
        if int(np.count_nonzero(near_mask)) < self.support_view_min_overlap_points:
            return
        mask_observations = self._build_mask_observations(image_rgb, depth_m, support_packet.keyframe_id)
        if not mask_observations:
            return
        near_uv = projection.projected_uv[near_mask]
        near_quality = projection.quality_score[near_mask]
        anchor_embedding = np.asarray(observation.semantic_embedding, dtype=np.float32) if observation.semantic_embedding else None
        best_mask = None
        best_score = None
        best_overlap = 0
        best_similarity = 0.0
        best_quality = 0.0
        for mask in mask_observations:
            hits = mask.binary_mask[near_uv[:, 1], near_uv[:, 0]]
            if hits.dtype != np.bool_:
                hits = hits.astype(bool)
            overlap_points = int(np.count_nonzero(hits))
            if overlap_points < self.support_view_min_overlap_points:
                continue
            support_quality = float(np.mean(near_quality[hits])) if overlap_points > 0 else 0.0
            similarity = 0.0
            if anchor_embedding is not None and mask.semantic_embedding is not None:
                anchor_vec = anchor_embedding.reshape(-1)
                mask_vec = np.asarray(mask.semantic_embedding, dtype=np.float32).reshape(-1)
                anchor_norm = float(np.linalg.norm(anchor_vec))
                mask_norm = float(np.linalg.norm(mask_vec))
                if anchor_norm > 1e-6 and mask_norm > 1e-6:
                    similarity = float(np.dot(anchor_vec / anchor_norm, mask_vec / mask_norm))
            if similarity < self.support_view_min_descriptor_similarity and overlap_points < (self.support_view_min_overlap_points * 2):
                continue
            label_overlap = 0.0
            anchor_labels = {str(v) for v in observation.semantic_label_candidates[:2]}
            support_labels = {str(v) for v in mask.semantic_label_candidates[:2]}
            if anchor_labels and support_labels and (anchor_labels & support_labels):
                label_overlap = 0.25
            score = float(overlap_points) + (support_quality * 4.0) + (similarity * 3.0) + label_overlap
            if best_score is None or score > best_score:
                best_score = score
                best_mask = mask
                best_overlap = overlap_points
                best_similarity = similarity
                best_quality = support_quality
        if best_mask is None:
            return
        registered = self.object_memory.register_support_view(
            object_id=str(obj.object_id),
            keyframe_id=int(support_packet.keyframe_id),
            stamp_sec=float(support_packet.stamp_sec),
            semantic_label_candidates=list(best_mask.semantic_label_candidates),
            semantic_scores=list(best_mask.semantic_scores),
            yolo_label_candidates=list(best_mask.yolo_label_candidates),
            yolo_scores=[float(v) for v in best_mask.yolo_scores],
            semantic_embedding=best_mask.semantic_embedding,
            quality_score=max(best_quality, 0.05),
        )
        if not registered:
            return
        self._append_jsonl(
            self.support_view_refinement_debug_path,
            {
                "anchor_keyframe_id": int(anchor_packet.keyframe_id),
                "support_keyframe_id": int(support_packet.keyframe_id),
                "object_id": str(obj.object_id),
                "object_state": str(obj.state),
                "overlap_points": int(best_overlap),
                "descriptor_similarity": float(best_similarity),
                "support_quality": float(best_quality),
                "support_labels": list(best_mask.semantic_label_candidates),
                "support_scores": [float(v) for v in best_mask.semantic_scores],
            },
        )

    def _export_snapshot(self) -> None:
        xyz_parts: list[np.ndarray] = []
        rgb_parts: list[np.ndarray] = []
        instance_parts: list[np.ndarray] = []
        class_parts: list[np.ndarray] = []
        instance_centers: list[np.ndarray] = []
        instance_bboxes: list[np.ndarray] = []
        instance_labels: list[str] = []
        instance_ids: list[int] = []
        instance_point_counts: list[int] = []
        instance_view_counts: list[int] = []

        class_lookup = {name: idx for idx, name in enumerate(self.class_names)}
        object_export_meta: dict[str, tuple[int, int, np.ndarray]] = {}

        for obj in self.object_memory.objects.values():
            if not self._should_export_object(obj):
                continue
            xyz = self._collect_object_points(obj.point_support_refs)
            if xyz.size == 0:
                continue
            label = self._object_label(obj)
            xyz = self._cleanup_snapshot_object_points(xyz, label)
            if xyz.size == 0:
                continue
            xyz = self._voxel_downsample(xyz, self.args.snapshot_voxel_size_m)
            if xyz.size == 0:
                continue

            object_index = len(instance_ids)
            class_id = int(class_lookup.get(label, -1))
            semantic_rgb = self._semantic_color(class_id)
            bbox_min = xyz.min(axis=0).astype(np.float32)
            bbox_max = xyz.max(axis=0).astype(np.float32)
            object_export_meta[str(obj.object_id)] = (object_index, class_id, semantic_rgb.copy())

            instance_centers.append(xyz.mean(axis=0).astype(np.float32))
            instance_bboxes.append(np.concatenate((bbox_min, bbox_max)).astype(np.float32))
            instance_labels.append(label)
            instance_ids.append(object_index)
            instance_point_counts.append(int(xyz.shape[0]))
            instance_view_counts.append(int(obj.observation_count))

        if not instance_ids and not self.snapshot_include_unassigned_points:
            return

        if self.object_only_mode:
            xyz, rgb, instance_id, class_id, direct_instance_labels = self._build_direct_semantic_snapshot()
            if xyz.size > 0:
                instance_labels = direct_instance_labels
                instance_ids = list(range(len(instance_labels)))
                instance_centers = []
                instance_bboxes = []
                instance_point_counts = []
                instance_view_counts = []
                for instance_index in range(len(instance_labels)):
                    mask = instance_id == int(instance_index)
                    if not np.any(mask):
                        instance_centers.append(np.zeros((3,), dtype=np.float32))
                        instance_bboxes.append(np.zeros((6,), dtype=np.float32))
                        instance_point_counts.append(0)
                        instance_view_counts.append(1)
                        continue
                    pts = xyz[mask]
                    instance_centers.append(pts.mean(axis=0).astype(np.float32))
                    instance_bboxes.append(
                        np.concatenate((pts.min(axis=0), pts.max(axis=0))).astype(np.float32)
                    )
                    instance_point_counts.append(int(pts.shape[0]))
                    instance_view_counts.append(1)
            else:
                xyz, rgb, instance_id, class_id = self._build_dense_snapshot_from_local_clouds(object_export_meta)
            if xyz.size == 0:
                xyz, rgb, instance_id, class_id = self._build_sparse_snapshot_from_objects(object_export_meta)
            if xyz.size == 0 and self.snapshot_include_unassigned_points:
                xyz, rgb, instance_id, class_id = self._build_voxel_snapshot(object_export_meta)
        else:
            xyz, rgb, instance_id, class_id = self._build_voxel_snapshot(object_export_meta)
            if xyz.size == 0:
                if self.snapshot_include_unassigned_points:
                    xyz, rgb, instance_id, class_id = self._build_dense_snapshot_from_local_clouds(object_export_meta)
                    if xyz.size == 0 and instance_ids:
                        xyz, rgb, instance_id, class_id = self._build_sparse_snapshot_from_objects(object_export_meta)
                else:
                    xyz, rgb, instance_id, class_id = self._build_sparse_snapshot_from_objects(object_export_meta)

        if xyz.size == 0:
            return

        if xyz.shape[0] > self.args.snapshot_max_points:
            step = int(np.ceil(xyz.shape[0] / self.args.snapshot_max_points))
            keep = np.arange(0, xyz.shape[0], step, dtype=np.int64)
            xyz = xyz[keep]
            rgb = rgb[keep]
            instance_id = instance_id[keep]
            class_id = class_id[keep]

        tmp_path = self.snapshot_path.with_name(self.snapshot_path.name + ".tmp.npz")
        np.savez_compressed(
            tmp_path,
            xyz=xyz,
            rgb=rgb,
            instance_id=instance_id,
            class_id=class_id,
            instance_ids=np.asarray(instance_ids, dtype=np.int32),
            instance_centers=np.asarray(instance_centers, dtype=np.float32),
            instance_bboxes=np.asarray(instance_bboxes, dtype=np.float32),
            instance_labels=np.asarray([str(label) for label in instance_labels], dtype=np.str_),
            instance_point_counts=np.asarray(instance_point_counts, dtype=np.int32),
            instance_view_counts=np.asarray(instance_view_counts, dtype=np.int32),
            class_names=np.asarray([str(name) for name in self.class_names], dtype=np.str_),
            classify_error=np.asarray("", dtype=np.str_),
            mode=np.asarray("semantic", dtype=np.str_),
        )
        tmp_path.replace(self.snapshot_path)
        self.voxel_map.save(self.voxel_map_path)

        lines = [
            f"num_points={xyz.shape[0]}",
            f"num_instances={len(instance_ids)}",
            f"processed_keyframes={len(self.processed_keyframe_ids)}",
            f"total_observations={self.total_observations}",
        ]
        for label, center, point_count, view_count in zip(
            instance_labels[:50],
            instance_centers[:50],
            instance_point_counts[:50],
            instance_view_counts[:50],
        ):
            lines.append(
                f"label={label}, points={point_count}, views={view_count}, "
                f"center={[round(float(v), 4) for v in center.tolist()]}"
            )
        self.snapshot_summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _build_voxel_snapshot(
        self,
        object_export_meta: dict[str, tuple[int, int, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.voxel_map.num_voxels() <= 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )
        xyz, _weight, parent_node_id, _parent_conf = self.voxel_map.top_assignment()
        if xyz.size == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )
        unknown_rgb = self._semantic_color(-1)
        rgb = np.tile(unknown_rgb[None, :], (xyz.shape[0], 1))
        instance_id = np.full((xyz.shape[0],), -1, dtype=np.int32)
        class_id = np.full((xyz.shape[0],), -1, dtype=np.int32)
        for idx, object_id in enumerate(parent_node_id.tolist()):
            export_meta = object_export_meta.get(str(object_id))
            if export_meta is None:
                continue
            object_index, export_class_id, semantic_rgb = export_meta
            instance_id[idx] = int(object_index)
            class_id[idx] = int(export_class_id)
            rgb[idx] = semantic_rgb
        if not self.snapshot_include_unassigned_points:
            keep = instance_id >= 0
            xyz = xyz[keep]
            rgb = rgb[keep]
            instance_id = instance_id[keep]
            class_id = class_id[keep]
        return xyz, rgb, instance_id, class_id

    def _build_sparse_snapshot_from_objects(
        self,
        object_export_meta: dict[str, tuple[int, int, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xyz_parts: list[np.ndarray] = []
        rgb_parts: list[np.ndarray] = []
        instance_parts: list[np.ndarray] = []
        class_parts: list[np.ndarray] = []
        for object_id, (object_index, class_id, semantic_rgb) in object_export_meta.items():
            obj = self.object_memory.objects.get(str(object_id))
            if obj is None:
                continue
            xyz = self._collect_object_points(obj.point_support_refs)
            if xyz.size == 0:
                continue
            label = self._object_label(obj)
            xyz = self._cleanup_snapshot_object_points(xyz, label)
            if xyz.size == 0:
                continue
            xyz = self._voxel_downsample(xyz, self.args.snapshot_voxel_size_m)
            if xyz.size == 0:
                continue
            xyz_parts.append(xyz)
            rgb_parts.append(np.tile(semantic_rgb[None, :], (xyz.shape[0], 1)))
            instance_parts.append(np.full((xyz.shape[0],), object_index, dtype=np.int32))
            class_parts.append(np.full((xyz.shape[0],), class_id, dtype=np.int32))
        if not xyz_parts:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )
        return (
            np.vstack(xyz_parts).astype(np.float32, copy=False),
            np.vstack(rgb_parts).astype(np.uint8, copy=False),
            np.concatenate(instance_parts),
            np.concatenate(class_parts),
        )

    def _accumulate_direct_semantic_labels(self, local_cloud: LocalCloudPacket, observations: list) -> None:
        xyz = self._load_cloud_xyz(local_cloud)
        if xyz.size == 0:
            return
        local_votes = self.direct_semantic_votes.setdefault(local_cloud.local_cloud_id, {})
        for observation in observations:
            label = "unknown"
            score = 1.0
            if observation.yolo_label_candidates:
                label = str(observation.yolo_label_candidates[0])
                if observation.yolo_scores:
                    score = float(observation.yolo_scores[0])
            elif observation.semantic_label_candidates:
                label = str(observation.semantic_label_candidates[0])
                if observation.semantic_scores:
                    score = float(observation.semantic_scores[0])
            if not label or label == "unknown" or label in STRUCTURAL_LABELS:
                continue
            instance_token = f"{label}|{local_cloud.local_cloud_id}|{int(observation.mask_id)}"
            self.direct_instance_token_to_label[instance_token] = str(label)

            support = np.asarray(sorted(set(int(v) for v in observation.point_indices)), dtype=np.int32)
            support = support[(support >= 0) & (support < xyz.shape[0])]
            if support.size == 0:
                continue
            support_xyz = xyz[support].astype(np.float32, copy=False)
            bbox_min = support_xyz.min(axis=0)
            bbox_max = support_xyz.max(axis=0)
            extent = np.maximum(bbox_max - bbox_min, 0.0)
            max_extent = float(np.max(extent)) if extent.size == 3 else 0.0
            bbox_margin = float(np.clip(max(max_extent * 1.25, 0.18), 0.18, 0.65))
            grow_radius_m = float(np.clip(max(max_extent * 2.0, 0.22), 0.22, 0.75))
            expanded_min = bbox_min - bbox_margin
            expanded_max = bbox_max + bbox_margin
            in_box = np.all((xyz >= expanded_min[None, :]) & (xyz <= expanded_max[None, :]), axis=1)
            candidate_indices = np.flatnonzero(in_box).astype(np.int32)
            if candidate_indices.size == 0:
                candidate_indices = support
            candidate_xyz = xyz[candidate_indices].astype(np.float32, copy=False)
            diff = candidate_xyz[:, None, :] - support_xyz[None, :, :]
            min_dist = np.linalg.norm(diff, axis=2).min(axis=1)
            grown = candidate_indices[min_dist <= grow_radius_m]
            if grown.size == 0:
                grown = support
            support_set = {int(v) for v in support.tolist()}
            for point_index in grown.tolist():
                votes = local_votes.setdefault(int(point_index), {})
                votes[instance_token] = votes.get(instance_token, 0.0) + (
                    max(score, 0.05) * (2.5 if int(point_index) in support_set else 1.0)
                )

    def _build_direct_semantic_snapshot(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        xyz_parts: list[np.ndarray] = []
        rgb_parts: list[np.ndarray] = []
        instance_parts: list[np.ndarray] = []
        class_parts: list[np.ndarray] = []
        unknown_rgb = self._semantic_color(-1)
        class_lookup = {name: idx for idx, name in enumerate(self.class_names)}
        per_point_tokens: list[str] = []

        for local_cloud_id, local_cloud in sorted(self.local_cloud_lookup.items()):
            vote_map = self.direct_semantic_votes.get(local_cloud_id, {})
            xyz = self._load_cloud_xyz(local_cloud)
            if xyz.size == 0:
                continue
            rgb = np.tile(unknown_rgb[None, :], (xyz.shape[0], 1))
            instance_id = np.full((xyz.shape[0],), -1, dtype=np.int32)
            class_id = np.full((xyz.shape[0],), -1, dtype=np.int32)
            for point_index, votes in vote_map.items():
                if point_index < 0 or point_index >= xyz.shape[0]:
                    continue
                if not votes:
                    continue
                token, _score = max(votes.items(), key=lambda item: item[1])
                label = str(self.direct_instance_token_to_label.get(str(token), "")).strip()
                if not label or label == "unknown" or label in STRUCTURAL_LABELS:
                    continue
                cls = int(class_lookup.get(label, -1))
                instance_id[int(point_index)] = -2
                class_id[int(point_index)] = cls
                rgb[int(point_index)] = self._semantic_color(cls)
            tokens_local = np.asarray([""] * xyz.shape[0], dtype=object)
            valid_local = class_id >= 0
            if np.any(valid_local):
                for point_index, votes in vote_map.items():
                    if point_index < 0 or point_index >= xyz.shape[0] or not votes:
                        continue
                    token, _score = max(votes.items(), key=lambda item: item[1])
                    label = str(self.direct_instance_token_to_label.get(str(token), "")).strip()
                    if not label or label == "unknown" or label in STRUCTURAL_LABELS:
                        continue
                    tokens_local[int(point_index)] = str(token)
            if not self.snapshot_include_unassigned_points:
                keep = class_id >= 0
                xyz = xyz[keep]
                rgb = rgb[keep]
                instance_id = instance_id[keep]
                class_id = class_id[keep]
                tokens_local = tokens_local[keep]
            xyz_parts.append(xyz.astype(np.float32, copy=False))
            rgb_parts.append(rgb.astype(np.uint8, copy=False))
            instance_parts.append(instance_id)
            class_parts.append(class_id)
            per_point_tokens.extend([str(v) for v in tokens_local.tolist()])

        if not xyz_parts:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                [],
            )
        xyz_all = np.vstack(xyz_parts).astype(np.float32, copy=False)
        rgb_all = np.vstack(rgb_parts).astype(np.uint8, copy=False)
        instance_id_all = np.concatenate(instance_parts)
        class_id_all = np.concatenate(class_parts)
        token_array = np.asarray(per_point_tokens, dtype=object)

        token_to_instance: dict[str, int] = {}
        instance_labels: list[str] = []
        valid_mask = class_id_all >= 0
        if np.any(valid_mask):
            for idx, token in enumerate(token_array.tolist()):
                if class_id_all[idx] < 0:
                    continue
                token = str(token).strip()
                if not token:
                    continue
                if token not in token_to_instance:
                    token_to_instance[token] = len(token_to_instance)
                    instance_labels.append(str(self.direct_instance_token_to_label.get(token, "unknown")))
                instance_id_all[idx] = int(token_to_instance[token])
        return (
            xyz_all,
            rgb_all,
            instance_id_all.astype(np.int32, copy=False),
            class_id_all.astype(np.int32, copy=False),
            [str(v) for v in instance_labels],
        )

    def _build_dense_snapshot_from_local_clouds(
        self,
        object_export_meta: dict[str, tuple[int, int, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xyz_parts: list[np.ndarray] = []
        rgb_parts: list[np.ndarray] = []
        instance_parts: list[np.ndarray] = []
        class_parts: list[np.ndarray] = []
        unknown_rgb = self._semantic_color(-1)

        for local_cloud_id, point_states in self.object_memory.point_assignments.items():
            local_cloud = self.local_cloud_lookup.get(local_cloud_id)
            if local_cloud is None:
                continue
            xyz = self._load_cloud_xyz(local_cloud)
            if xyz.size == 0:
                continue
            rgb = np.tile(unknown_rgb[None, :], (xyz.shape[0], 1))
            instance_id = np.full((xyz.shape[0],), -1, dtype=np.int32)
            class_id = np.full((xyz.shape[0],), -1, dtype=np.int32)
            for point_index, state in point_states.items():
                if point_index < 0 or point_index >= xyz.shape[0]:
                    continue
                object_id = state.get("current_object_id")
                current_score = float(state.get("current_score", 0.0))
                if not object_id or current_score < self.object_memory.point_keep_threshold:
                    continue
                export_meta = object_export_meta.get(str(object_id))
                if export_meta is None:
                    continue
                export_index, export_class_id, semantic_rgb = export_meta
                instance_id[int(point_index)] = int(export_index)
                class_id[int(point_index)] = int(export_class_id)
                rgb[int(point_index)] = semantic_rgb
            self._grow_local_object_labels(
                local_cloud_id=local_cloud_id,
                xyz=xyz,
                instance_id=instance_id,
                class_id=class_id,
                rgb=rgb,
                object_export_meta=object_export_meta,
            )
            xyz_parts.append(xyz.astype(np.float32, copy=False))
            rgb_parts.append(rgb.astype(np.uint8, copy=False))
            instance_parts.append(instance_id)
            class_parts.append(class_id)

        if not xyz_parts:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )
        return (
            np.vstack(xyz_parts).astype(np.float32, copy=False),
            np.vstack(rgb_parts).astype(np.uint8, copy=False),
            np.concatenate(instance_parts),
            np.concatenate(class_parts),
        )

    def _grow_local_object_labels(
        self,
        local_cloud_id: str,
        xyz: np.ndarray,
        instance_id: np.ndarray,
        class_id: np.ndarray,
        rgb: np.ndarray,
        object_export_meta: dict[str, tuple[int, int, np.ndarray]],
    ) -> None:
        if xyz.size == 0 or not object_export_meta:
            return
        if np.count_nonzero(instance_id >= 0) >= xyz.shape[0]:
            return

        point_votes: dict[int, dict[str, float]] = {}
        for object_id, obj in self.object_memory.objects.items():
            export_meta = object_export_meta.get(str(object_id))
            if export_meta is None:
                continue
            support_indices: list[int] = []
            for support_local_cloud_id, point_indices in obj.point_support_refs:
                if str(support_local_cloud_id) != str(local_cloud_id):
                    continue
                support_indices.extend(int(v) for v in point_indices)
            if not support_indices:
                continue
            valid = np.asarray(sorted(set(support_indices)), dtype=np.int32)
            valid = valid[(valid >= 0) & (valid < xyz.shape[0])]
            if valid.size == 0:
                continue

            support_xyz = xyz[valid].astype(np.float32, copy=False)
            support_set = {int(v) for v in valid.tolist()}
            bbox_min = support_xyz.min(axis=0)
            bbox_max = support_xyz.max(axis=0)
            extent = np.maximum(bbox_max - bbox_min, 0.0)
            max_extent = float(np.max(extent)) if extent.size == 3 else 0.0
            bbox_margin = float(np.clip(max(max_extent * 0.75, 0.10), 0.10, 0.45))
            grow_radius_m = float(np.clip(max(max_extent * 1.25, 0.12), 0.12, 0.55))
            expanded_min = bbox_min - bbox_margin
            expanded_max = bbox_max + bbox_margin
            in_box = np.all((xyz >= expanded_min[None, :]) & (xyz <= expanded_max[None, :]), axis=1)
            candidate_indices = np.flatnonzero(in_box).astype(np.int32)
            if candidate_indices.size == 0:
                candidate_indices = valid

            candidate_xyz = xyz[candidate_indices].astype(np.float32, copy=False)
            diff = candidate_xyz[:, None, :] - support_xyz[None, :, :]
            min_dist = np.linalg.norm(diff, axis=2).min(axis=1)
            keep_candidate = min_dist <= grow_radius_m
            grown_indices = candidate_indices[keep_candidate]
            candidate_set = {int(v) for v in grown_indices.tolist()}
            candidate_set.update(support_set)

            for point_idx in candidate_set:
                vote_map = point_votes.setdefault(int(point_idx), {})
                vote_map[str(object_id)] = vote_map.get(str(object_id), 0.0) + (
                    3.0 if int(point_idx) in support_set else 1.25
                )

        for point_idx, votes in point_votes.items():
            if point_idx < 0 or point_idx >= xyz.shape[0]:
                continue
            if instance_id[point_idx] >= 0:
                continue
            best_object_id, best_score = max(votes.items(), key=lambda item: item[1])
            if float(best_score) < 1.0:
                continue
            export_meta = object_export_meta.get(str(best_object_id))
            if export_meta is None:
                continue
            export_index, export_class_id, semantic_rgb = export_meta
            instance_id[point_idx] = int(export_index)
            class_id[point_idx] = int(export_class_id)
            rgb[point_idx] = semantic_rgb

    def _integrate_geometry_keyframe(self, packet: KeyframePacket, local_cloud: LocalCloudPacket) -> None:
        xyz = self._load_cloud_xyz(local_cloud)
        if xyz.size == 0:
            return
        point_object_ids = self.object_memory.point_object_ids_for_local_cloud(local_cloud.local_cloud_id, xyz.shape[0])
        sensor_origin_world = (
            np.asarray(packet.t_world_cam, dtype=np.float32).reshape(4, 4)[:3, 3].astype(np.float32, copy=False)
        )
        self.voxel_map.integrate(
            xyz_world=xyz.astype(np.float32, copy=False),
            node_ids=point_object_ids,
            stamp_sec=float(packet.stamp_sec),
            sensor_origin_world=sensor_origin_world,
        )

    def _collect_object_points(self, point_support_refs: list[tuple[str, list[int]]]) -> np.ndarray:
        parts: list[np.ndarray] = []
        for local_cloud_id, point_indices in point_support_refs:
            local_cloud = self.local_cloud_lookup.get(local_cloud_id)
            if local_cloud is None:
                continue
            xyz = self._load_cloud_xyz(local_cloud)
            if xyz.size == 0:
                continue
            unique_indices = np.unique(np.asarray(point_indices, dtype=np.int32))
            valid = unique_indices[(unique_indices >= 0) & (unique_indices < xyz.shape[0])]
            if valid.size == 0:
                continue
            parts.append(xyz[valid])
        if not parts:
            return np.zeros((0, 3), dtype=np.float32)
        return np.vstack(parts).astype(np.float32, copy=False)

    def _voxel_downsample(self, xyz: np.ndarray, voxel_size_m: float) -> np.ndarray:
        if voxel_size_m <= 0.0 or xyz.shape[0] <= 1:
            return xyz
        voxel = np.floor(xyz / voxel_size_m).astype(np.int64)
        _, inverse = np.unique(voxel, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        sums = np.zeros((counts.shape[0], 3), dtype=np.float64)
        np.add.at(sums, inverse, xyz.astype(np.float64))
        downsampled = sums / np.maximum(counts[:, None], 1)
        return downsampled.astype(np.float32)

    def _object_label(self, obj) -> str:
        node_type = str(getattr(obj, "node_type", "thing") or "thing").strip().lower()
        posterior = getattr(obj, "posterior", {})
        if posterior:
            ranked_posterior = sorted(posterior.items(), key=lambda item: item[1], reverse=True)
            if ranked_posterior:
                top_label, top_prob = ranked_posterior[0]
                unknown_score = float(getattr(obj, "unknown_score", 1.0))
                reject_score = float(getattr(obj, "reject_score", 0.0))
                if node_type == "thing" and str(top_label) in STRUCTURAL_LABELS:
                    yolo_label, _yolo_conf = self._object_yolo_label(obj)
                    if yolo_label != "unknown":
                        return yolo_label
                if unknown_score >= 0.92 and float(top_prob) < 0.35:
                    yolo_label, _yolo_conf = self._object_yolo_label(obj)
                    if node_type == "thing" and yolo_label != "unknown":
                        return yolo_label
                    return "unknown"
                if reject_score >= 0.98 and float(top_prob) < 0.30:
                    yolo_label, _yolo_conf = self._object_yolo_label(obj)
                    if node_type == "thing" and yolo_label != "unknown":
                        return yolo_label
                    return "unknown"
                if node_type == "thing" and (
                    str(getattr(obj, "node_status", "tentative")) != "confirmed" or float(top_prob) < 0.60
                ):
                    yolo_label, _yolo_conf = self._object_yolo_label(obj)
                    if yolo_label != "unknown":
                        return yolo_label
                return str(top_label)
        if node_type == "thing":
            yolo_label, _yolo_conf = self._object_yolo_label(obj)
            if yolo_label != "unknown":
                return yolo_label
        descriptor_label, descriptor_conf, descriptor_margin = self._object_descriptor_label(obj)
        if descriptor_label != "unknown" and descriptor_conf > 0.0:
            if obj.state == "active" and (descriptor_conf >= 0.08 or descriptor_margin >= 0.01):
                return descriptor_label
            if descriptor_conf >= 0.10 and descriptor_margin >= 0.01:
                return descriptor_label
        if not obj.label_votes:
            return "unknown"
        return max(obj.label_votes.items(), key=lambda item: item[1])[0]

    def _object_yolo_label(self, obj) -> tuple[str, float]:
        raw_scores = {
            str(label): float(score)
            for label, score in getattr(obj, "yolo_logit_sum", {}).items()
            if str(label) not in STRUCTURAL_LABELS and np.isfinite(float(score)) and float(score) > 1e-6
        }
        if not raw_scores:
            return "unknown", 0.0
        total = float(sum(raw_scores.values()))
        if total <= 1e-6:
            return "unknown", 0.0
        label, score = max(raw_scores.items(), key=lambda item: item[1])
        return str(label), float(score / total)

    def _object_descriptor_label(self, obj) -> tuple[str, float, float]:
        descriptor = getattr(obj, "semantic_descriptor", None)
        if not descriptor:
            return "unknown", 0.0, 0.0
        ranked = self.semantic_ranker.rank_descriptor(descriptor, topk=2)
        if not ranked:
            return "unknown", 0.0, 0.0
        label, score = ranked[0]
        margin = float(score - ranked[1][1]) if len(ranked) > 1 else float(score)
        return label, float(score), margin

    def _should_export_object(self, obj) -> bool:
        if str(getattr(obj, "node_type", "thing") or "thing").strip().lower() != "thing":
            return False
        label = self._object_label(obj)
        if label == "unknown" or label in STRUCTURAL_LABELS:
            return False
        support_count = sum(len(point_indices) for _local_cloud_id, point_indices in obj.point_support_refs)
        if obj.state == "active":
            return True
        if obj.state == "stale":
            if not self.export_stale_objects:
                return False
            if support_count < max(self.args.min_observation_points, 3):
                return False
            if obj.observation_count < 1:
                return False
            return True
        if obj.state != "pending":
            return False
        if support_count < max(int(self.args.min_observation_points * 0.5), 3):
            return False
        if obj.observation_count < 1:
            return False
        if obj.stability_score < 0.05 and obj.promotion_evidence < 0.10:
            return False
        return True

    def _cleanup_snapshot_object_points(self, xyz: np.ndarray, label: str) -> np.ndarray:
        if not self.online_cleanup_object_points:
            return xyz
        if xyz.shape[0] == 0 or label in STRUCTURAL_LABELS or label == "unknown":
            return xyz
        if xyz.shape[0] < 24:
            return xyz
        voxel_size = 0.18
        voxel = np.floor(xyz / voxel_size).astype(np.int32)
        unique_voxel, inverse = np.unique(voxel, axis=0, return_inverse=True)
        voxel_to_points: dict[int, list[int]] = {}
        for point_idx, voxel_idx in enumerate(inverse.tolist()):
            voxel_to_points.setdefault(int(voxel_idx), []).append(int(point_idx))
        coord_to_index = {tuple(coord.tolist()): int(idx) for idx, coord in enumerate(unique_voxel)}

        visited: set[int] = set()
        best_indices: np.ndarray | None = None
        best_score = None
        for voxel_idx, coord in enumerate(unique_voxel):
            if voxel_idx in visited:
                continue
            queue = [int(voxel_idx)]
            visited.add(int(voxel_idx))
            component_voxels: list[int] = []
            component_points: list[int] = []
            while queue:
                current = queue.pop()
                component_voxels.append(current)
                component_points.extend(voxel_to_points.get(current, []))
                base = unique_voxel[current]
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            neighbor_key = (int(base[0] + dx), int(base[1] + dy), int(base[2] + dz))
                            neighbor = coord_to_index.get(neighbor_key)
                            if neighbor is None or neighbor in visited:
                                continue
                            visited.add(neighbor)
                            queue.append(neighbor)
            if not component_points:
                continue
            component_xyz = xyz[np.asarray(component_points, dtype=np.int32)]
            bbox_min = component_xyz.min(axis=0)
            bbox_max = component_xyz.max(axis=0)
            diag = float(np.linalg.norm(bbox_max - bbox_min))
            score = float(component_xyz.shape[0]) / max(diag, 0.10)
            if best_score is None or score > best_score:
                best_score = score
                best_indices = np.asarray(sorted(set(component_points)), dtype=np.int32)
        if best_indices is None or best_indices.size == 0:
            return xyz
        return xyz[best_indices]

    def _semantic_color(self, class_id: int) -> np.ndarray:
        if class_id < 0:
            return np.array([180, 180, 180], dtype=np.uint8)
        hue = (class_id * 0.137) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        return np.asarray([int(255 * channel) for channel in rgb], dtype=np.uint8)

    def _write_status(self, total_keyframes: int, status: str, processed_now: int) -> None:
        payload = {
            "scene_dir": str(self.scene_dir),
            "export_dir": str(self.export_dir),
            "output_dir": str(self.output_dir),
            "status": status,
            "total_keyframes_seen": int(total_keyframes),
            "processed_keyframes": int(len(self.processed_keyframe_ids)),
            "processed_in_last_pass": int(processed_now),
            "last_semantic_keyframe_id": None if self.last_semantic_keyframe_id is None else int(self.last_semantic_keyframe_id),
            "num_objects": int(len(self.object_memory.objects)),
            "num_thing_nodes": int(
                sum(1 for obj in self.object_memory.objects.values() if str(getattr(obj, "node_type", "thing")) == "thing")
            ),
            "num_stuff_nodes": int(
                sum(1 for obj in self.object_memory.objects.values() if str(getattr(obj, "node_type", "thing")) == "stuff")
            ),
            "num_active_objects": int(sum(1 for obj in self.object_memory.objects.values() if obj.state == "active")),
            "num_pending_objects": int(sum(1 for obj in self.object_memory.objects.values() if obj.state == "pending")),
            "num_stale_objects": int(sum(1 for obj in self.object_memory.objects.values() if obj.state == "stale")),
            "num_confirmed_nodes": int(
                sum(1 for obj in self.object_memory.objects.values() if str(getattr(obj, "node_status", "")) == "confirmed")
            ),
            "num_tentative_nodes": int(
                sum(1 for obj in self.object_memory.objects.values() if str(getattr(obj, "node_status", "")) == "tentative")
            ),
            "num_unknown_nodes": int(
                sum(1 for obj in self.object_memory.objects.values() if str(getattr(obj, "node_status", "")) == "unknown")
            ),
            "num_reject_nodes": int(
                sum(1 for obj in self.object_memory.objects.values() if str(getattr(obj, "node_status", "")) == "reject")
            ),
            "num_observations": int(self.total_observations),
            "artifact_path": str(self.snapshot_path),
            "voxel_node_map_path": str(self.voxel_map_path),
            "semantic_frontend": self.semantic_frontend,
            "geometry": self.voxel_map.summary(),
            "debug": {
                "per_mask_vote_debug": str(self.per_mask_vote_debug_path),
                "per_object_support_history": str(self.per_object_support_history_path),
                "pending_object_debug": str(self.pending_object_debug_path),
                "support_view_refinement_debug": str(self.support_view_refinement_debug_path),
                "keyframe_schedule_debug": str(self.keyframe_schedule_debug_path),
                "visibility_projector_debug": str(self.visibility_projector_debug_path),
            },
        }
        self.status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    runner = OnlineSemanticObserverRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
