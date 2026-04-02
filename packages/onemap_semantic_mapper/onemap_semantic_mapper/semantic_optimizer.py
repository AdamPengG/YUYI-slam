from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .data_types import KeyframePacket
from .io.keyframe_manifest import load_keyframe_packets
from .object_memory import ObjectMemoryStore
from .semantic_observer import MaskObservation
from .textregion_extractor import TextRegionExtractor

STRUCTURAL_LABELS = {
    "wall",
    "floor",
    "ceiling",
    "door",
    "window",
    "blinds",
}


def _ensure_ovo_import_path(ovo_root: Path) -> None:
    ovo_root_str = str(ovo_root.expanduser().resolve())
    if ovo_root_str not in sys.path:
        sys.path.insert(0, ovo_root_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline node-level semantic optimizer for LiDAR-mainline semantic mapping.")
    parser.add_argument("--object-memory-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ovo-root", default="/home/peng/isacc_slam/reference/OVO")
    parser.add_argument("--ovo-config", default="data/working/configs/ovo_livo2_vanilla.yaml")
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--dataset-name", default="Replica")
    parser.add_argument("--class-set", choices=["full", "reduced"], default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--descriptor-model", choices=["pe", "siglip2"], default="pe")
    parser.add_argument("--reassign-unknown-points", action="store_true")
    parser.add_argument("--pairwise-radius-m", type=float, default=1.4)
    parser.add_argument("--pairwise-iters", type=int, default=2)
    parser.add_argument("--ambiguous-threshold", type=float, default=0.55)
    return parser.parse_args()


def _load_config(ovo_root: Path, ovo_config: str) -> dict[str, Any]:
    ovo_config_path = Path(ovo_config)
    if not ovo_config_path.is_absolute():
        ovo_config_path = ovo_root / ovo_config_path
    with ovo_config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return dict(config["semantic"])


def _load_classes(ovo_root: Path, dataset_name: str, class_set: str) -> list[str]:
    eval_info_path = ovo_root / "data" / "working" / "configs" / dataset_name / "eval_info.yaml"
    with eval_info_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if class_set == "reduced":
        return list(payload.get("class_names_reduced", payload["class_names"]))
    classes = [str(name) for name in payload["class_names"]]
    return [name for name in classes if name and name != "0"]


def _build_descriptor_cfg(semantic_cfg: dict[str, Any], descriptor_model: str) -> dict[str, Any]:
    patched = dict(semantic_cfg)
    clip_cfg = dict(semantic_cfg["clip"])
    if descriptor_model == "siglip2":
        clip_cfg["embed_type"] = "vanilla"
        clip_cfg["model_card"] = "SigLIP2-384"
    patched["clip"] = clip_cfg
    return patched


def _scene_nodes_payload(store: ObjectMemoryStore) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for obj in store.objects.values():
        support_count = sum(len(indices) for _cloud_id, indices in obj.point_support_refs)
        posterior = dict(getattr(obj, "posterior", {}))
        ranked = max((posterior or {"unknown": 1.0}).items(), key=lambda item: item[1])
        payload.append(
            {
                "object_id": str(obj.object_id),
                "node_type": str(getattr(obj, "node_type", "thing")),
                "node_status": str(getattr(obj, "node_status", "tentative")),
                "state": str(obj.state),
                "top_label": str(ranked[0]),
                "posterior": {str(k): float(v) for k, v in posterior.items()},
                "unknown_score": float(getattr(obj, "unknown_score", 1.0)),
                "reject_score": float(getattr(obj, "reject_score", 0.0)),
                "observation_count": int(obj.observation_count),
                "support_count": int(support_count),
                "best_view_keyframes": [int(v) for v in obj.best_view_keyframes],
                "descriptor_bank_size": int(len(getattr(obj, "descriptor_bank", []))),
                "yolo_logit_sum": {str(k): float(v) for k, v in getattr(obj, "yolo_logit_sum", {}).items()},
                "centroid_world": [float(v) for v in obj.centroid_world],
                "bbox_world": [float(v) for v in obj.bbox_world],
            }
        )
    payload.sort(key=lambda item: (item["node_type"], -item["support_count"], item["object_id"]))
    return payload


def _build_node_graph(store: ObjectMemoryStore, radius_m: float) -> list[dict[str, Any]]:
    object_ids = sorted(store.objects.keys())
    edges: list[dict[str, Any]] = []
    for idx, object_id_a in enumerate(object_ids):
        obj_a = store.objects[object_id_a]
        center_a = np.asarray(obj_a.centroid_world, dtype=np.float32)
        bbox_a = np.asarray(obj_a.bbox_world, dtype=np.float32)
        for object_id_b in object_ids[idx + 1 :]:
            obj_b = store.objects[object_id_b]
            if str(getattr(obj_a, "node_type", "thing")) != str(getattr(obj_b, "node_type", "thing")):
                continue
            center_b = np.asarray(obj_b.centroid_world, dtype=np.float32)
            dist = float(np.linalg.norm(center_a - center_b))
            if dist > float(radius_m):
                continue
            bbox_b = np.asarray(obj_b.bbox_world, dtype=np.float32)
            overlap_min = np.maximum(bbox_a[:3], bbox_b[:3])
            overlap_max = np.minimum(bbox_a[3:], bbox_b[3:])
            overlap = float(np.prod(np.maximum(overlap_max - overlap_min, 0.0)))
            edges.append(
                {
                    "src": str(object_id_a),
                    "dst": str(object_id_b),
                    "relation": "overlap" if overlap > 1e-4 else "nearby",
                    "distance_m": dist,
                    "weight": float(np.exp(-dist / max(radius_m, 1e-4))) + (0.25 if overlap > 1e-4 else 0.0),
                }
            )
    return edges


def _normalize_distribution(values: dict[str, float]) -> dict[str, float]:
    cleaned = {str(label): max(float(score), 0.0) for label, score in values.items() if float(score) > 0.0}
    total = float(sum(cleaned.values()))
    if total <= 1e-6:
        return {}
    return {label: score / total for label, score in cleaned.items()}


def _update_unknown_scores(store: ObjectMemoryStore) -> None:
    for obj in store.objects.values():
        posterior = _normalize_distribution(getattr(obj, "posterior", {}))
        obj.posterior = posterior
        if not posterior:
            obj.unknown_score = 1.0
            obj.reject_score = 0.0
            continue
        ranked = sorted(posterior.items(), key=lambda item: item[1], reverse=True)
        best_prob = float(ranked[0][1])
        second_prob = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        obj.unknown_score = float(np.clip(1.0 - min(best_prob * 1.6, 1.0), 0.0, 1.0))
        obj.reject_score = float(np.clip(1.0 - max(best_prob - second_prob, 0.0) * 4.0, 0.0, 1.0))
        best_label = str(ranked[0][0])
        obj.node_type = "stuff" if best_label in STRUCTURAL_LABELS else "thing"


def _is_ambiguous_node(obj, ambiguous_threshold: float) -> bool:
    posterior = _normalize_distribution(getattr(obj, "posterior", {}))
    if not posterior:
        return True
    ranked = sorted(posterior.items(), key=lambda item: item[1], reverse=True)
    best_prob = float(ranked[0][1]) if ranked else 0.0
    second_prob = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    if float(getattr(obj, "unknown_score", 1.0)) >= 0.50:
        return True
    if float(getattr(obj, "reject_score", 0.0)) >= 0.45:
        return True
    return best_prob < float(ambiguous_threshold) or (best_prob - second_prob) < 0.10


def _load_keyframe_lookup(export_dir: Path | None) -> dict[int, KeyframePacket]:
    if export_dir is None:
        return {}
    manifest = export_dir / "keyframe_packets.jsonl"
    if not manifest.exists():
        return {}
    return {packet.keyframe_id: packet for packet in load_keyframe_packets(manifest)}


def _review_ambiguous_nodes_with_descriptor(
    store: ObjectMemoryStore,
    ranker: TextRegionExtractor,
    keyframe_lookup: dict[int, KeyframePacket],
    ambiguous_threshold: float,
) -> int:
    if not keyframe_lookup:
        return 0
    reviewed = 0
    image_cache: dict[str, np.ndarray] = {}
    for obj in store.objects.values():
        if not _is_ambiguous_node(obj, ambiguous_threshold):
            continue
        review_scores: dict[str, float] = {}
        candidate_views = list(getattr(obj, "descriptor_bank", [])) or list(getattr(obj, "descriptor_views", []))
        candidate_views = sorted(
            candidate_views,
            key=lambda item: float(item.get("quality_score", 0.0)),
            reverse=True,
        )[:3]
        for view in candidate_views:
            keyframe_id = int(view.get("keyframe_id", -1))
            bbox_xyxy = [int(v) for v in view.get("bbox_xyxy", [0, 0, 0, 0])]
            if keyframe_id < 0 or bbox_xyxy[2] <= bbox_xyxy[0] or bbox_xyxy[3] <= bbox_xyxy[1]:
                continue
            packet = keyframe_lookup.get(keyframe_id)
            if packet is None:
                continue
            image_rgb = image_cache.get(packet.rgb_path)
            if image_rgb is None:
                image_bgr = cv2.imread(str(packet.rgb_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_cache[packet.rgb_path] = image_rgb
            mask = np.zeros(image_rgb.shape[:2], dtype=bool)
            x0, y0, x1, y1 = bbox_xyxy
            x0 = int(np.clip(x0, 0, image_rgb.shape[1] - 1))
            x1 = int(np.clip(x1, x0 + 1, image_rgb.shape[1]))
            y0 = int(np.clip(y0, 0, image_rgb.shape[0] - 1))
            y1 = int(np.clip(y1, y0 + 1, image_rgb.shape[0]))
            mask[y0:y1, x0:x1] = True
            review_obs = ranker.enrich_mask_observations(
                image_rgb,
                [
                    MaskObservation(
                        mask_id=keyframe_id,
                        binary_mask=mask,
                        bbox_xyxy=[x0, y0, x1, y1],
                        semantic_label_candidates=[],
                        semantic_scores=[],
                        observation_kind=str(view.get("observation_kind", getattr(obj, "node_type", "thing"))),
                        view_quality=float(view.get("view_quality", view.get("quality_score", 0.0))),
                    )
                ],
            )
            if not review_obs:
                continue
            reviewed += 1
            obs = review_obs[0]
            for label, score in zip(obs.semantic_label_candidates, obs.semantic_scores, strict=False):
                review_scores[str(label)] = review_scores.get(str(label), 0.0) + float(score)
        if not review_scores:
            continue
        posterior = dict(getattr(obj, "posterior", {}))
        for label, score in review_scores.items():
            posterior[str(label)] = posterior.get(str(label), 0.0) + float(score) * 0.65
        obj.posterior = _normalize_distribution(posterior)
    _update_unknown_scores(store)
    return reviewed


def _node_pairwise_optimize(store: ObjectMemoryStore, radius_m: float, iters: int) -> None:
    object_ids = sorted(store.objects.keys())
    if len(object_ids) <= 1:
        return
    neighbors: dict[str, list[tuple[str, float]]] = {object_id: [] for object_id in object_ids}
    for idx, object_id_a in enumerate(object_ids):
        obj_a = store.objects[object_id_a]
        center_a = np.asarray(obj_a.centroid_world, dtype=np.float32)
        bbox_a = np.asarray(obj_a.bbox_world, dtype=np.float32)
        for object_id_b in object_ids[idx + 1 :]:
            obj_b = store.objects[object_id_b]
            if str(getattr(obj_a, "node_type", "thing")) != str(getattr(obj_b, "node_type", "thing")):
                continue
            center_b = np.asarray(obj_b.centroid_world, dtype=np.float32)
            dist = float(np.linalg.norm(center_a - center_b))
            if dist > float(radius_m):
                continue
            bbox_b = np.asarray(obj_b.bbox_world, dtype=np.float32)
            overlap_min = np.maximum(bbox_a[:3], bbox_b[:3])
            overlap_max = np.minimum(bbox_a[3:], bbox_b[3:])
            overlap = float(np.prod(np.maximum(overlap_max - overlap_min, 0.0)))
            weight = float(np.exp(-dist / max(radius_m, 1e-4))) + (0.25 if overlap > 1e-4 else 0.0)
            neighbors[object_id_a].append((object_id_b, weight))
            neighbors[object_id_b].append((object_id_a, weight))
    for _ in range(max(int(iters), 0)):
        next_posteriors: dict[str, dict[str, float]] = {}
        for object_id in object_ids:
            obj = store.objects[object_id]
            blended = {str(label): float(score) for label, score in getattr(obj, "posterior", {}).items()}
            if not blended:
                blended = dict(obj.label_votes)
            for neighbor_id, weight in neighbors.get(object_id, []):
                neighbor = store.objects[neighbor_id]
                for label, score in getattr(neighbor, "posterior", {}).items():
                    blended[str(label)] = blended.get(str(label), 0.0) + float(score) * float(weight) * 0.25
            next_posteriors[object_id] = _normalize_distribution(blended)
        for object_id, posterior in next_posteriors.items():
            store.objects[object_id].posterior = posterior
    _update_unknown_scores(store)


def main() -> None:
    args = parse_args()
    ovo_root = Path(args.ovo_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_ovo_import_path(ovo_root)
    os.chdir(ovo_root)

    object_memory_path = Path(args.object_memory_path).expanduser().resolve()
    payload = json.loads(object_memory_path.read_text(encoding="utf-8"))
    store = ObjectMemoryStore.from_dict(payload)

    semantic_cfg = _load_config(ovo_root, args.ovo_config)
    semantic_cfg = _build_descriptor_cfg(semantic_cfg, args.descriptor_model)
    class_names = _load_classes(ovo_root, args.dataset_name, args.class_set)
    ranker = TextRegionExtractor(
        semantic_cfg=semantic_cfg,
        class_names=class_names,
        device=args.device,
        topk_labels=5,
    )
    store.update_posterior(ranker.rank_descriptor, STRUCTURAL_LABELS)
    export_dir = None if args.export_dir in (None, "", "None") else Path(args.export_dir).expanduser().resolve()
    keyframe_lookup = _load_keyframe_lookup(export_dir)
    reviewed_nodes = 0
    if args.descriptor_model == "siglip2":
        reviewed_nodes = _review_ambiguous_nodes_with_descriptor(
            store=store,
            ranker=ranker,
            keyframe_lookup=keyframe_lookup,
            ambiguous_threshold=float(args.ambiguous_threshold),
        )
    _node_pairwise_optimize(
        store=store,
        radius_m=float(args.pairwise_radius_m),
        iters=int(args.pairwise_iters),
    )
    reassigned = 0
    if args.reassign_unknown_points:
        reassigned = store.reassign_points_from_posterior()

    optimized_path = output_dir / "optimized_object_memory.json"
    optimized_path.write_text(json.dumps(store.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    scene_nodes = _scene_nodes_payload(store)
    scene_nodes_path = output_dir / "scene_nodes.json"
    scene_nodes_path.write_text(json.dumps(scene_nodes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    node_graph = _build_node_graph(store, radius_m=float(args.pairwise_radius_m))
    node_graph_path = output_dir / "scene_node_graph.json"
    node_graph_path.write_text(json.dumps(node_graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    optimized_sidecar_path = output_dir / "optimized_semantic_sidecar.npz"
    np.savez_compressed(
        optimized_sidecar_path,
        vocabulary=np.asarray(class_names, dtype=np.str_),
        text_embeddings=ranker.text_embeddings.detach().cpu().numpy().astype(np.float32, copy=False),
        node_ids=np.asarray([item["object_id"] for item in scene_nodes], dtype=np.str_),
        node_types=np.asarray([item["node_type"] for item in scene_nodes], dtype=np.str_),
        node_status=np.asarray([item["node_status"] for item in scene_nodes], dtype=np.str_),
        node_labels=np.asarray([item["top_label"] for item in scene_nodes], dtype=np.str_),
        node_posterior_json=np.asarray([json.dumps(item["posterior"], ensure_ascii=False) for item in scene_nodes], dtype=np.str_),
        node_graph_src=np.asarray([edge["src"] for edge in node_graph], dtype=np.str_),
        node_graph_dst=np.asarray([edge["dst"] for edge in node_graph], dtype=np.str_),
        node_graph_relation=np.asarray([edge["relation"] for edge in node_graph], dtype=np.str_),
        node_graph_distance_m=np.asarray([edge["distance_m"] for edge in node_graph], dtype=np.float32),
        node_graph_weight=np.asarray([edge["weight"] for edge in node_graph], dtype=np.float32),
    )

    summary = {
        "object_memory_path": str(object_memory_path),
        "optimized_object_memory_path": str(optimized_path),
        "scene_nodes_path": str(scene_nodes_path),
        "scene_node_graph_path": str(node_graph_path),
        "optimized_semantic_sidecar_path": str(optimized_sidecar_path),
        "descriptor_model": str(args.descriptor_model),
        "class_set": str(args.class_set),
        "num_nodes": int(len(store.objects)),
        "reviewed_nodes": int(reviewed_nodes),
        "reassigned_unknown_points": int(reassigned),
    }
    (output_dir / "semantic_optimizer_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
