from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .data_types import LocalCloudPacket
from .final_consolidation import write_ascii_ply
from .io.local_cloud_manifest import load_local_cloud_packets
from .object_memory import ObjectMemoryStore
from .textregion_extractor import TextRegionExtractor
from .voxel_node_map import VoxelNodeMap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dense semantic point cloud and sidecar from node-centric object memory.")
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--object-memory-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="scene")
    parser.add_argument("--color-mode", choices=["top1", "confidence"], default="top1")
    parser.add_argument("--voxel-map-path", default=None)
    parser.add_argument("--ovo-root", default="/home/peng/isacc_slam/reference/OVO")
    parser.add_argument("--ovo-config", default="data/working/configs/ovo_livo2_vanilla.yaml")
    parser.add_argument("--dataset-name", default="Replica")
    parser.add_argument("--class-set", choices=["full", "reduced"], default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--node-graph-radius-m", type=float, default=1.4)
    parser.add_argument("--skip-text-embeddings", action="store_true")
    return parser.parse_args()


def _ensure_ovo_import_path(ovo_root: Path) -> None:
    ovo_root_str = str(ovo_root.expanduser().resolve())
    if ovo_root_str not in sys.path:
        sys.path.insert(0, ovo_root_str)


def _semantic_color(class_id: int, confidence: float, color_mode: str) -> np.ndarray:
    if class_id < 0:
        return np.array([180, 180, 180], dtype=np.uint8)
    hue = (class_id * 0.137) % 1.0
    sat = 0.65
    val = 0.95 if color_mode == "top1" else float(np.clip(0.35 + (confidence * 0.60), 0.35, 0.95))
    import colorsys

    rgb = colorsys.hsv_to_rgb(hue, sat, val)
    return np.asarray([int(255 * channel) for channel in rgb], dtype=np.uint8)


def _load_clouds(local_clouds: list[LocalCloudPacket]) -> dict[str, np.ndarray]:
    clouds: dict[str, np.ndarray] = {}
    for packet in local_clouds:
        with np.load(Path(packet.cloud_path)) as data:
            clouds[packet.local_cloud_id] = np.asarray(data["xyz"], dtype=np.float32)
    return clouds


def _top_posterior(obj) -> tuple[str, float]:
    posterior = getattr(obj, "posterior", {})
    if posterior:
        label, conf = max(posterior.items(), key=lambda item: item[1])
        return str(label), float(conf)
    if obj.label_votes:
        total = max(float(sum(obj.label_votes.values())), 1e-6)
        label, score = max(obj.label_votes.items(), key=lambda item: item[1])
        return str(label), float(score) / total
    return "unknown", 0.0


def _export_label_and_confidence(obj) -> tuple[str, float]:
    node_status = str(getattr(obj, "node_status", "tentative"))
    if node_status in {"unknown", "reject"}:
        return "unknown", 0.0
    label, conf = _top_posterior(obj)
    if node_status == "tentative":
        return label, float(conf) * 0.5
    return label, conf


def _estimate_normals(xyz: np.ndarray) -> np.ndarray:
    if xyz.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    try:
        import open3d as o3d  # type: ignore

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64, copy=False))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=min(max(xyz.shape[0] // 200, 8), 24)))
        normals = np.asarray(pcd.normals, dtype=np.float32)
        if normals.shape != xyz.shape:
            return np.zeros_like(xyz, dtype=np.float32)
        return normals
    except Exception:
        return np.zeros_like(xyz, dtype=np.float32)


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
            relation = "overlap" if overlap > 1e-4 else "nearby"
            weight = float(np.exp(-dist / max(radius_m, 1e-4))) + (0.25 if overlap > 1e-4 else 0.0)
            edges.append(
                {
                    "src": str(object_id_a),
                    "dst": str(object_id_b),
                    "relation": relation,
                    "distance_m": dist,
                    "weight": weight,
                }
            )
    return edges


def _load_class_names(ovo_root: Path, dataset_name: str, class_set: str) -> list[str]:
    eval_info_path = ovo_root / "data" / "working" / "configs" / dataset_name / "eval_info.yaml"
    with eval_info_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if class_set == "reduced":
        return list(payload.get("class_names_reduced", payload["class_names"]))
    classes = [str(name) for name in payload["class_names"]]
    return [name for name in classes if name and name != "0"]


def _load_semantic_cfg(ovo_root: Path, ovo_config: str) -> dict[str, Any]:
    config_path = Path(ovo_config)
    if not config_path.is_absolute():
        config_path = ovo_root / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return dict(config["semantic"])


def _text_embedding_bundle(
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[str], str]:
    if args.skip_text_embeddings:
        return np.zeros((0, 0), dtype=np.float32), [], "skipped"
    try:
        ovo_root = Path(args.ovo_root).expanduser().resolve()
        _ensure_ovo_import_path(ovo_root)
        os.chdir(ovo_root)
        class_names = _load_class_names(ovo_root, args.dataset_name, args.class_set)
        semantic_cfg = _load_semantic_cfg(ovo_root, args.ovo_config)
        ranker = TextRegionExtractor(
            semantic_cfg=semantic_cfg,
            class_names=class_names,
            device=args.device,
            topk_labels=5,
        )
        text_embeddings = ranker.text_embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
        return text_embeddings, class_names, ""
    except Exception as exc:
        return np.zeros((0, 0), dtype=np.float32), [], str(exc)


def main() -> None:
    args = parse_args()
    export_dir = Path(args.export_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    local_clouds = load_local_cloud_packets(export_dir / "local_cloud_packets.jsonl")
    cloud_lookup = _load_clouds(local_clouds)
    memory_payload = json.loads(Path(args.object_memory_path).expanduser().resolve().read_text(encoding="utf-8"))
    store = ObjectMemoryStore.from_dict(memory_payload)

    class_names: list[str] = sorted(
        {
            label
            for obj in store.objects.values()
            for label in getattr(obj, "posterior", {}).keys()
        }
        | {
            label
            for obj in store.objects.values()
            for label in obj.label_votes.keys()
        }
    )
    class_lookup = {name: idx for idx, name in enumerate(class_names)}
    object_index_lookup = {object_id: idx for idx, object_id in enumerate(sorted(store.objects.keys()))}
    voxel_map_path = None if args.voxel_map_path in (None, "", "None") else Path(args.voxel_map_path).expanduser().resolve()

    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    instance_parts: list[np.ndarray] = []
    class_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    point_to_node: list[tuple[str, int]] = []

    if voxel_map_path is not None and voxel_map_path.exists():
        voxel_map = VoxelNodeMap.load(voxel_map_path)
        xyz, _weight, node_ids, node_conf = voxel_map.top_assignment()
        if xyz.size > 0:
            rgb = np.tile(np.array([[180, 180, 180]], dtype=np.uint8), (xyz.shape[0], 1))
            instance_id = np.full((xyz.shape[0],), -1, dtype=np.int32)
            class_id = np.full((xyz.shape[0],), -1, dtype=np.int32)
            confidence = np.asarray(node_conf, dtype=np.float32).copy()
            for idx, object_id in enumerate(node_ids.tolist()):
                if not object_id:
                    continue
                obj = store.objects.get(str(object_id))
                if obj is None:
                    continue
                label, conf = _export_label_and_confidence(obj)
                cls_id = int(class_lookup.get(label, -1))
                rgb[idx] = _semantic_color(cls_id, conf, args.color_mode)
                instance_id[idx] = int(object_index_lookup[str(object_id)])
                class_id[idx] = cls_id
                confidence[idx] = max(float(confidence[idx]), float(conf))
                point_to_node.append(("voxel", int(idx)))
            xyz_parts.append(xyz.astype(np.float32, copy=False))
            rgb_parts.append(rgb.astype(np.uint8, copy=False))
            instance_parts.append(instance_id)
            class_parts.append(class_id)
            confidence_parts.append(confidence.astype(np.float32, copy=False))
    else:
        for local_cloud_id, point_states in store.point_assignments.items():
            xyz = cloud_lookup.get(local_cloud_id)
            if xyz is None or xyz.size == 0:
                continue
            rgb = np.tile(np.array([[180, 180, 180]], dtype=np.uint8), (xyz.shape[0], 1))
            instance_id = np.full((xyz.shape[0],), -1, dtype=np.int32)
            class_id = np.full((xyz.shape[0],), -1, dtype=np.int32)
            confidence = np.zeros((xyz.shape[0],), dtype=np.float32)
            for point_index, state in point_states.items():
                if point_index < 0 or point_index >= xyz.shape[0]:
                    continue
                object_id = state.get("current_object_id")
                if not object_id:
                    continue
                obj = store.objects.get(str(object_id))
                if obj is None:
                    continue
                label, conf = _export_label_and_confidence(obj)
                cls_id = int(class_lookup.get(label, -1))
                rgb[int(point_index)] = _semantic_color(cls_id, conf, args.color_mode)
                instance_id[int(point_index)] = int(object_index_lookup[str(object_id)])
                class_id[int(point_index)] = cls_id
                confidence[int(point_index)] = float(conf)
                point_to_node.append((local_cloud_id, int(point_index)))
            xyz_parts.append(xyz)
            rgb_parts.append(rgb)
            instance_parts.append(instance_id)
            class_parts.append(class_id)
            confidence_parts.append(confidence)

    if not xyz_parts:
        raise RuntimeError(f"No dense points available for export from {export_dir}")

    xyz = np.vstack(xyz_parts).astype(np.float32, copy=False)
    rgb = np.vstack(rgb_parts).astype(np.uint8, copy=False)
    instance_id = np.concatenate(instance_parts)
    class_id = np.concatenate(class_parts)
    confidence = np.concatenate(confidence_parts)
    normals = _estimate_normals(xyz)
    node_graph = _build_node_graph(store, radius_m=float(args.node_graph_radius_m))
    text_embeddings, vocabulary, text_embedding_error = _text_embedding_bundle(args)

    ply_path = output_dir / f"{args.prefix}_dense_semantic.ply"
    write_ascii_ply(ply_path, xyz, rgb, instance_id, class_id, normals=normals, confidence=confidence)

    node_ids = np.asarray(sorted(store.objects.keys()), dtype=np.str_)
    node_types = np.asarray([str(store.objects[node_id].node_type) for node_id in node_ids.tolist()], dtype=np.str_)
    node_unknown = np.asarray([float(store.objects[node_id].unknown_score) for node_id in node_ids.tolist()], dtype=np.float32)
    node_reject = np.asarray([float(store.objects[node_id].reject_score) for node_id in node_ids.tolist()], dtype=np.float32)
    node_labels = np.asarray([_top_posterior(store.objects[node_id])[0] for node_id in node_ids.tolist()], dtype=np.str_)

    sidecar_path = output_dir / f"{args.prefix}_semantic_sidecar.npz"
    np.savez_compressed(
        sidecar_path,
        xyz=xyz,
        rgb=rgb,
        normals=normals,
        instance_id=instance_id,
        class_id=class_id,
        confidence=confidence,
        class_names=np.asarray(class_names, dtype=np.str_),
        vocabulary=np.asarray(vocabulary if vocabulary else class_names, dtype=np.str_),
        text_embeddings=text_embeddings.astype(np.float32, copy=False),
        text_embedding_error=np.asarray(text_embedding_error, dtype=np.str_),
        node_ids=node_ids,
        node_types=node_types,
        node_status=np.asarray([str(getattr(store.objects[node_id], "node_status", "tentative")) for node_id in node_ids.tolist()], dtype=np.str_),
        node_unknown=node_unknown,
        node_reject=node_reject,
        node_labels=node_labels,
        node_posterior_json=np.asarray(
            [json.dumps(getattr(store.objects[node_id], "posterior", {}), ensure_ascii=False) for node_id in node_ids.tolist()],
            dtype=np.str_,
        ),
        node_descriptor_bank_json=np.asarray(
            [json.dumps(getattr(store.objects[node_id], "descriptor_bank", []), ensure_ascii=False) for node_id in node_ids.tolist()],
            dtype=np.str_,
        ),
        node_graph_src=np.asarray([edge["src"] for edge in node_graph], dtype=np.str_),
        node_graph_dst=np.asarray([edge["dst"] for edge in node_graph], dtype=np.str_),
        node_graph_relation=np.asarray([edge["relation"] for edge in node_graph], dtype=np.str_),
        node_graph_distance_m=np.asarray([edge["distance_m"] for edge in node_graph], dtype=np.float32),
        node_graph_weight=np.asarray([edge["weight"] for edge in node_graph], dtype=np.float32),
        point_to_node_cloud=np.asarray([item[0] for item in point_to_node], dtype=np.str_),
        point_to_node_point_idx=np.asarray([item[1] for item in point_to_node], dtype=np.int32),
    )

    scene_nodes_payload: list[dict[str, Any]] = []
    for object_id, obj in sorted(store.objects.items()):
        label, conf = _export_label_and_confidence(obj)
        scene_nodes_payload.append(
            {
                "object_id": str(object_id),
                "node_type": str(getattr(obj, "node_type", "thing")),
                "node_status": str(getattr(obj, "node_status", "tentative")),
                "top_label": str(label),
                "top_confidence": float(conf),
                "posterior": {str(k): float(v) for k, v in getattr(obj, "posterior", {}).items()},
                "unknown_score": float(getattr(obj, "unknown_score", 1.0)),
                "reject_score": float(getattr(obj, "reject_score", 0.0)),
                "observation_count": int(obj.observation_count),
                "best_view_keyframes": [int(v) for v in obj.best_view_keyframes],
                "descriptor_bank_size": int(len(getattr(obj, "descriptor_bank", []))),
            }
        )
    (output_dir / f"{args.prefix}_nodes.json").write_text(
        json.dumps(scene_nodes_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
