from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data_types import LocalCloudPacket
from .io.keyframe_manifest import load_keyframe_packets
from .io.local_cloud_manifest import load_local_cloud_packets
from .io.observation_manifest import load_observation_links

STRUCTURAL_LABELS = {
    "wall",
    "floor",
    "ceiling",
    "door",
    "window",
    "blinds",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate FAST-LIVO2-guided semantic observations into a final semantic map.")
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--prefix", default="final_observer")
    parser.add_argument("--min-instance-points", type=int, default=120)
    return parser.parse_args()


def load_clouds(local_clouds: list[LocalCloudPacket]) -> dict[str, np.ndarray]:
    clouds: dict[str, np.ndarray] = {}
    for packet in local_clouds:
        data = np.load(Path(packet.cloud_path))
        clouds[packet.local_cloud_id] = np.asarray(data["xyz"], dtype=np.float32)
    return clouds


def write_ascii_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    instance_id: np.ndarray,
    class_id: np.ndarray,
    normals: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
) -> None:
    include_normals = normals is not None and np.asarray(normals).shape == xyz.shape
    include_confidence = confidence is not None and np.asarray(confidence).reshape(-1).shape[0] == xyz.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {xyz.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if include_normals:
        header.extend(
            [
                "property float nx",
                "property float ny",
                "property float nz",
            ]
        )
    header.extend(
        [
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        ]
    )
    if include_confidence:
        header.append("property float confidence")
    header.extend(
        [
        "property int object_id",
        "property int instance_id",
        "property int class_id",
        "end_header",
        ]
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(header) + "\n")
        normals_arr = np.asarray(normals, dtype=np.float32) if include_normals else None
        confidence_arr = np.asarray(confidence, dtype=np.float32).reshape(-1) if include_confidence else None
        for idx, (point, color, ins_id, cls_id) in enumerate(zip(xyz, rgb, instance_id, class_id)):
            parts = [f"{point[0]:.6f}", f"{point[1]:.6f}", f"{point[2]:.6f}"]
            if normals_arr is not None:
                normal = normals_arr[idx]
                parts.extend([f"{normal[0]:.6f}", f"{normal[1]:.6f}", f"{normal[2]:.6f}"])
            parts.extend([str(int(color[0])), str(int(color[1])), str(int(color[2]))])
            if confidence_arr is not None:
                parts.append(f"{float(confidence_arr[idx]):.6f}")
            parts.extend([str(int(ins_id)), str(int(ins_id)), str(int(cls_id))])
            handle.write(" ".join(parts) + "\n")


def cleanup_object_points(xyz: np.ndarray, label: str) -> np.ndarray:
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
    best_points = None
    best_score = None
    for voxel_idx, coord in enumerate(unique_voxel):
        if voxel_idx in visited:
            continue
        queue = [int(voxel_idx)]
        visited.add(int(voxel_idx))
        component_points: list[int] = []
        while queue:
            current = queue.pop()
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
            best_points = np.asarray(sorted(set(component_points)), dtype=np.int32)
    if best_points is None or best_points.size == 0:
        return xyz
    return xyz[best_points]


def bbox_overlap_score(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    inter_min = np.maximum(bbox_a[:3], bbox_b[:3])
    inter_max = np.minimum(bbox_a[3:], bbox_b[3:])
    inter_size = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(inter_size))
    if inter_vol <= 0.0:
        return 0.0
    vol_a = float(np.prod(np.maximum(bbox_a[3:] - bbox_a[:3], 1e-4)))
    vol_b = float(np.prod(np.maximum(bbox_b[3:] - bbox_b[:3], 1e-4)))
    return inter_vol / max(min(vol_a, vol_b), 1e-4)


def best_view_similarity(sources_a: list[dict], sources_b: list[dict]) -> float:
    views_a = {int(item["keyframe_id"]) for item in sources_a if item.get("keyframe_id") is not None}
    views_b = {int(item["keyframe_id"]) for item in sources_b if item.get("keyframe_id") is not None}
    if not views_a or not views_b:
        return 0.0
    inter = len(views_a & views_b)
    union = len(views_a | views_b)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def temporal_continuity_score(sources_a: list[dict], sources_b: list[dict], keyframe_lookup: dict[int, object]) -> float:
    stamps_a = [float(keyframe_lookup[item["keyframe_id"]].stamp_sec) for item in sources_a if item.get("keyframe_id") in keyframe_lookup]
    stamps_b = [float(keyframe_lookup[item["keyframe_id"]].stamp_sec) for item in sources_b if item.get("keyframe_id") in keyframe_lookup]
    if not stamps_a or not stamps_b:
        return 0.0
    delta = abs(float(np.mean(stamps_a)) - float(np.mean(stamps_b)))
    if delta <= 0.5:
        return 1.0
    if delta >= 10.0:
        return 0.0
    return float(max(0.0, 1.0 - (delta / 10.0)))


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir)
    export_dir = Path(args.export_dir)

    keyframes = load_keyframe_packets(export_dir / "keyframe_packets.jsonl")
    local_clouds = load_local_cloud_packets(export_dir / "local_cloud_packets.jsonl")
    observations = load_observation_links(export_dir / "observation_links.jsonl")
    if not keyframes:
        raise RuntimeError(f"No keyframe packets found in {export_dir}")
    if not local_clouds:
        raise RuntimeError(f"No local cloud packets found in {export_dir}")
    if not observations:
        raise RuntimeError(f"No observation links found in {export_dir}")

    cloud_lookup = load_clouds(local_clouds)
    keyframe_lookup = {packet.keyframe_id: packet for packet in keyframes}
    instance_lookup: dict[str, int] = {}
    class_lookup: dict[str, int] = {}
    class_names: list[str] = []

    grouped_points: dict[str, list[np.ndarray]] = {}
    grouped_labels: dict[str, list[str]] = {}
    grouped_sources: dict[str, list[dict]] = {}
    removed_supports: list[dict] = []
    merge_trace: list[dict] = []

    for idx, observation in enumerate(observations):
        if observation.abstained:
            continue
        if len(observation.point_indices) < args.min_instance_points:
            continue
        cloud_xyz = cloud_lookup.get(observation.local_cloud_id)
        if cloud_xyz is None:
            continue
        valid_indices = np.asarray(observation.point_indices, dtype=np.int32)
        valid_indices = valid_indices[(valid_indices >= 0) & (valid_indices < cloud_xyz.shape[0])]
        if valid_indices.size < args.min_instance_points:
            continue
        pts = cloud_xyz[valid_indices]
        object_id = observation.candidate_object_id or f"obs_{idx:05d}"
        label = observation.semantic_label_candidates[0] if observation.semantic_label_candidates else "unknown"
        grouped_points.setdefault(object_id, []).append(pts.astype(np.float32))
        grouped_labels.setdefault(object_id, []).append(label)
        grouped_sources.setdefault(object_id, []).append(
            {
                "keyframe_id": observation.keyframe_id,
                "local_cloud_id": observation.local_cloud_id,
                "point_count": int(pts.shape[0]),
                "source_rgb_path": keyframe_lookup[observation.keyframe_id].rgb_path if observation.keyframe_id in keyframe_lookup else None,
            }
        )

    consolidated_objects: list[dict] = []
    for object_id, parts in grouped_points.items():
        xyz = np.vstack(parts).astype(np.float32, copy=False)
        labels = grouped_labels.get(object_id, ["unknown"])
        label_counts: dict[str, int] = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        label = max(label_counts.items(), key=lambda item: item[1])[0]
        raw_point_count = int(xyz.shape[0])
        xyz = cleanup_object_points(xyz, label)
        if xyz.shape[0] != raw_point_count:
            removed_supports.append(
                {
                    "object_id": str(object_id),
                    "label": str(label),
                    "raw_point_count": raw_point_count,
                    "kept_point_count": int(xyz.shape[0]),
                    "removed_point_count": int(raw_point_count - xyz.shape[0]),
                }
            )
        if xyz.shape[0] < args.min_instance_points:
            continue
        bbox_min = xyz.min(axis=0)
        bbox_max = xyz.max(axis=0)
        consolidated_objects.append(
            {
                "object_id": object_id,
                "label": label,
                "xyz": xyz,
                "bbox": np.concatenate((bbox_min, bbox_max)).astype(np.float32),
                "sources": grouped_sources.get(object_id, []),
            }
        )

    merged = True
    while merged:
        merged = False
        next_objects: list[dict] = []
        consumed: set[int] = set()
        for idx, obj_a in enumerate(consolidated_objects):
            if idx in consumed:
                continue
            current = obj_a
            for jdx in range(idx + 1, len(consolidated_objects)):
                if jdx in consumed:
                    continue
                obj_b = consolidated_objects[jdx]
                if current["label"] != obj_b["label"]:
                    continue
                centroid_dist = float(
                    np.linalg.norm(current["xyz"].mean(axis=0) - obj_b["xyz"].mean(axis=0))
                )
                overlap = bbox_overlap_score(current["bbox"], obj_b["bbox"])
                view_similarity = best_view_similarity(current["sources"], obj_b["sources"])
                continuity = temporal_continuity_score(current["sources"], obj_b["sources"], keyframe_lookup)
                max_dist = 0.45 if current["label"] not in STRUCTURAL_LABELS else 0.75
                allow_merge = False
                if centroid_dist <= max_dist and continuity >= 0.40:
                    allow_merge = True
                if overlap >= 0.05:
                    allow_merge = True
                if view_similarity >= 0.35 and centroid_dist <= (max_dist * 1.25):
                    allow_merge = True
                if not allow_merge:
                    continue
                merged = True
                consumed.add(jdx)
                merge_trace.append(
                    {
                        "keep_object_id": str(current["object_id"]),
                        "drop_object_id": str(obj_b["object_id"]),
                        "label": str(current["label"]),
                        "centroid_dist": float(centroid_dist),
                        "bbox_overlap": float(overlap),
                        "best_view_similarity": float(view_similarity),
                        "temporal_continuity": float(continuity),
                        "reason": "same_label_merge",
                    }
                )
                merged_xyz = np.vstack((current["xyz"], obj_b["xyz"])).astype(np.float32, copy=False)
                merged_xyz = cleanup_object_points(merged_xyz, current["label"])
                bbox_min = merged_xyz.min(axis=0)
                bbox_max = merged_xyz.max(axis=0)
                current = {
                    "object_id": current["object_id"],
                    "label": current["label"],
                    "xyz": merged_xyz,
                    "bbox": np.concatenate((bbox_min, bbox_max)).astype(np.float32),
                    "sources": current["sources"] + obj_b["sources"],
                }
            next_objects.append(current)
        consolidated_objects = next_objects

    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    instance_parts: list[np.ndarray] = []
    class_parts: list[np.ndarray] = []
    objects_summary: list[dict] = []

    for object_id, obj in enumerate(consolidated_objects):
        label = str(obj["label"])
        if label not in class_lookup:
            class_lookup[label] = len(class_lookup)
            class_names.append(label)
        class_id = class_lookup[label]
        instance_lookup[str(obj["object_id"])] = object_id
        seed = np.random.default_rng(object_id)
        color = seed.integers(0, 255, size=(3,), dtype=np.uint8)
        xyz = np.asarray(obj["xyz"], dtype=np.float32)
        xyz_parts.append(xyz)
        rgb_parts.append(np.tile(color, (xyz.shape[0], 1)))
        instance_parts.append(np.full((xyz.shape[0],), object_id, dtype=np.int32))
        class_parts.append(np.full((xyz.shape[0],), class_id, dtype=np.int32))
        objects_summary.append(
            {
                "object_id": str(obj["object_id"]),
                "instance_id": object_id,
                "label": label,
                "point_count": int(xyz.shape[0]),
                "bbox_world": [float(v) for v in obj["bbox"].tolist()],
                "sources": obj["sources"],
            }
        )

    if not xyz_parts:
        raise RuntimeError("No observations survived consolidation thresholds.")

    xyz = np.vstack(xyz_parts)
    rgb = np.vstack(rgb_parts)
    instance_id = np.concatenate(instance_parts)
    class_id = np.concatenate(class_parts)

    output_cloud = scene_dir / f"{args.prefix}_semantic_cloud.ply"
    output_objects = scene_dir / f"{args.prefix}_objects.json"
    output_hydra = scene_dir / f"{args.prefix}_hydra_nodes_edges.json"
    output_summary = scene_dir / f"{args.prefix}_summary.json"
    output_object_debug = scene_dir / f"{args.prefix}_object_debug.json"
    output_removed_supports = scene_dir / f"{args.prefix}_removed_supports.json"
    output_merge_trace = scene_dir / f"{args.prefix}_merge_trace.json"

    write_ascii_ply(output_cloud, xyz, rgb, instance_id, class_id)
    output_objects.write_text(json.dumps(objects_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_object_debug.write_text(
        json.dumps(
            {
                "objects": objects_summary,
                "grouped_input_object_ids": sorted(grouped_points.keys()),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_removed_supports.write_text(json.dumps(removed_supports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_merge_trace.write_text(json.dumps(merge_trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_hydra.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": obj["object_id"],
                        "label": obj["label"],
                        "bbox_world": obj["bbox_world"],
                    }
                    for obj in objects_summary
                ],
                "edges": [],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_summary.write_text(
        json.dumps(
            {
                "scene_dir": str(scene_dir),
                "export_dir": str(export_dir),
                "num_keyframes": len(keyframes),
                "num_local_clouds": len(local_clouds),
                "num_observations": len(observations),
                "num_points": int(xyz.shape[0]),
                "num_objects": len(objects_summary),
                "class_names": class_names,
                "removed_support_events": len(removed_supports),
                "merge_events": len(merge_trace),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
