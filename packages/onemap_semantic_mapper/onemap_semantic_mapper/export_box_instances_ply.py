from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .final_consolidation import write_ascii_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export box instance submaps to a single colored PLY."
    )
    parser.add_argument(
        "--semantic-output-dir",
        required=True,
        help="Directory containing object_memory.json and track_submaps/",
    )
    parser.add_argument(
        "--output-ply",
        required=True,
        help="Output PLY path.",
    )
    parser.add_argument(
        "--label",
        default="box",
        help="Semantic label to export. Defaults to 'box'.",
    )
    parser.add_argument(
        "--min-votes",
        type=float,
        default=0.0,
        help="Minimum accumulated vote for the requested label.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=80,
        help="Minimum voxelized points in a track submap to keep.",
    )
    parser.add_argument(
        "--min-hit-count",
        type=int,
        default=0,
        help="Minimum summed hit_count across the track submap.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional JSON summary output path.",
    )
    parser.add_argument(
        "--merge-dedup",
        action="store_true",
        help="Merge duplicate tracks that likely belong to the same physical box.",
    )
    parser.add_argument(
        "--merge-voxel-size",
        type=float,
        default=0.06,
        help="Voxel size used for overlap-based deduplication.",
    )
    parser.add_argument(
        "--merge-max-centroid-dist",
        type=float,
        default=0.45,
        help="Maximum centroid distance for duplicate merging.",
    )
    parser.add_argument(
        "--merge-min-iou",
        type=float,
        default=0.12,
        help="Minimum voxel IoU for duplicate merging.",
    )
    parser.add_argument(
        "--merge-min-containment",
        type=float,
        default=0.35,
        help="Minimum voxel containment ratio for duplicate merging.",
    )
    return parser.parse_args()


def color_for_index(index: int) -> np.ndarray:
    palette = np.asarray(
        [
            [230, 57, 70],
            [29, 185, 84],
            [66, 135, 245],
            [249, 199, 79],
            [155, 93, 229],
            [0, 187, 249],
            [255, 146, 43],
            [67, 170, 139],
            [247, 37, 133],
            [87, 117, 144],
            [144, 190, 109],
            [249, 65, 68],
        ],
        dtype=np.uint8,
    )
    return palette[index % len(palette)]


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def build_groups(records: list[dict[str, object]], args: argparse.Namespace) -> tuple[list[list[int]], list[dict[str, object]]]:
    if not args.merge_dedup or len(records) <= 1:
        return [[idx] for idx in range(len(records))], []

    uf = UnionFind(len(records))
    merge_events: list[dict[str, object]] = []
    max_centroid_dist = float(args.merge_max_centroid_dist)
    min_iou = float(args.merge_min_iou)
    min_containment = float(args.merge_min_containment)

    for i in range(len(records)):
        ci = np.asarray(records[i]["centroid"], dtype=np.float32)
        si = records[i]["voxel_set"]
        for j in range(i + 1, len(records)):
            cj = np.asarray(records[j]["centroid"], dtype=np.float32)
            centroid_dist = float(np.linalg.norm(ci - cj))
            if centroid_dist > max_centroid_dist:
                continue
            sj = records[j]["voxel_set"]
            inter = len(si & sj)
            if inter <= 0:
                continue
            union = len(si | sj)
            iou = inter / max(union, 1)
            containment = max(inter / max(len(si), 1), inter / max(len(sj), 1))
            if iou >= min_iou or containment >= min_containment:
                uf.union(i, j)
                merge_events.append(
                    {
                        "lhs_object_id": records[i]["object_id"],
                        "rhs_object_id": records[j]["object_id"],
                        "centroid_dist": centroid_dist,
                        "voxel_intersection": inter,
                        "voxel_iou": iou,
                        "voxel_containment": containment,
                    }
                )

    groups_map: dict[int, list[int]] = {}
    for idx in range(len(records)):
        groups_map.setdefault(uf.find(idx), []).append(idx)
    groups = list(groups_map.values())
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups, merge_events


def fuse_group(records: list[dict[str, object]], group: list[int]) -> tuple[np.ndarray, np.ndarray]:
    voxel_accumulator: dict[tuple[int, int, int], dict[str, object]] = {}
    for idx in group:
        record = records[idx]
        voxel_keys = np.asarray(record["voxel_keys"], dtype=np.int32)
        xyz = np.asarray(record["xyz"], dtype=np.float32)
        hit_count = np.asarray(record["hit_count"], dtype=np.int32).reshape(-1)
        for key_arr, point, hits in zip(voxel_keys, xyz, hit_count):
            key = (int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))
            entry = voxel_accumulator.get(key)
            weight = max(int(hits), 1)
            if entry is None:
                voxel_accumulator[key] = {
                    "weighted_xyz": point.astype(np.float64) * float(weight),
                    "weight": float(weight),
                    "hits": int(hits),
                }
            else:
                entry["weighted_xyz"] += point.astype(np.float64) * float(weight)
                entry["weight"] += float(weight)
                entry["hits"] += int(hits)

    merged_xyz = []
    merged_confidence = []
    for entry in voxel_accumulator.values():
        merged_xyz.append((entry["weighted_xyz"] / max(float(entry["weight"]), 1.0)).astype(np.float32))
        merged_confidence.append(float(entry["hits"]))
    return np.asarray(merged_xyz, dtype=np.float32), np.asarray(merged_confidence, dtype=np.float32)


def main() -> int:
    args = parse_args()
    semantic_output_dir = Path(args.semantic_output_dir).resolve()
    object_memory_path = semantic_output_dir / "object_memory.json"
    track_submaps_dir = semantic_output_dir / "track_submaps"
    output_ply = Path(args.output_ply).resolve()
    summary_json = Path(args.summary_json).resolve() if args.summary_json else None

    if not object_memory_path.exists():
        raise FileNotFoundError(f"object_memory.json not found: {object_memory_path}")
    if not track_submaps_dir.exists():
        raise FileNotFoundError(f"track_submaps not found: {track_submaps_dir}")

    payload = json.loads(object_memory_path.read_text(encoding="utf-8"))
    objects = payload.get("objects", {})

    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    instance_parts: list[np.ndarray] = []
    class_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    kept_objects: list[dict[str, object]] = []

    class_id = 1
    raw_records: list[dict[str, object]] = []
    for object_id in sorted(objects.keys()):
        obj = objects[object_id]
        label_votes = obj.get("label_votes", {}) or {}
        vote = float(label_votes.get(args.label, 0.0))
        if vote < float(args.min_votes):
            continue

        submap_path = track_submaps_dir / f"{object_id}.npz"
        if not submap_path.exists():
            continue

        data = np.load(submap_path)
        xyz = np.asarray(data["xyz_mean"], dtype=np.float32)
        voxel_keys = np.asarray(data["voxel_keys"], dtype=np.int32)
        hit_count = np.asarray(data["hit_count"], dtype=np.int32).reshape(-1)
        if xyz.shape[0] < int(args.min_points):
            continue
        total_hits = int(hit_count.sum())
        if total_hits < int(args.min_hit_count):
            continue

        raw_records.append(
            {
                "object_id": object_id,
                "label": args.label,
                "vote": vote,
                "num_points": int(xyz.shape[0]),
                "total_hits": total_hits,
                "track_submap_path": str(submap_path),
                "xyz": xyz,
                "voxel_keys": voxel_keys,
                "hit_count": hit_count,
                "centroid": xyz.mean(axis=0).tolist(),
                "voxel_set": set(map(tuple, np.floor(xyz / float(args.merge_voxel_size)).astype(np.int32).tolist())),
            }
        )

    if not raw_records:
        raise RuntimeError(
            f"No track submaps matched label={args.label!r} with "
            f"min_votes={args.min_votes}, min_points={args.min_points}, "
            f"min_hit_count={args.min_hit_count}"
        )

    groups, merge_events = build_groups(raw_records, args)

    instance_index = 0
    for group in groups:
        merged_xyz, merged_confidence = fuse_group(raw_records, group)
        color = color_for_index(instance_index)
        rgb = np.repeat(color.reshape(1, 3), merged_xyz.shape[0], axis=0)
        instance_ids = np.full((merged_xyz.shape[0],), instance_index, dtype=np.int32)
        class_ids = np.full((merged_xyz.shape[0],), class_id, dtype=np.int32)

        xyz_parts.append(merged_xyz)
        rgb_parts.append(rgb)
        instance_parts.append(instance_ids)
        class_parts.append(class_ids)
        confidence_parts.append(merged_confidence)
        kept_objects.append(
            {
                "merged_instance_id": instance_index,
                "label": args.label,
                "num_points": int(merged_xyz.shape[0]),
                "total_hits": float(merged_confidence.sum()),
                "color_rgb": [int(v) for v in color.tolist()],
                "source_object_ids": [str(raw_records[idx]["object_id"]) for idx in group],
                "source_votes": {
                    str(raw_records[idx]["object_id"]): float(raw_records[idx]["vote"])
                    for idx in group
                },
                "source_track_submaps": [str(raw_records[idx]["track_submap_path"]) for idx in group],
            }
        )
        instance_index += 1

    xyz = np.concatenate(xyz_parts, axis=0).astype(np.float32, copy=False)
    rgb = np.concatenate(rgb_parts, axis=0).astype(np.uint8, copy=False)
    instance_id = np.concatenate(instance_parts, axis=0).astype(np.int32, copy=False)
    class_id_arr = np.concatenate(class_parts, axis=0).astype(np.int32, copy=False)
    confidence = np.concatenate(confidence_parts, axis=0).astype(np.float32, copy=False)

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    write_ascii_ply(
        output_ply,
        xyz,
        rgb,
        instance_id,
        class_id_arr,
        confidence=confidence,
    )

    if summary_json is not None:
        summary_payload = {
            "semantic_output_dir": str(semantic_output_dir),
            "output_ply": str(output_ply),
            "label": args.label,
            "num_instances": len(kept_objects),
            "num_points": int(xyz.shape[0]),
            "min_votes": float(args.min_votes),
            "min_points": int(args.min_points),
            "min_hit_count": int(args.min_hit_count),
            "merge_dedup": bool(args.merge_dedup),
            "merge_voxel_size": float(args.merge_voxel_size),
            "merge_max_centroid_dist": float(args.merge_max_centroid_dist),
            "merge_min_iou": float(args.merge_min_iou),
            "merge_min_containment": float(args.merge_min_containment),
            "raw_instance_count": len(raw_records),
            "objects": kept_objects,
            "merge_events": merge_events,
        }
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "output_ply": str(output_ply),
                "num_instances": len(kept_objects),
                "num_points": int(xyz.shape[0]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
