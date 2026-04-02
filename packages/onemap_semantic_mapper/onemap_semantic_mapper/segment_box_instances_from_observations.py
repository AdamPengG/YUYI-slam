from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .final_consolidation import write_ascii_ply
from .io.local_cloud_manifest import load_local_cloud_packets
from .io.observation_manifest import load_observation_links


@dataclass
class Fragment:
    fragment_id: int
    keyframe_id: int
    local_cloud_id: str
    points: np.ndarray
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    voxel_set: set[tuple[int, int, int]]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment box instances from observation links and local clouds."
    )
    parser.add_argument("--export-dir", required=True, help="Run export directory.")
    parser.add_argument("--output-ply", required=True, help="Output PLY path.")
    parser.add_argument("--summary-json", default="", help="Optional JSON summary path.")
    parser.add_argument("--label", default="box", help="Target label. Defaults to box.")
    parser.add_argument("--min-observation-points", type=int, default=40)
    parser.add_argument("--fragment-voxel-size", type=float, default=0.05)
    parser.add_argument("--min-fragment-voxels", type=int, default=12)
    parser.add_argument("--merge-max-centroid-dist", type=float, default=0.28)
    parser.add_argument("--merge-max-z-dist", type=float, default=0.45)
    parser.add_argument("--merge-min-iou", type=float, default=0.10)
    parser.add_argument("--merge-min-containment", type=float, default=0.30)
    parser.add_argument("--fuse-voxel-size", type=float, default=0.035)
    parser.add_argument("--min-instance-points", type=int, default=120)
    parser.add_argument("--near-ground-z-min", type=float, default=-1e9)
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


def connected_voxel_components(voxels: np.ndarray) -> list[np.ndarray]:
    if voxels.shape[0] == 0:
        return []
    unique_voxels, inverse = np.unique(voxels, axis=0, return_inverse=True)
    coord_to_index = {tuple(coord.tolist()): int(idx) for idx, coord in enumerate(unique_voxels)}
    visited: set[int] = set()
    voxel_components: list[list[int]] = []
    for voxel_idx, coord in enumerate(unique_voxels):
        if int(voxel_idx) in visited:
            continue
        stack = [int(voxel_idx)]
        visited.add(int(voxel_idx))
        comp = [int(voxel_idx)]
        while stack:
            cur = stack.pop()
            base = unique_voxels[cur]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        nb = (int(base[0] + dx), int(base[1] + dy), int(base[2] + dz))
                        nb_idx = coord_to_index.get(nb)
                        if nb_idx is None or nb_idx in visited:
                            continue
                        visited.add(nb_idx)
                        stack.append(nb_idx)
                        comp.append(int(nb_idx))
        voxel_components.append(comp)

    point_components: list[np.ndarray] = []
    for comp in voxel_components:
        comp_set = set(comp)
        point_mask = np.asarray([idx in comp_set for idx in inverse.tolist()], dtype=bool)
        point_components.append(np.nonzero(point_mask)[0].astype(np.int32))
    return point_components


def fuse_points(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    vox = np.floor(points / float(voxel_size)).astype(np.int32)
    unique_vox, inverse = np.unique(vox, axis=0, return_inverse=True)
    accum = np.zeros((unique_vox.shape[0], 3), dtype=np.float64)
    counts = np.zeros((unique_vox.shape[0],), dtype=np.int32)
    for idx, point in zip(inverse.tolist(), points):
        accum[int(idx)] += point.astype(np.float64)
        counts[int(idx)] += 1
    fused = accum / np.maximum(counts[:, None], 1)
    return fused.astype(np.float32)


def build_fragments(args: argparse.Namespace) -> tuple[list[Fragment], dict[str, np.ndarray]]:
    export_dir = Path(args.export_dir).resolve()
    observations = load_observation_links(export_dir / "observation_links.jsonl")
    local_cloud_packets = load_local_cloud_packets(export_dir / "local_cloud_packets.jsonl")
    cloud_cache: dict[str, np.ndarray] = {}
    cloud_paths = {packet.local_cloud_id: Path(packet.cloud_path) for packet in local_cloud_packets}
    fragments: list[Fragment] = []
    fragment_id = 0

    def load_cloud(local_cloud_id: str) -> np.ndarray:
        cached = cloud_cache.get(local_cloud_id)
        if cached is not None:
            return cached
        cloud_path = cloud_paths.get(local_cloud_id)
        if cloud_path is None:
            raise KeyError(f"Local cloud not found for {local_cloud_id}")
        cloud_data = np.load(cloud_path)
        xyz = np.asarray(cloud_data["xyz"], dtype=np.float32)
        cloud_cache[local_cloud_id] = xyz
        return xyz

    for obs in observations:
        labels = list(obs.semantic_label_candidates or [])
        if args.label not in labels:
            continue
        if len(obs.point_indices) < int(args.min_observation_points):
            continue
        xyz_all = load_cloud(obs.local_cloud_id)
        indices = np.asarray(obs.point_indices, dtype=np.int32)
        valid = indices[(indices >= 0) & (indices < xyz_all.shape[0])]
        if valid.size < int(args.min_observation_points):
            continue
        points = xyz_all[valid]
        if float(args.near_ground_z_min) > -1e8:
            points = points[points[:, 2] >= float(args.near_ground_z_min)]
        if points.shape[0] < int(args.min_observation_points):
            continue
        vox = np.floor(points / float(args.fragment_voxel_size)).astype(np.int32)
        components = connected_voxel_components(vox)
        for comp_indices in components:
            comp_points = points[comp_indices]
            comp_vox = vox[comp_indices]
            comp_voxel_set = set(map(tuple, np.unique(comp_vox, axis=0).tolist()))
            if len(comp_voxel_set) < int(args.min_fragment_voxels):
                continue
            fragments.append(
                Fragment(
                    fragment_id=fragment_id,
                    keyframe_id=int(obs.keyframe_id),
                    local_cloud_id=str(obs.local_cloud_id),
                    points=comp_points.astype(np.float32, copy=False),
                    centroid=comp_points.mean(axis=0).astype(np.float32),
                    bbox_min=comp_points.min(axis=0).astype(np.float32),
                    bbox_max=comp_points.max(axis=0).astype(np.float32),
                    voxel_set=comp_voxel_set,
                )
            )
            fragment_id += 1
    return fragments, cloud_cache


def group_fragments(fragments: list[Fragment], args: argparse.Namespace) -> tuple[list[list[int]], list[dict[str, float]]]:
    if not fragments:
        return [], []
    uf = UnionFind(len(fragments))
    merge_events: list[dict[str, float]] = []
    for i in range(len(fragments)):
        fi = fragments[i]
        for j in range(i + 1, len(fragments)):
            fj = fragments[j]
            centroid_dist = float(np.linalg.norm(fi.centroid - fj.centroid))
            if centroid_dist > float(args.merge_max_centroid_dist):
                continue
            z_dist = float(abs(fi.centroid[2] - fj.centroid[2]))
            if z_dist > float(args.merge_max_z_dist):
                continue
            inter = len(fi.voxel_set & fj.voxel_set)
            if inter <= 0:
                continue
            union = len(fi.voxel_set | fj.voxel_set)
            iou = inter / max(union, 1)
            containment = max(inter / max(len(fi.voxel_set), 1), inter / max(len(fj.voxel_set), 1))
            if iou >= float(args.merge_min_iou) or containment >= float(args.merge_min_containment):
                uf.union(i, j)
                merge_events.append(
                    {
                        "lhs_fragment_id": int(fi.fragment_id),
                        "rhs_fragment_id": int(fj.fragment_id),
                        "centroid_dist": centroid_dist,
                        "z_dist": z_dist,
                        "voxel_intersection": int(inter),
                        "voxel_iou": float(iou),
                        "voxel_containment": float(containment),
                    }
                )
    groups_map: dict[int, list[int]] = {}
    for idx in range(len(fragments)):
        groups_map.setdefault(uf.find(idx), []).append(idx)
    groups = list(groups_map.values())
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups, merge_events


def export_instances(args: argparse.Namespace, fragments: list[Fragment], groups: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    instance_parts: list[np.ndarray] = []
    class_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    summary_objects: list[dict[str, object]] = []
    class_id_value = 1

    instance_index = 0
    for group in groups:
        raw_points = np.concatenate([fragments[idx].points for idx in group], axis=0).astype(np.float32, copy=False)
        fused_xyz = fuse_points(raw_points, float(args.fuse_voxel_size))
        if fused_xyz.shape[0] < int(args.min_instance_points):
            continue
        color = color_for_index(instance_index)
        rgb = np.repeat(color.reshape(1, 3), fused_xyz.shape[0], axis=0)
        instance_id = np.full((fused_xyz.shape[0],), instance_index, dtype=np.int32)
        class_id = np.full((fused_xyz.shape[0],), class_id_value, dtype=np.int32)
        confidence = np.full((fused_xyz.shape[0],), float(len(group)), dtype=np.float32)

        xyz_parts.append(fused_xyz)
        rgb_parts.append(rgb)
        instance_parts.append(instance_id)
        class_parts.append(class_id)
        confidence_parts.append(confidence)
        summary_objects.append(
            {
                "instance_id": instance_index,
                "num_points": int(fused_xyz.shape[0]),
                "num_fragments": len(group),
                "source_fragments": [int(fragments[idx].fragment_id) for idx in group],
                "source_keyframes": sorted({int(fragments[idx].keyframe_id) for idx in group}),
                "source_local_cloud_ids": sorted({str(fragments[idx].local_cloud_id) for idx in group}),
                "color_rgb": [int(v) for v in color.tolist()],
            }
        )
        instance_index += 1

    if not xyz_parts:
        raise RuntimeError("No fused instances remained after filtering.")

    xyz = np.concatenate(xyz_parts, axis=0).astype(np.float32, copy=False)
    rgb = np.concatenate(rgb_parts, axis=0).astype(np.uint8, copy=False)
    instance_id = np.concatenate(instance_parts, axis=0).astype(np.int32, copy=False)
    class_id = np.concatenate(class_parts, axis=0).astype(np.int32, copy=False)
    confidence = np.concatenate(confidence_parts, axis=0).astype(np.float32, copy=False)
    return xyz, rgb, instance_id, class_id, confidence, summary_objects


def main() -> int:
    args = parse_args()
    output_ply = Path(args.output_ply).resolve()
    summary_json = Path(args.summary_json).resolve() if args.summary_json else None

    fragments, _cloud_cache = build_fragments(args)
    if not fragments:
        raise RuntimeError("No observation fragments built for the requested label.")

    groups, merge_events = group_fragments(fragments, args)
    xyz, rgb, instance_id, class_id, confidence, summary_objects = export_instances(args, fragments, groups)

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    write_ascii_ply(output_ply, xyz, rgb, instance_id, class_id, confidence=confidence)

    summary_payload = {
        "export_dir": str(Path(args.export_dir).resolve()),
        "output_ply": str(output_ply),
        "label": args.label,
        "num_fragments": len(fragments),
        "num_fragment_groups": len(groups),
        "num_instances": len(summary_objects),
        "num_points": int(xyz.shape[0]),
        "parameters": {
            "min_observation_points": int(args.min_observation_points),
            "fragment_voxel_size": float(args.fragment_voxel_size),
            "min_fragment_voxels": int(args.min_fragment_voxels),
            "merge_max_centroid_dist": float(args.merge_max_centroid_dist),
            "merge_max_z_dist": float(args.merge_max_z_dist),
            "merge_min_iou": float(args.merge_min_iou),
            "merge_min_containment": float(args.merge_min_containment),
            "fuse_voxel_size": float(args.fuse_voxel_size),
            "min_instance_points": int(args.min_instance_points),
            "near_ground_z_min": float(args.near_ground_z_min),
        },
        "merge_events": merge_events,
        "instances": summary_objects,
    }
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_ply": str(output_ply),
                "num_fragments": len(fragments),
                "num_instances": len(summary_objects),
                "num_points": int(xyz.shape[0]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
