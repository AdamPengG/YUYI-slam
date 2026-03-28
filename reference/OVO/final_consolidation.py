from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import scipy.spatial
import yaml


STRUCTURAL_CLASSES = {
    "wall",
    "floor",
    "ceiling",
    "door",
    "window",
    "stairs",
    "column",
    "beam",
}


@dataclass
class KeyframeRecord:
    frame_id: str
    rgb_path: Path
    depth_path: Path
    c2w: np.ndarray
    reason: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate a live OVO run into a cleaner final semantic map.")
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-id", default=None, help="Specific exporter run id under runs/ovo_pose_keyframes/. Defaults to latest matching scene.")
    parser.add_argument("--dataset-name", default="Replica")
    parser.add_argument("--voxel-length", type=float, default=0.02)
    parser.add_argument("--sdf-trunc", type=float, default=0.06)
    parser.add_argument("--depth-trunc", type=float, default=6.0)
    parser.add_argument("--semantic-radius", type=float, default=0.08)
    parser.add_argument("--semantic-knn", type=int, default=8)
    parser.add_argument("--instance-dbscan-eps", type=float, default=0.10)
    parser.add_argument("--instance-dbscan-min-points", type=int, default=25)
    parser.add_argument("--instance-min-points", type=int, default=120)
    parser.add_argument("--max-plot-points", type=int, default=300000)
    parser.add_argument("--prefix", default="final")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_scene_config(scene_name: str, dataset_name: str) -> dict:
    config_path = repo_root() / "reference" / "OVO" / "data" / "working" / "configs" / dataset_name / f"{scene_name}.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_export_run(scene_name: str, run_id: str | None) -> Path:
    base = repo_root() / "runs" / "ovo_pose_keyframes"
    if run_id is not None:
        run_dir = base / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Exporter run not found: {run_dir}")
        return run_dir

    candidates = sorted(base.glob(f"*_{scene_name}"))
    if not candidates:
        raise FileNotFoundError(f"No exporter runs found for scene {scene_name} under {base}")
    return candidates[-1]


def load_keyframes(export_run: Path, dataset_root: Path) -> list[KeyframeRecord]:
    keyframes_path = export_run / "export" / "keyframes.jsonl"
    if not keyframes_path.exists():
        raise FileNotFoundError(f"Missing keyframes manifest: {keyframes_path}")

    keyframes: list[KeyframeRecord] = []
    for line in keyframes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("is_keyframe", False):
            continue
        keyframes.append(
            KeyframeRecord(
                frame_id=str(row["frame_id"]),
                rgb_path=dataset_root / row["rgb_path"],
                depth_path=dataset_root / row["depth_path"],
                c2w=np.asarray(row["T_world_cam"], dtype=np.float64),
                reason=list(row.get("reason", [])),
            )
        )
    if not keyframes:
        raise RuntimeError(f"No keyframes found in {keyframes_path}")
    return keyframes


def load_semantic_snapshot(run_output: Path) -> dict[str, np.ndarray]:
    snapshot_path = run_output / "semantic_snapshot.npz"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing semantic snapshot: {snapshot_path}")
    snapshot = np.load(snapshot_path, allow_pickle=True)
    return {key: snapshot[key] for key in snapshot.files}


def build_intrinsic(cam_cfg: dict) -> o3d.camera.PinholeCameraIntrinsic:
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        int(cam_cfg["W"]),
        int(cam_cfg["H"]),
        float(cam_cfg["fx"]),
        float(cam_cfg["fy"]),
        float(cam_cfg["cx"]),
        float(cam_cfg["cy"]),
    )
    return intrinsic


def integrate_tsdf(keyframes: Iterable[KeyframeRecord], intrinsic: o3d.camera.PinholeCameraIntrinsic, depth_scale: float, voxel_length: float, sdf_trunc: float, depth_trunc: float) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for keyframe in keyframes:
        image_bgr = cv2.imread(str(keyframe.rgb_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(keyframe.depth_path), cv2.IMREAD_UNCHANGED)
        if image_bgr is None or depth_raw is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        color = o3d.geometry.Image(image_rgb)
        depth = o3d.geometry.Image(depth_raw.astype(np.uint16))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=float(depth_scale),
            depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
        )
        w2c = np.linalg.inv(keyframe.c2w)
        volume.integrate(rgbd, intrinsic, w2c)

    pcd = volume.extract_point_cloud()
    mesh = volume.extract_triangle_mesh()
    if mesh.has_vertices():
        mesh.compute_vertex_normals()
    return pcd, mesh


def assign_semantics(
    points_xyz: np.ndarray,
    snapshot: dict[str, np.ndarray],
    semantic_radius: float,
    semantic_knn: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    semantic_xyz = np.asarray(snapshot["xyz"], dtype=np.float32)
    semantic_instance = np.asarray(snapshot["instance_id"], dtype=np.int32)
    semantic_class = np.asarray(snapshot["class_id"], dtype=np.int32)
    class_names = np.asarray(snapshot["class_names"], dtype=object)

    tree = scipy.spatial.cKDTree(semantic_xyz)
    dists, indices = tree.query(points_xyz, k=min(semantic_knn, max(len(semantic_xyz), 1)), distance_upper_bound=semantic_radius)
    if np.ndim(indices) == 1:
        indices = indices[:, None]
        dists = dists[:, None]

    out_instance = np.full((points_xyz.shape[0],), -1, dtype=np.int32)
    out_class = np.full((points_xyz.shape[0],), -1, dtype=np.int32)

    for point_idx in range(points_xyz.shape[0]):
        valid = indices[point_idx] < len(semantic_xyz)
        if not np.any(valid):
            continue
        nn_idx = indices[point_idx][valid]
        inst_ids = semantic_instance[nn_idx]
        class_ids = semantic_class[nn_idx]

        inst_values, inst_counts = np.unique(inst_ids[inst_ids >= 0], return_counts=True)
        if inst_values.size > 0:
            out_instance[point_idx] = int(inst_values[np.argmax(inst_counts)])

        class_values, class_counts = np.unique(class_ids[class_ids >= 0], return_counts=True)
        if class_values.size > 0:
            out_class[point_idx] = int(class_values[np.argmax(class_counts)])

    return out_instance, out_class, class_names


def clean_instances(
    xyz: np.ndarray,
    instance_ids: np.ndarray,
    class_ids: np.ndarray,
    class_names: np.ndarray,
    eps: float,
    min_points: int,
    min_instance_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    cleaned_instance = instance_ids.copy()
    cleaned_class = class_ids.copy()
    unique_ids = sorted(int(v) for v in np.unique(instance_ids) if int(v) >= 0)

    for instance_id in unique_ids:
        mask = instance_ids == instance_id
        if int(mask.sum()) < min_instance_points:
            cleaned_instance[mask] = -1
            cleaned_class[mask] = -1
            continue

        class_id = int(np.bincount(class_ids[mask][class_ids[mask] >= 0]).argmax()) if np.any(class_ids[mask] >= 0) else -1
        class_name = str(class_names[class_id]) if 0 <= class_id < len(class_names) else "unknown"
        if class_name in STRUCTURAL_CLASSES:
            continue

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(xyz[mask])
        labels = np.asarray(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False), dtype=np.int32)
        valid = labels >= 0
        if not np.any(valid):
            cleaned_instance[mask] = -1
            cleaned_class[mask] = -1
            continue

        label_values, label_counts = np.unique(labels[valid], return_counts=True)
        keep_label = int(label_values[np.argmax(label_counts)])
        drop_mask = mask.copy()
        drop_mask[mask] = labels != keep_label
        cleaned_instance[drop_mask] = -1
        cleaned_class[drop_mask] = -1

    return cleaned_instance, cleaned_class


def colorize_semantic(class_ids: np.ndarray, class_names: np.ndarray) -> np.ndarray:
    palette = (plt.get_cmap("tab20")(np.linspace(0, 1, max(len(class_names), 20)))[:, :3] * 255).astype(np.uint8)
    rgb = np.full((class_ids.shape[0], 3), 180, dtype=np.uint8)
    valid = class_ids >= 0
    if np.any(valid):
        rgb[valid] = palette[class_ids[valid] % len(palette)]
    return rgb


def colorize_instances(instance_ids: np.ndarray) -> np.ndarray:
    unique_ids = sorted(int(v) for v in np.unique(instance_ids) if int(v) >= 0)
    palette = (plt.get_cmap("tab20b")(np.linspace(0, 1, max(len(unique_ids), 20)))[:, :3] * 255).astype(np.uint8)
    rgb = np.full((instance_ids.shape[0], 3), 180, dtype=np.uint8)
    lookup = {obj_id: idx for idx, obj_id in enumerate(unique_ids)}
    for obj_id in unique_ids:
        rgb[instance_ids == obj_id] = palette[lookup[obj_id] % len(palette)]
    return rgb


def write_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, instance_ids: np.ndarray, class_ids: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {xyz.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("property int instance_id\n")
        handle.write("property int class_id\n")
        handle.write("end_header\n")
        for point, color, instance_id, class_id in zip(xyz, rgb, instance_ids, class_ids):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{int(instance_id)} {int(class_id)}\n"
            )


def render_overview(points_xyz: np.ndarray, colors: np.ndarray, labels: list[tuple[str, np.ndarray]], out_path: Path, title: str, max_points: int) -> None:
    if points_xyz.shape[0] == 0:
        return
    if points_xyz.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(points_xyz.shape[0], size=max_points, replace=False))
    else:
        keep = np.arange(points_xyz.shape[0])

    fig, ax = plt.subplots(figsize=(12, 9), dpi=180)
    ax.scatter(points_xyz[keep, 0], points_xyz[keep, 1], s=0.2, c=colors[keep] / 255.0, linewidths=0, alpha=0.95)
    for text, center in labels:
        ax.text(center[0], center[1], text, fontsize=9, color="black",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def summarize_instances(points_xyz: np.ndarray, instance_ids: np.ndarray, class_ids: np.ndarray, class_names: np.ndarray) -> list[tuple[int, int, str, np.ndarray]]:
    stats = []
    for instance_id in sorted(int(v) for v in np.unique(instance_ids) if int(v) >= 0):
        mask = instance_ids == instance_id
        class_mask = class_ids[mask]
        valid_class = class_mask[class_mask >= 0]
        class_id = int(np.bincount(valid_class).argmax()) if valid_class.size > 0 else -1
        class_name = str(class_names[class_id]) if 0 <= class_id < len(class_names) else "unknown"
        stats.append((int(mask.sum()), instance_id, class_name, points_xyz[mask].mean(axis=0)))
    stats.sort(reverse=True)
    return stats


def main() -> None:
    args = parse_args()
    root = repo_root()
    dataset_root = root / "reference" / "OVO" / "data" / "input" / "Datasets" / args.dataset_name / args.scene_name
    run_output = root / "reference" / "OVO" / "data" / "output" / args.dataset_name / args.experiment_name / args.scene_name
    export_run = find_export_run(args.scene_name, args.run_id)

    scene_cfg = load_scene_config(args.scene_name, args.dataset_name)
    keyframes = load_keyframes(export_run, dataset_root)
    snapshot = load_semantic_snapshot(run_output)
    intrinsic = build_intrinsic(scene_cfg["cam"])

    pcd, mesh = integrate_tsdf(
        keyframes,
        intrinsic,
        depth_scale=float(scene_cfg["cam"]["depth_scale"]),
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        depth_trunc=args.depth_trunc,
    )
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if xyz.shape[0] == 0:
        raise RuntimeError("TSDF integration produced an empty point cloud.")

    instance_ids, class_ids, class_names = assign_semantics(
        xyz,
        snapshot,
        semantic_radius=args.semantic_radius,
        semantic_knn=args.semantic_knn,
    )
    instance_ids, class_ids = clean_instances(
        xyz,
        instance_ids,
        class_ids,
        class_names,
        eps=args.instance_dbscan_eps,
        min_points=args.instance_dbscan_min_points,
        min_instance_points=args.instance_min_points,
    )

    semantic_rgb = colorize_semantic(class_ids, class_names)
    instance_rgb = colorize_instances(instance_ids)

    labels = [(f"{class_name}#{instance_id}", center[:2]) for _, instance_id, class_name, center in summarize_instances(xyz, instance_ids, class_ids, class_names)[:12]]

    status = {
        "scene_name": args.scene_name,
        "experiment_name": args.experiment_name,
        "num_keyframes": len(keyframes),
        "num_tsdf_points": int(xyz.shape[0]),
        "num_semantic_instances": int(len([v for v in np.unique(instance_ids) if int(v) >= 0])),
        "source_export_run": str(export_run),
    }
    (run_output / "final_consolidation_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    o3d.io.write_triangle_mesh(str(run_output / f"{args.prefix}_tsdf_mesh.ply"), mesh)
    write_ascii_ply(run_output / f"{args.prefix}_semantic_segments.ply", xyz, semantic_rgb, instance_ids, class_ids)
    write_ascii_ply(run_output / f"{args.prefix}_instance_segments.ply", xyz, instance_rgb, instance_ids, class_ids)

    render_overview(xyz, semantic_rgb, labels, run_output / f"{args.prefix}_semantic_overview.png", f"Semantic Overview: {args.scene_name}", args.max_plot_points)
    render_overview(xyz, instance_rgb, labels, run_output / f"{args.prefix}_instance_overview.png", f"Instance Overview: {args.scene_name}", args.max_plot_points)

    summary_lines = [
        f"scene_name={args.scene_name}",
        f"experiment_name={args.experiment_name}",
        f"num_keyframes={len(keyframes)}",
        f"num_tsdf_points={xyz.shape[0]}",
        f"num_semantic_instances={len([v for v in np.unique(instance_ids) if int(v) >= 0])}",
    ]
    for count, instance_id, class_name, center in summarize_instances(xyz, instance_ids, class_ids, class_names)[:50]:
        summary_lines.append(
            f"instance {instance_id}: class={class_name}, points={count}, center={[round(float(v), 4) for v in center.tolist()]}"
        )
    (run_output / f"{args.prefix}_semantic_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
