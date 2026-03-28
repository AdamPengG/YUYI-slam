from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from run_eval import load_representation


def _load_classes(config_path: Path) -> list[str]:
    with config_path.open("r", encoding="utf-8") as handle:
        info = yaml.safe_load(handle)
    return info.get("class_names_reduced", info["class_names"])


def _get_rgb_colors(map_params: dict) -> np.ndarray:
    sh_c0 = 0.28209479177387814
    if map_params.get("features_dc", None) is not None:
        colors = (map_params["features_dc"] * sh_c0 + 0.5).clip(0, 1).flatten(0, 1)
        return colors.detach().cpu().numpy()
    if map_params.get("color", None) is not None:
        color = map_params["color"]
        if color.max() > 1.0:
            color = color / 255.0
        return color.detach().cpu().numpy()
    return np.ones((map_params["xyz"].shape[0], 3), dtype=np.float32) * 0.8


def _sample_indices(n_points: int, max_points: int) -> np.ndarray:
    if n_points <= max_points:
        return np.arange(n_points)
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(n_points, size=max_points, replace=False))


def _plot_points(xy: np.ndarray, colors: np.ndarray, out_path: Path, title: str, labels: list[tuple[str, np.ndarray]] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 9), dpi=180)
    ax.scatter(xy[:, 0], xy[:, 1], s=0.2, c=colors, linewidths=0, alpha=0.95)
    if labels:
        for text, center in labels:
            ax.text(center[0], center[1], text, fontsize=9, color="black",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_run_overview(run_path: str | Path, max_points: int = 250000, prefix: str = "demo") -> None:
    run_path = Path(run_path).expanduser().resolve()
    ovo, map_params = load_representation(run_path, eval=True)

    classes = _load_classes(run_path.parents[3] / "working" / "configs" / "Replica" / "eval_info.yaml")
    object_ids = list(ovo.objects.keys())
    classify_error = None
    obj_to_class: dict[int, int] = {}
    try:
        instances_info = ovo.classify_instances(classes)
        obj_to_class = {
            int(obj_id): int(instances_info["classes"][idx])
            for idx, obj_id in enumerate(object_ids)
        }
    except Exception as exc:
        classify_error = str(exc)

    points = map_params["xyz"].detach().cpu().numpy()
    obj_ids = map_params["obj_ids"].squeeze(-1).detach().cpu().numpy().astype(np.int64)
    rgb_colors = _get_rgb_colors(map_params)

    class_palette = plt.get_cmap("tab20")(np.linspace(0, 1, max(20, len(classes))))[:, :3]
    instance_palette = plt.get_cmap("tab20b")(np.linspace(0, 1, 20))[:, :3]

    semantic_colors = np.full((points.shape[0], 3), 0.75, dtype=np.float32)
    instance_colors = np.full((points.shape[0], 3), 0.75, dtype=np.float32)

    labels_to_draw: list[tuple[str, np.ndarray]] = []
    summary_lines = [
        f"num_points={points.shape[0]}",
        f"num_instances={len(object_ids)}",
    ]
    if classify_error is not None:
        summary_lines.append(f"classification_warning={classify_error}")

    instance_stats = []
    for obj_id in object_ids:
        mask = obj_ids == obj_id
        if not np.any(mask):
            continue
        class_idx = obj_to_class.get(int(obj_id), -1)
        class_name = classes[class_idx] if 0 <= class_idx < len(classes) else "unknown"
        center = points[mask].mean(axis=0)
        instance_stats.append((mask.sum(), int(obj_id), class_name, center))
        semantic_colors[mask] = class_palette[class_idx % len(class_palette)] if class_idx >= 0 else np.array([0.5, 0.5, 0.5])
        instance_colors[mask] = instance_palette[int(obj_id) % len(instance_palette)]

    instance_stats.sort(reverse=True)
    for count, obj_id, class_name, center in instance_stats[:12]:
        labels_to_draw.append((f"{class_name}#{obj_id}", center[:2]))
        summary_lines.append(
            f"instance {obj_id}: class={class_name}, points={count}, center={[round(float(v), 4) for v in center.tolist()]}"
        )

    idx = _sample_indices(points.shape[0], max_points)
    points_xy = points[idx, :2]
    _plot_points(points_xy, instance_colors[idx], run_path / f"{prefix}_instance_overview.png", f"Instance Overview: {run_path.name}", labels_to_draw)
    _plot_points(points_xy, semantic_colors[idx], run_path / f"{prefix}_semantic_overview.png", f"Semantic Overview: {run_path.name}", labels_to_draw)
    _plot_points(points_xy, rgb_colors[idx], run_path / f"{prefix}_rgb_overview.png", f"RGB Overview: {run_path.name}", labels_to_draw)

    with (run_path / f"{prefix}_semantic_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render static overview images and a summary for an OVO run.")
    parser.add_argument("run_path", help="Path to a finished OVO run directory.")
    parser.add_argument("--max-points", type=int, default=250000)
    parser.add_argument("--prefix", default="demo")
    args = parser.parse_args()

    render_run_overview(args.run_path, max_points=args.max_points, prefix=args.prefix)


if __name__ == "__main__":
    main()
