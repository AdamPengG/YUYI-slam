#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

OVO_ROOT = Path("/home/peng/isacc_slam/reference/OVO")
if str(OVO_ROOT) not in sys.path:
    sys.path.insert(0, str(OVO_ROOT))

from run_eval import load_representation
from ovo.entities.ovo import OVO
from ovo.utils import io_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an OVO checkpoint as compressed semantic arrays.")
    parser.add_argument("run_path", help="Path to finished or intermediate OVO run directory")
    parser.add_argument("--ckpt", default="ovo_map.ckpt")
    parser.add_argument("--output-npz", default=None)
    parser.add_argument("--output-summary", default=None)
    parser.add_argument("--mode", choices=["instance", "semantic"], default="semantic")
    return parser.parse_args()


def load_classes(run_path: Path) -> list[str]:
    config_path = run_path.parents[3] / "working" / "configs" / "Replica" / "eval_info.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        info = yaml.safe_load(handle)
    return info.get("class_names_reduced", info["class_names"])


def load_representation_from_ckpt(run_path: Path, ckpt_name: str):
    if ckpt_name == "ovo_map.ckpt":
        return load_representation(run_path, eval=True)

    config = io_utils.load_config(run_path / "config.yaml", inherit=False)
    ckpt_path = run_path / ckpt_name
    submap_ckpt = torch.load(ckpt_path, weights_only=False)
    map_params = submap_ckpt.get("map_params", None)
    if map_params is None:
        map_params = submap_ckpt["gaussian_params"]
    config["semantic"]["verbose"] = False
    ovo = OVO(config["semantic"], None, config["data"]["scene_name"], eval=True, device=config.get("device", "cuda"))
    ovo.restore_dict(submap_ckpt["ovo_map_params"], debug_info=False)
    return ovo, map_params


def build_obj_to_class(ovo, classes: list[str]) -> tuple[dict[int, int], str | None]:
    try:
        instances_info = ovo.classify_instances(classes)
        object_ids = list(ovo.objects.keys())
        return {
            int(obj_id): int(instances_info["classes"][idx])
            for idx, obj_id in enumerate(object_ids)
        }, None
    except Exception as exc:
        return {}, str(exc)


def main() -> None:
    args = parse_args()
    run_path = Path(args.run_path).expanduser().resolve()
    output_npz = Path(args.output_npz).expanduser().resolve() if args.output_npz else run_path / "semantic_snapshot.npz"
    output_summary = Path(args.output_summary).expanduser().resolve() if args.output_summary else run_path / "semantic_snapshot_summary.txt"

    ovo, map_params = load_representation_from_ckpt(run_path, args.ckpt)
    classes = load_classes(run_path)
    obj_to_class, classify_error = build_obj_to_class(ovo, classes)

    xyz = map_params["xyz"].detach().cpu().numpy().astype(np.float32)
    obj_ids = map_params["obj_ids"].squeeze(-1).detach().cpu().numpy().astype(np.int32)

    unique_obj_ids = sorted(int(v) for v in np.unique(obj_ids) if int(v) >= 0)
    inst_palette = (plt.get_cmap("tab20b")(np.linspace(0, 1, max(20, len(unique_obj_ids) + 1)))[:, :3] * 255).astype(np.uint8)
    sem_palette = (plt.get_cmap("tab20")(np.linspace(0, 1, max(20, len(classes) + 1)))[:, :3] * 255).astype(np.uint8)

    rgb = np.full((xyz.shape[0], 3), 180, dtype=np.uint8)
    class_ids = np.full((xyz.shape[0],), -1, dtype=np.int32)

    obj_to_color_index = {obj_id: idx for idx, obj_id in enumerate(unique_obj_ids)}
    instance_centers = []
    instance_labels = []
    instance_ids = []

    for obj_id in unique_obj_ids:
        mask = obj_ids == obj_id
        class_id = int(obj_to_class.get(obj_id, -1))
        class_ids[mask] = class_id
        if args.mode == "semantic" and class_id >= 0:
            rgb[mask] = sem_palette[class_id % len(sem_palette)]
        else:
            rgb[mask] = inst_palette[obj_to_color_index[obj_id] % len(inst_palette)]

        center = xyz[mask].mean(axis=0).astype(np.float32)
        class_name = classes[class_id] if 0 <= class_id < len(classes) else "unknown"
        instance_centers.append(center)
        instance_labels.append(class_name)
        instance_ids.append(obj_id)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz_path = output_npz.with_name(output_npz.name + ".tmp.npz")
    if tmp_npz_path.exists():
        tmp_npz_path.unlink()
    np.savez_compressed(
        tmp_npz_path,
        xyz=xyz,
        rgb=rgb,
        instance_id=obj_ids,
        class_id=class_ids,
        instance_ids=np.asarray(instance_ids, dtype=np.int32),
        instance_centers=np.asarray(instance_centers, dtype=np.float32) if instance_centers else np.zeros((0, 3), dtype=np.float32),
        instance_labels=np.asarray(instance_labels, dtype=object),
        class_names=np.asarray(classes, dtype=object),
        classify_error=np.asarray("" if classify_error is None else classify_error, dtype=object),
        mode=np.asarray(args.mode, dtype=object),
    )
    tmp_npz_path.replace(output_npz)

    lines = [
        f"run_path={run_path}",
        f"ckpt={args.ckpt}",
        f"num_points={xyz.shape[0]}",
        f"num_instances={len(unique_obj_ids)}",
        f"mode={args.mode}",
    ]
    if classify_error is not None:
        lines.append(f"classification_warning={classify_error}")

    for obj_id, class_name, center in zip(instance_ids[:50], instance_labels[:50], instance_centers[:50]):
        mask = obj_ids == obj_id
        lines.append(
            f"instance {obj_id}: class={class_name}, points={int(mask.sum())}, "
            f"center={[round(float(v), 4) for v in center.tolist()]}"
        )
    output_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {output_npz}")
    print(f"wrote {output_summary}")


if __name__ == "__main__":
    main()
