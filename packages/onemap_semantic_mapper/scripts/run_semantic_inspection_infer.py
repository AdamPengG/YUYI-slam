#!/usr/bin/env python3

from __future__ import annotations

import argparse
import colorsys
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM+CLIP inspection inference for one keyframe image.")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--ovo-root", required=True)
    parser.add_argument("--ovo-config", required=True)
    parser.add_argument("--dataset-name", default="Replica")
    parser.add_argument("--scene-name", default="inspection")
    parser.add_argument("--class-set", choices=["full", "reduced"], default="full")
    parser.add_argument("--topk-labels", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--abstain-min-score", type=float, default=0.0)
    parser.add_argument("--abstain-min-margin", type=float, default=0.0)
    return parser.parse_args()


def semantic_color(class_id: int) -> np.ndarray:
    if class_id < 0:
        return np.array([180, 180, 180], dtype=np.uint8)
    hue = (class_id * 0.137) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return np.asarray([int(255 * channel) for channel in rgb], dtype=np.uint8)


def overlay_mask(base_rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    if not np.any(mask):
        return base_rgb
    out = base_rgb.astype(np.float32, copy=True)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color.astype(np.float32)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def mask_anchor(binary_mask: np.ndarray, bbox: list[int]) -> tuple[int, int]:
    ys, xs = np.nonzero(binary_mask)
    if xs.size == 0:
        return int((bbox[0] + bbox[2]) // 2), int((bbox[1] + bbox[3]) // 2)
    return int(np.median(xs)), int(np.median(ys))


def load_class_names(ovo_root: Path, dataset_name: str, class_set: str) -> list[str]:
    eval_info_path = ovo_root / "data" / "working" / "configs" / dataset_name / "eval_info.yaml"
    with eval_info_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if class_set == "reduced":
        return list(payload.get("class_names_reduced", payload["class_names"]))
    classes = [str(name) for name in payload["class_names"]]
    return [name for name in classes if name and name != "0"]


def build_text_embeddings(
    clip_generator,
    class_names: list[str],
    templates: list[str],
) -> torch.Tensor:
    text_embeddings: list[torch.Tensor] = []
    for class_name in class_names:
        phrases = [template.format(class_name) for template in templates]
        embed = clip_generator.get_txt_embedding(phrases).mean(0, keepdim=True).float()
        embed = torch.nn.functional.normalize(embed, p=2, dim=-1)
        text_embeddings.append(embed.squeeze(0).cpu())
    return torch.vstack(text_embeddings) if text_embeddings else torch.zeros((0, clip_generator.clip_dim), dtype=torch.float32)


def classify_similarities(
    similarities: torch.Tensor,
    class_names: list[str],
    topk_labels: int,
    abstain_min_score: float,
    abstain_min_margin: float,
) -> list[dict[str, object]]:
    if similarities.numel() == 0:
        return []
    topk = min(topk_labels, similarities.shape[1])
    top_scores, top_indices = torch.topk(similarities, k=topk, dim=1)
    records: list[dict[str, object]] = []
    for mask_idx in range(similarities.shape[0]):
        candidate_labels = [
            class_names[int(class_idx)] for class_idx in top_indices[mask_idx].detach().cpu().tolist()
        ]
        candidate_scores = [float(score) for score in top_scores[mask_idx].detach().cpu().tolist()]
        top_label = candidate_labels[0] if candidate_labels else "unknown"
        second_score = candidate_scores[1] if len(candidate_scores) > 1 else float("-inf")
        margin = (candidate_scores[0] - second_score) if candidate_scores else 0.0
        accepted = bool(candidate_scores) and (
            candidate_scores[0] >= abstain_min_score and margin >= abstain_min_margin
        )
        records.append(
            {
                "semantic_label_candidates": candidate_labels,
                "semantic_scores": candidate_scores,
                "assigned_label": top_label if accepted else "abstain",
                "accepted": accepted,
                "top1_margin": float(margin),
            }
        )
    return records


def render_variant_canvases(
    image_rgb: np.ndarray,
    binary_maps_np: np.ndarray,
    per_mask_records: list[dict[str, object]],
    class_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    clip_canvas = image_rgb.copy()
    sam_clip_canvas = image_rgb.copy()
    for mask_idx, binary_mask in enumerate(binary_maps_np):
        ys, xs = np.nonzero(binary_mask)
        if xs.size == 0 or mask_idx >= len(per_mask_records):
            continue
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        record = per_mask_records[mask_idx]
        assigned_label = str(record.get("assigned_label", "abstain"))
        candidate_scores = [float(score) for score in record.get("semantic_scores", [])]
        accepted = bool(record.get("accepted", False))
        top_label = (
            str(record.get("semantic_label_candidates", ["unknown"])[0])
            if record.get("semantic_label_candidates")
            else "unknown"
        )
        class_id = class_names.index(top_label) if accepted and top_label in class_names else -1
        color = semantic_color(class_id)

        clip_canvas = overlay_mask(clip_canvas, binary_mask, color, alpha=0.42)
        sam_clip_canvas = overlay_mask(sam_clip_canvas, binary_mask, color, alpha=0.40)
        contour_mask = (binary_mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(clip_canvas, contours, -1, color.tolist(), 2)
            cv2.drawContours(sam_clip_canvas, contours, -1, color.tolist(), 1)
        if accepted:
            label_text = top_label if not candidate_scores else f"{top_label} {candidate_scores[0]:.2f}"
        else:
            margin = float(record.get("top1_margin", 0.0))
            label_text = (
                "abstain"
                if not candidate_scores
                else f"abstain {candidate_scores[0]:.2f}/{margin:.2f}"
            )
        anchor_x, anchor_y = mask_anchor(binary_mask, bbox)
        text_y = max(anchor_y - 6, 18)
        cv2.putText(
            clip_canvas,
            label_text,
            (anchor_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color.tolist(),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            sam_clip_canvas,
            label_text,
            (anchor_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color.tolist(),
            2,
            cv2.LINE_AA,
        )
    return clip_canvas, sam_clip_canvas


def absolutize_semantic_config_paths(semantic_cfg: dict, ovo_root: Path) -> dict:
    cfg = json.loads(json.dumps(semantic_cfg))
    sam_cfg = cfg.get("sam", {})
    clip_cfg = cfg.get("clip", {})

    sam_ckpt_path = sam_cfg.get("sam_ckpt_path")
    if sam_ckpt_path:
        sam_path = Path(str(sam_ckpt_path))
        if not sam_path.is_absolute():
            sam_cfg["sam_ckpt_path"] = str((ovo_root / sam_path).resolve())

    masks_base_path = sam_cfg.get("masks_base_path")
    if masks_base_path:
        masks_path = Path(str(masks_base_path))
        if not masks_path.is_absolute():
            sam_cfg["masks_base_path"] = str((ovo_root / masks_path).resolve())

    weights_predictor_path = clip_cfg.get("weights_predictor_path")
    if weights_predictor_path:
        weights_path = Path(str(weights_predictor_path))
        if not weights_path.is_absolute():
            clip_cfg["weights_predictor_path"] = str((ovo_root / weights_path).resolve())
    return cfg


def main() -> None:
    args = parse_args()
    ovo_root = Path(args.ovo_root).expanduser().resolve()
    os.chdir(str(ovo_root))
    if str(ovo_root) not in sys.path:
        sys.path.insert(0, str(ovo_root))

    from ovo.entities.clip_generator import CLIPGenerator
    from ovo.entities.mask_generator import MaskGenerator
    from ovo.utils import clip_utils, segment_utils
    from ovo.utils import io_utils

    image_bgr = cv2.imread(str(args.image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image {args.image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    ovo_config_path = Path(args.ovo_config)
    if not ovo_config_path.is_absolute():
        ovo_config_path = ovo_root / ovo_config_path
    semantic_cfg = absolutize_semantic_config_paths(
        dict(io_utils.load_config(str(ovo_config_path))["semantic"]),
        ovo_root,
    )
    class_names = load_class_names(ovo_root, args.dataset_name, args.class_set)

    mask_generator = MaskGenerator(semantic_cfg["sam"], scene_name=args.scene_name, device=args.device)
    clip_generator = CLIPGenerator(semantic_cfg["clip"], device=args.device)
    templates = semantic_cfg.get("classify_templates", ["This is a photo of a {}"])
    if isinstance(templates, str):
        templates = [templates]
    text_embeddings_tensor = build_text_embeddings(clip_generator, class_names, templates)

    seg_map, binary_maps = mask_generator.get_masks(image_rgb, frame_id=int(args.prefix))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix)

    sam_canvas = image_rgb.copy()
    records: list[dict[str, object]] = []
    textregion_records: list[dict[str, object]] = []
    paper_records: list[dict[str, object]] = []

    binary_maps_np = np.zeros((0, 0, 0), dtype=bool)
    if seg_map.numel() > 0 and binary_maps.numel() > 0:
        image_tensor = torch.from_numpy(image_rgb.transpose((2, 0, 1))).to(args.device)
        textregion_embeds = clip_generator.extract_clip(image_tensor, binary_maps, return_all=False)
        textregion_similarities = clip_generator.get_similarity(
            text_embeddings_tensor.to(textregion_embeds.device, dtype=textregion_embeds.dtype),
            textregion_embeds.to(textregion_embeds.device),
            *clip_generator.similarity_args,
        )
        binary_maps_np = binary_maps.detach().cpu().numpy().astype(bool, copy=False)

        textregion_records = classify_similarities(
            similarities=textregion_similarities,
            class_names=class_names,
            topk_labels=args.topk_labels,
            abstain_min_score=args.abstain_min_score,
            abstain_min_margin=args.abstain_min_margin,
        )

        mask_res = int(semantic_cfg["clip"].get("mask_res", 336))
        seg_imgs = segment_utils.segmap2segimg(binary_maps, image_tensor.squeeze(), True, out_l=mask_res) / 255.0
        paper_records = []
        if seg_imgs.shape[0] > 0:
            clip_g = torch.nn.functional.normalize(clip_generator.encode_image(image_tensor[None] / 255.0), p=2, dim=-1)
            clip_masked = torch.nn.functional.normalize(clip_generator.encode_image(seg_imgs[:, :3]), p=2, dim=-1)
            clip_bbox = torch.nn.functional.normalize(clip_generator.encode_image(seg_imgs[:, 3:]), p=2, dim=-1)
            paper_embeds = clip_utils.fuse_clips(
                clip_g.repeat(seg_imgs.shape[0], 1),
                clip_masked,
                clip_bbox,
                "fixed_weights",
                0.0975,
                0.45,
            )
            paper_similarities = clip_generator.get_similarity(
                text_embeddings_tensor.to(paper_embeds.device, dtype=paper_embeds.dtype),
                paper_embeds.to(paper_embeds.device),
                *clip_generator.similarity_args,
            )
            paper_records = classify_similarities(
                similarities=paper_similarities,
                class_names=class_names,
                topk_labels=args.topk_labels,
                abstain_min_score=args.abstain_min_score,
                abstain_min_margin=args.abstain_min_margin,
            )

        for mask_idx, binary_mask in enumerate(binary_maps_np):
            ys, xs = np.nonzero(binary_mask)
            if xs.size == 0:
                continue
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            sam_record = textregion_records[mask_idx] if mask_idx < len(textregion_records) else {}
            top_label = (
                str(sam_record.get("semantic_label_candidates", ["unknown"])[0])
                if sam_record.get("semantic_label_candidates")
                else "unknown"
            )
            accepted = bool(sam_record.get("accepted", False))
            class_id = class_names.index(top_label) if accepted and top_label in class_names else -1
            color = semantic_color(class_id)
            sam_canvas = overlay_mask(sam_canvas, binary_mask, color, alpha=0.40)
            records.append(
                {
                    "mask_id": int(mask_idx),
                    "bbox_xyxy": bbox,
                    "semantic_label_candidates": sam_record.get("semantic_label_candidates", []),
                    "semantic_scores": sam_record.get("semantic_scores", []),
                    "assigned_label": sam_record.get("assigned_label", "abstain"),
                    "accepted": bool(sam_record.get("accepted", False)),
                    "top1_margin": float(sam_record.get("top1_margin", 0.0)),
                    "variants": {
                        "textregion": dict(sam_record),
                        "paper_fusion": dict(paper_records[mask_idx]) if mask_idx < len(paper_records) else {},
                    },
                }
            )

        np.savez_compressed(output_dir / f"{prefix}_masks.npz", binary_masks=binary_maps_np.astype(np.uint8))
    else:
        np.savez_compressed(output_dir / f"{prefix}_masks.npz", binary_masks=np.zeros((0, 0, 0), dtype=np.uint8))

    cv2.imwrite(str(output_dir / f"{prefix}_sam.png"), cv2.cvtColor(sam_canvas, cv2.COLOR_RGB2BGR))
    textregion_clip_canvas, textregion_sam_clip_canvas = render_variant_canvases(
        image_rgb=image_rgb,
        binary_maps_np=binary_maps_np,
        per_mask_records=textregion_records,
        class_names=class_names,
    )
    paper_clip_canvas, paper_sam_clip_canvas = render_variant_canvases(
        image_rgb=image_rgb,
        binary_maps_np=binary_maps_np,
        per_mask_records=paper_records,
        class_names=class_names,
    )
    cv2.imwrite(str(output_dir / f"{prefix}_clip.png"), cv2.cvtColor(textregion_clip_canvas, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f"{prefix}_sam_clip.png"), cv2.cvtColor(textregion_sam_clip_canvas, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f"{prefix}_clip_textregion.png"), cv2.cvtColor(textregion_clip_canvas, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f"{prefix}_sam_clip_textregion.png"), cv2.cvtColor(textregion_sam_clip_canvas, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f"{prefix}_clip_paper.png"), cv2.cvtColor(paper_clip_canvas, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f"{prefix}_sam_clip_paper.png"), cv2.cvtColor(paper_sam_clip_canvas, cv2.COLOR_RGB2BGR))
    (output_dir / f"{prefix}_masks.json").write_text(
        json.dumps({"masks": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"prefix": prefix, "mask_count": len(records)}))


if __name__ == "__main__":
    main()
