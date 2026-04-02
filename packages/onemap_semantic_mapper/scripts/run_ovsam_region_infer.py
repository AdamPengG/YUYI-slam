#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ext.templates import VILD_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OVSAM region inference helper.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--weights-root", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--proposals-json", required=True)
    parser.add_argument("--proposal-masks-npz", required=True)
    parser.add_argument("--class-names-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-masks-npz", required=True)
    parser.add_argument("--mode", default="prompt_region")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--score-thresh", type=float, default=0.10)
    parser.add_argument("--use-mask-bbox-as-prompt", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _fail(message: str) -> int:
    sys.stderr.write(message.rstrip() + "\n")
    return 1


def _log(message: str) -> None:
    sys.stderr.write(message.rstrip() + "\n")
    sys.stderr.flush()


def _bootstrap_repo(repo_root: Path) -> None:
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    # Import repo packages early so custom modules are registered into MMDet/MMEngine registries.
    __import__("seg")
    __import__("ext")


def _canonical_class_name(raw_name: str) -> str:
    tokens = [token.strip() for token in str(raw_name).split(",")]
    for token in tokens:
        if token:
            return token
    return str(raw_name).strip() or "unknown"


def _split_label_variants(raw_name: str) -> list[str]:
    tokens = (
        str(raw_name)
        .replace("_or_", ",")
        .replace("/", ",")
        .replace("_", " ")
        .lower()
        .split(",")
    )
    cleaned = [token.strip() for token in tokens if token.strip()]
    return cleaned or [_canonical_class_name(raw_name).lower()]


def _build_class_names(cfg: Any) -> list[str]:
    from mmdet.registry import DATASETS

    dataset_cfg = copy.deepcopy(cfg.train_dataloader.dataset)
    dataset_cfg.update(lazy_init=True)
    dataset = DATASETS.build(dataset_cfg)
    raw_names = list(dataset.metainfo["classes"])
    return [_canonical_class_name(name) for name in raw_names]


def _load_runtime_class_names(class_names_json: Path | None) -> list[str] | None:
    if class_names_json is None:
        return None
    payload = json.loads(class_names_json.read_text(encoding="utf-8"))
    class_names = [str(name).strip() for name in payload.get("class_names", [])]
    class_names = [name for name in class_names if name]
    return class_names or None


def _resolve_checkpoint_paths(obj: Any, weights_root: Path) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "checkpoint" and isinstance(value, str) and value.startswith("./models/"):
                obj[key] = str((weights_root / Path(value).name).resolve())
            else:
                _resolve_checkpoint_paths(value, weights_root)
    elif isinstance(obj, list):
        for item in obj:
            _resolve_checkpoint_paths(item, weights_root)


def _load_inputs(
    image_path: Path,
    proposals_path: Path,
    masks_path: Path,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    payload = json.loads(proposals_path.read_text(encoding="utf-8"))
    proposals = list(payload.get("proposals", []))
    with np.load(masks_path, allow_pickle=False) as data:
        proposal_masks = data["masks"].astype(bool, copy=False)
    return image_rgb, proposals, proposal_masks


def _resize_inputs_for_config(
    image_rgb: np.ndarray,
    proposals: list[dict[str, Any]],
    proposal_masks: np.ndarray,
    image_scale: tuple[int, int],
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    orig_h, orig_w = image_rgb.shape[:2]
    target_w, target_h = int(image_scale[0]), int(image_scale[1])
    if target_w <= 0 or target_h <= 0:
        raise RuntimeError(f"Invalid image scale: {image_scale}")

    scale = min(float(target_w) / float(orig_w), float(target_h) / float(orig_h))
    resize_w = max(1, int(round(orig_w * scale)))
    resize_h = max(1, int(round(orig_h * scale)))

    resized_image = cv2.resize(image_rgb, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    resized_masks = np.stack(
        [
            cv2.resize(mask.astype(np.uint8), (resize_w, resize_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            for mask in proposal_masks
        ],
        axis=0,
    ) if proposal_masks.size else np.zeros((0, resize_h, resize_w), dtype=bool)

    resized_proposals: list[dict[str, Any]] = []
    for item in proposals:
        resized_item = dict(item)
        bbox_xyxy = item.get("bbox_xyxy")
        if bbox_xyxy:
            x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
            resized_item["bbox_xyxy"] = [
                int(round(x1 * scale)),
                int(round(y1 * scale)),
                int(round(x2 * scale)),
                int(round(y2 * scale)),
            ]
            resized_item["original_bbox_xyxy"] = [int(round(v)) for v in bbox_xyxy]
        centroid_xy = item.get("centroid_xy")
        if centroid_xy:
            resized_item["centroid_xy"] = [float(centroid_xy[0]) * scale, float(centroid_xy[1]) * scale]
        resized_item["resize_scale_factor"] = [float(scale), float(scale)]
        resized_proposals.append(resized_item)

    meta = {
        "orig_shape": (orig_h, orig_w),
        "img_shape": (resize_h, resize_w),
        "scale_factor": (float(scale), float(scale)),
    }
    return resized_image, resized_proposals, resized_masks, meta


def _prepare_det_sample(
    image_rgb: np.ndarray,
    proposals: list[dict[str, Any]],
    proposal_masks: np.ndarray,
    meta: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample
    from mmdet.structures.mask import BitmapMasks

    height, width = image_rgb.shape[:2]
    valid_proposals: list[dict[str, Any]] = []
    bbox_list: list[list[float]] = []
    mask_list: list[np.ndarray] = []

    for item in proposals:
        proposal_id = int(item.get("proposal_id", item.get("mask_id", -1)))
        if proposal_id < 0 or proposal_id >= int(proposal_masks.shape[0]):
            continue
        binary_mask = proposal_masks[proposal_id]
        if binary_mask.ndim != 2 or not bool(binary_mask.any()):
            continue
        bbox_xyxy = item.get("bbox_xyxy")
        if not bbox_xyxy:
            ys, xs = np.nonzero(binary_mask)
            bbox_xyxy = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        valid_item = dict(item)
        valid_item["proposal_id"] = proposal_id
        valid_item["bbox_xyxy"] = [int(v) for v in bbox_xyxy]
        valid_proposals.append(valid_item)
        bbox_list.append([float(v) for v in bbox_xyxy])
        mask_list.append(binary_mask.astype(np.uint8, copy=False))

    if not valid_proposals:
        return {"inputs": [], "data_samples": []}, []

    gt_instances = InstanceData()
    gt_instances.labels = torch.zeros((len(valid_proposals),), dtype=torch.long)
    gt_instances.bboxes = torch.tensor(bbox_list, dtype=torch.float32)
    gt_instances.masks = BitmapMasks(np.stack(mask_list, axis=0), height=height, width=width)

    data_sample = DetDataSample()
    data_sample.set_metainfo(
        {
            "img_shape": tuple(meta["img_shape"]),
            "ori_shape": tuple(meta["orig_shape"]),
            "scale_factor": tuple(meta["scale_factor"]),
        }
    )
    data_sample.gt_instances = gt_instances

    image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float()
    return {"inputs": [image_tensor], "data_samples": [data_sample]}, valid_proposals


def _build_model(config_path: Path, weights_root: Path, checkpoint_path: Path | None, device: str):
    from mmengine.config import Config
    from mmdet.registry import MODELS

    cfg = Config.fromfile(str(config_path))
    _resolve_checkpoint_paths(cfg.model, weights_root)
    if checkpoint_path is not None:
        cfg.load_from = str(checkpoint_path)
    model = MODELS.build(cfg.model)
    # In the official OVSAM tools, weights are initialized explicitly by the runner.
    # For this standalone helper we need to do it ourselves.
    model.init_weights()
    model.to(device)
    model.eval()
    return cfg, model


def _build_custom_classifier(
    model: Any,
    class_names: list[str],
    device: str,
) -> torch.Tensor:
    text_model = model.backbone.get_text_model().to(device)
    text_model.eval()

    descriptions: list[str] = []
    candidates_per_class: list[int] = []
    for class_name in class_names:
        label_variants = _split_label_variants(class_name)
        candidates_per_class.append(len(label_variants))
        for label_variant in label_variants:
            for template in VILD_PROMPT:
                descriptions.append(template.format(label_variant))

    num_batch = 256
    features: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(descriptions), num_batch):
            batch = descriptions[start : start + num_batch]
            if not batch:
                continue
            feat = text_model(batch).to(device="cpu")
            features.append(feat)
    if not features:
        raise RuntimeError("No text features were generated for runtime OVSAM classifier.")

    feature_tensor = torch.cat(features, dim=0)
    dim = int(feature_tensor.shape[-1])
    candidate_tot = sum(candidates_per_class)
    candidate_max = max(candidates_per_class)
    feature_tensor = feature_tensor.reshape(candidate_tot, len(VILD_PROMPT), dim)
    feature_tensor = feature_tensor / feature_tensor.norm(dim=-1, keepdim=True)
    feature_tensor = feature_tensor.mean(dim=1, keepdim=False)
    feature_tensor = feature_tensor / feature_tensor.norm(dim=-1, keepdim=True)

    classifier = []
    cur_pos = 0
    for candidate in candidates_per_class:
        cur_feat = feature_tensor[cur_pos : cur_pos + candidate]
        if candidate < candidate_max:
            cur_feat = torch.cat([cur_feat, cur_feat[0].repeat(candidate_max - candidate, 1)], dim=0)
        classifier.append(cur_feat)
        cur_pos += candidate
    classifier = torch.stack(classifier, dim=0)
    return classifier


def _override_runtime_classifier(
    model: Any,
    class_names: list[str],
    device: str,
) -> None:
    classifier = _build_custom_classifier(model, class_names, device=device)
    cls_embed = classifier.to(device=device, dtype=torch.float32)
    dim = int(cls_embed.shape[-1])
    prototypes = int(cls_embed.shape[1])
    background = torch.zeros((1, prototypes, dim), dtype=torch.float32, device=device)
    cls_embed = torch.cat([cls_embed, background], dim=0)
    model.mask_decoder.cls_embed = cls_embed.permute(2, 0, 1).contiguous()
    model.num_classes = len(class_names)
    model.base_novel_indicator = None


def _predict_regions(model: Any, data: dict[str, Any], topk: int) -> tuple[np.ndarray, np.ndarray]:
    from seg.models.detectors.ovsam import postprocess_masks

    with torch.no_grad():
        packed = model.data_preprocessor(data, training=False)
        batch_inputs = packed["inputs"]
        batch_data_samples = packed["data_samples"]
        assert len(batch_data_samples) == 1, "OVSAM helper only supports batch size 1."
        data_sample = batch_data_samples[0]

        backbone_feat = model.backbone(batch_inputs)
        batch_feats = model.neck(backbone_feat)
        feat = batch_feats[0]
        fpn_feats = model.fpn_neck(backbone_feat) if model.fpn_neck is not None else None
        prompt_instances = data_sample.gt_instances
        if model.use_point:
            sparse_embed, dense_embed = model.pe(
                prompt_instances,
                image_size=data_sample.batch_input_shape,
                with_points=True,
            )
        else:
            sparse_embed, dense_embed = model.pe(
                prompt_instances,
                image_size=data_sample.batch_input_shape,
                with_bboxes=True,
            )

        if fpn_feats is not None:
            low_res_masks, _, cls_pred = model.mask_decoder(
                image_embeddings=feat.unsqueeze(0),
                image_pe=model.pe.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                multi_mask_output=False,
                fpn_feats=[item[0:1] for item in fpn_feats],
                data_samples=data_sample,
            )
        else:
            low_res_masks, _, cls_pred = model.mask_decoder(
                image_embeddings=feat.unsqueeze(0),
                image_pe=model.pe.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                multi_mask_output=False,
                data_samples=data_sample,
            )

        cls_pred = model.open_voc_inference(backbone_feat, cls_pred, low_res_masks, data_samples=data_sample)
        probs = cls_pred[:, 0].softmax(-1)[:, :-1]
        k = max(1, min(int(topk), int(probs.shape[1])))
        top_scores, top_indices = torch.topk(probs, k=k, dim=1)

        masks = postprocess_masks(
            masks=low_res_masks,
            pad_size=data_sample.batch_input_shape,
            input_size=data_sample.img_shape,
            original_size=data_sample.ori_shape,
        )
        binary_masks = (masks.sigmoid() > model.MASK_THRESHOLD)[:, 0]
        return (
            binary_masks.detach().cpu().numpy().astype(bool, copy=False),
            torch.stack([top_indices, top_scores], dim=0).detach().cpu().numpy(),
        )


def _restore_masks_to_original_shape(
    binary_masks: np.ndarray,
    orig_shape: tuple[int, int],
) -> np.ndarray:
    if binary_masks.size == 0:
        return np.zeros((0, orig_shape[0], orig_shape[1]), dtype=bool)
    restored = []
    for mask in binary_masks:
        restored_mask = cv2.resize(
            mask.astype(np.uint8),
            (int(orig_shape[1]), int(orig_shape[0])),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        restored.append(restored_mask)
    return np.stack(restored, axis=0)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    weights_root = Path(args.weights_root).expanduser().resolve()
    config_path = Path(args.config_path).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve() if args.checkpoint_path else None
    image_path = Path(args.image_path).expanduser().resolve()
    proposals_path = Path(args.proposals_json).expanduser().resolve()
    proposal_masks_path = Path(args.proposal_masks_npz).expanduser().resolve()
    class_names_json = Path(args.class_names_json).expanduser().resolve() if args.class_names_json else None
    output_json = Path(args.output_json).expanduser().resolve()
    output_masks = Path(args.output_masks_npz).expanduser().resolve()

    if not repo_root.exists():
        return _fail(f"OVSAM repo_root does not exist: {repo_root}")
    if not weights_root.exists():
        return _fail(f"OVSAM weights_root does not exist: {weights_root}")
    if not config_path.exists():
        return _fail(f"OVSAM config file does not exist: {config_path}")
    if checkpoint_path is not None and not checkpoint_path.exists():
        return _fail(f"OVSAM checkpoint file does not exist: {checkpoint_path}")
    if not image_path.exists():
        return _fail(f"Input image does not exist: {image_path}")
    if not proposals_path.exists():
        return _fail(f"Proposals json does not exist: {proposals_path}")
    if not proposal_masks_path.exists():
        return _fail(f"Proposal masks npz does not exist: {proposal_masks_path}")
    if class_names_json is not None and not class_names_json.exists():
        return _fail(f"Runtime class names json does not exist: {class_names_json}")

    _bootstrap_repo(repo_root)

    try:
        _log("[ovsam] building model")
        cfg, model = _build_model(config_path, weights_root, checkpoint_path, args.device)
        runtime_class_names = _load_runtime_class_names(class_names_json)
        if runtime_class_names:
            _log("[ovsam] overriding classifier")
            _override_runtime_classifier(model, runtime_class_names, device=args.device)
        image_scale = tuple(int(v) for v in cfg.test_dataloader.dataset.pipeline[1].scale)
        _log("[ovsam] loading inputs")
        image_rgb, proposals, proposal_masks = _load_inputs(image_path, proposals_path, proposal_masks_path)
        if not proposals:
            output_json.write_text(json.dumps({"status": "ok", "results": []}, ensure_ascii=False, indent=2), encoding="utf-8")
            np.savez_compressed(output_masks, masks=np.zeros((0, 0, 0), dtype=bool))
            return 0

        resized_image, resized_proposals, resized_masks, meta = _resize_inputs_for_config(
            image_rgb,
            proposals,
            proposal_masks,
            image_scale=image_scale,
        )
        _log("[ovsam] preparing proposals")
        data, valid_proposals = _prepare_det_sample(resized_image, resized_proposals, resized_masks, meta=meta)
        if not valid_proposals:
            output_json.write_text(json.dumps({"status": "ok", "results": []}, ensure_ascii=False, indent=2), encoding="utf-8")
            np.savez_compressed(output_masks, masks=np.zeros((0, 0, 0), dtype=bool))
            return 0

        _log("[ovsam] building class names")
        class_names = runtime_class_names or _build_class_names(cfg)
        _log("[ovsam] running prediction")
        refined_masks, topk_pack = _predict_regions(model, data, args.topk)
        refined_masks = _restore_masks_to_original_shape(refined_masks, tuple(meta["orig_shape"]))
        top_indices = topk_pack[0].astype(np.int64, copy=False)
        top_scores = topk_pack[1].astype(np.float32, copy=False)

        _log("[ovsam] serializing outputs")
        results: list[dict[str, Any]] = []
        for idx, proposal in enumerate(valid_proposals):
            label_candidates: list[str] = []
            semantic_scores: list[float] = []
            for class_idx, score in zip(top_indices[idx].tolist(), top_scores[idx].tolist(), strict=False):
                if int(class_idx) < 0 or int(class_idx) >= len(class_names):
                    continue
                if float(score) < float(args.score_thresh):
                    continue
                label_candidates.append(class_names[int(class_idx)])
                semantic_scores.append(float(score))
            results.append(
                {
                    "proposal_id": int(proposal["proposal_id"]),
                    "mask_id": int(proposal.get("mask_id", proposal["proposal_id"])),
                    "bbox_xyxy": [int(v) for v in proposal.get("original_bbox_xyxy", proposal["bbox_xyxy"])],
                    "semantic_label_candidates": label_candidates,
                    "semantic_scores": semantic_scores,
                }
            )

        output_json.write_text(
            json.dumps({"status": "ok", "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.savez_compressed(output_masks, masks=refined_masks.astype(bool, copy=False))
        _log("[ovsam] done")
        return 0
    except Exception as exc:  # pragma: no cover - runtime integration path
        tb = traceback.format_exc()
        return _fail(f"OVSAM helper runtime failed: {exc}\n{tb}")


if __name__ == "__main__":
    raise SystemExit(main())
