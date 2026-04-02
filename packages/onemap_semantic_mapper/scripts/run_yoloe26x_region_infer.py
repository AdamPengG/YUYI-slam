#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOE-26X region inference helper.")
    parser.add_argument("--image-path", default="")
    parser.add_argument("--class-names-json", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-masks-npz", default="")
    parser.add_argument("--model-path", default="yoloe-26x-seg.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf-thresh", type=float, default=0.10)
    parser.add_argument("--iou-thresh", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--server", action="store_true", help="Run as a persistent stdin/stdout JSONL worker.")
    return parser.parse_args()


def _log(message: str) -> None:
    sys.stderr.write(message.rstrip() + "\n")
    sys.stderr.flush()


def _load_classes(class_names_json: Path) -> list[str]:
    payload = json.loads(class_names_json.read_text(encoding="utf-8"))
    class_names = [str(name).strip() for name in payload.get("class_names", [])]
    return [name for name in class_names if name]


def _decode_inline_image_bgr(request: dict[str, object]) -> np.ndarray | None:
    image_b64 = request.get("image_png_base64")
    if not image_b64:
        return None
    raw = base64.b64decode(str(image_b64))
    image_np = np.frombuffer(raw, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("Failed to decode inline PNG image for YOLOE helper.")
    return image_bgr


def _pack_masks_inline(masks: np.ndarray) -> tuple[list[int], str]:
    if masks.size == 0:
        return [0, 0, 0], ""
    bool_masks = np.asarray(masks, dtype=bool, order="C")
    packed = np.packbits(bool_masks.reshape(-1).astype(np.uint8), bitorder="little")
    return list(bool_masks.shape), base64.b64encode(packed.tobytes()).decode("ascii")


def _write_outputs(
    output_json: Path,
    output_masks: Path,
    class_names: list[str],
    records: list[dict[str, object]],
    masks: np.ndarray,
) -> None:
    payload = {"results": records, "class_names": class_names}
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.savez_compressed(output_masks, masks=masks)


def _import_yoloe(repo_root: str | None):
    try:
        from ultralytics import YOLOE  # type: ignore

        return YOLOE
    except ModuleNotFoundError:
        if not repo_root:
            raise
        repo_root_resolved = str(Path(repo_root).expanduser().resolve())
        if repo_root_resolved not in sys.path:
            sys.path.insert(0, repo_root_resolved)
        from ultralytics import YOLOE  # type: ignore

        return YOLOE


def _serialize_results(result) -> tuple[list[dict[str, object]], np.ndarray]:
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return [], np.zeros((0, 0, 0), dtype=bool)

    masks = result.masks.data.detach().cpu().numpy().astype(bool, copy=False)
    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32, copy=False)
    names = dict(result.names)

    count = min(len(boxes_xyxy), masks.shape[0], len(confs), len(classes))
    records: list[dict[str, object]] = []
    kept_masks: list[np.ndarray] = []
    for idx in range(count):
        class_id = int(classes[idx])
        class_name = str(names.get(class_id, class_id))
        score = float(confs[idx])
        bbox = [int(round(v)) for v in boxes_xyxy[idx].tolist()]
        mask = masks[idx]
        if mask.ndim != 2 or not bool(mask.any()):
            continue
        records.append(
            {
                "mask_id": int(idx),
                "bbox_xyxy": bbox,
                "semantic_label_candidates": [class_name],
                "semantic_scores": [score],
            }
        )
        kept_masks.append(mask)

    if not kept_masks:
        return [], np.zeros((0, 0, 0), dtype=bool)
    return records, np.stack(kept_masks, axis=0).astype(bool, copy=False)


class YOLOEServer:
    def __init__(self, args: argparse.Namespace) -> None:
        YOLOE = _import_yoloe(args.repo_root)
        _log("[yoloe] building persistent model")
        self.model = YOLOE(args.model_path)
        self.args = args
        self._current_classes: list[str] = []

    def _ensure_classes(self, class_names: list[str]) -> None:
        if class_names != self._current_classes:
            _log(f"[yoloe] updating classes ({len(class_names)})")
            self.model.set_classes(class_names)
            self._current_classes = list(class_names)

    def infer(self, image_source, class_names: list[str]) -> tuple[list[dict[str, object]], np.ndarray]:
        self._ensure_classes(class_names)
        results = self.model.predict(
            source=image_source,
            device=self.args.device,
            conf=float(self.args.conf_thresh),
            iou=float(self.args.iou_thresh),
            max_det=int(self.args.max_det),
            verbose=False,
            retina_masks=True,
        )
        if not results:
            return [], np.zeros((0, 0, 0), dtype=bool)
        return _serialize_results(results[0])


def _run_server(args: argparse.Namespace) -> int:
    try:
        server = YOLOEServer(args)
        sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
        sys.stdout.flush()
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                if request.get("command") == "shutdown":
                    sys.stdout.write(json.dumps({"status": "bye"}) + "\n")
                    sys.stdout.flush()
                    return 0
                inline_image = _decode_inline_image_bgr(request)
                if inline_image is None:
                    image_path = Path(str(request["image_path"])).expanduser().resolve()
                    image_source = str(image_path)
                else:
                    image_source = inline_image
                raw_class_names = request.get("class_names")
                if isinstance(raw_class_names, list):
                    class_names = [str(name).strip() for name in raw_class_names if str(name).strip()]
                else:
                    class_names_json = Path(str(request["class_names_json"])).expanduser().resolve()
                    class_names = _load_classes(class_names_json)
                if not class_names:
                    raise RuntimeError("No class names provided to YOLOE helper.")
                records, masks = server.infer(image_source, class_names)
                return_inline = bool(request.get("return_inline", False))
                include_masks = bool(request.get("include_masks", True))
                if return_inline:
                    response: dict[str, object] = {
                        "status": "ok",
                        "num_results": int(len(records)),
                        "results": records,
                    }
                    if include_masks:
                        shape, packed = _pack_masks_inline(masks)
                        response["mask_shape"] = shape
                        response["masks_packbits_base64"] = packed
                    sys.stdout.write(json.dumps(response) + "\n")
                else:
                    output_json = Path(str(request["output_json"])).expanduser().resolve()
                    output_masks = Path(str(request["output_masks_npz"])).expanduser().resolve()
                    _write_outputs(output_json, output_masks, class_names, records, masks)
                    sys.stdout.write(
                        json.dumps(
                            {
                                "status": "ok",
                                "num_results": int(len(records)),
                                "output_json": str(output_json),
                                "output_masks_npz": str(output_masks),
                            }
                        )
                        + "\n"
                    )
                sys.stdout.flush()
            except Exception as exc:
                sys.stdout.write(
                    json.dumps(
                        {
                            "status": "error",
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    + "\n"
                )
                sys.stdout.flush()
        return 0
    except Exception as exc:
        traceback.print_exc()
        sys.stderr.write(f"YOLOE server failed: {exc}\n")
        return 1


def main() -> int:
    args = parse_args()
    if args.server:
        return _run_server(args)

    if not args.image_path or not args.class_names_json or not args.output_json or not args.output_masks_npz:
        raise SystemExit(
            "--image-path, --class-names-json, --output-json, and --output-masks-npz are required unless --server is used."
        )
    image_path = Path(args.image_path).expanduser().resolve()
    class_names_json = Path(args.class_names_json).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_masks = Path(args.output_masks_npz).expanduser().resolve()

    try:
        _log("[yoloe] importing ultralytics")
        YOLOE = _import_yoloe(args.repo_root)

        class_names = _load_classes(class_names_json)
        if not class_names:
            raise RuntimeError("No class names provided to YOLOE helper.")

        _log("[yoloe] building model")
        model = YOLOE(args.model_path)
        _log("[yoloe] setting classes")
        model.set_classes(class_names)

        _log("[yoloe] running prediction")
        results = model.predict(
            source=str(image_path),
            device=args.device,
            conf=float(args.conf_thresh),
            iou=float(args.iou_thresh),
            max_det=int(args.max_det),
            verbose=False,
            retina_masks=True,
        )
        if not results:
            _write_outputs(output_json, output_masks, class_names, [], np.zeros((0, 0, 0), dtype=bool))
            return 0

        _log("[yoloe] serializing outputs")
        records, masks = _serialize_results(results[0])
        _write_outputs(output_json, output_masks, class_names, records, masks)
        _log("[yoloe] done")
        return 0
    except Exception as exc:  # pragma: no cover - helper diagnostics
        traceback.print_exc()
        sys.stderr.write(f"YOLOE helper failed: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
