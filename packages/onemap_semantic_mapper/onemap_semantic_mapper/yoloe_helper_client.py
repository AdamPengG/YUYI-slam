from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class YOLOEHelperClient:
    def __init__(
        self,
        *,
        helper_script: str | Path,
        repo_root: str | Path,
        conda_env: str = "yoloe_env",
        python_bin: str = "",
        cuda_visible_devices: str = "1",
        model_path: str = "yoloe-26x-seg.pt",
        device: str = "cuda",
        conf_thresh: float = 0.10,
        iou_thresh: float = 0.50,
        max_det: int = 100,
        topk_labels: int = 1,
    ) -> None:
        self.helper_script = Path(helper_script).expanduser().resolve()
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.conda_env = str(conda_env).strip()
        self.python_bin = str(python_bin).strip()
        self.cuda_visible_devices = str(cuda_visible_devices).strip()
        self.model_path = str(model_path).strip()
        self.device = str(device).strip()
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.max_det = int(max_det)
        self.topk_labels = max(int(topk_labels), 1)
        self._server_proc: subprocess.Popen[str] | None = None

    def _build_python_command(self) -> list[str]:
        if self.python_bin:
            return [self.python_bin, str(self.helper_script)]
        env_python = Path(f"/home/peng/miniconda3/envs/{self.conda_env}/bin/python")
        if env_python.exists():
            return [str(env_python), str(self.helper_script)]
        conda_exe = shutil.which("conda") or "/home/peng/miniconda3/bin/conda"
        return [conda_exe, "run", "-n", self.conda_env, "python", str(self.helper_script)]

    def _server_command(self) -> list[str]:
        return self._build_python_command() + [
            "--server",
            "--model-path",
            self.model_path,
            "--device",
            self.device,
            "--conf-thresh",
            str(self.conf_thresh),
            "--iou-thresh",
            str(self.iou_thresh),
            "--max-det",
            str(self.max_det),
            "--topk",
            str(self.topk_labels),
            "--repo-root",
            str(self.repo_root),
        ]

    def _server_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        return env

    def _ensure_server(self) -> subprocess.Popen[str]:
        proc = self._server_proc
        if proc is not None and proc.poll() is None and proc.stdin is not None and proc.stdout is not None:
            return proc

        proc = subprocess.Popen(
            self._server_command(),
            cwd=str(self.repo_root if self.repo_root.exists() else Path("/home/peng/isacc_slam")),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=self._server_env(),
        )
        if proc.stdout is None or proc.stdin is None:
            raise RuntimeError("Failed to start persistent YOLOE helper.")
        ready_line = proc.stdout.readline().strip()
        if not ready_line:
            raise RuntimeError("Persistent YOLOE helper exited before signaling readiness.")
        payload = json.loads(ready_line)
        if payload.get("status") != "ready":
            raise RuntimeError(f"Persistent YOLOE helper failed to start: {payload}")
        self._server_proc = proc
        return proc

    def close(self) -> None:
        proc = self._server_proc
        self._server_proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=5.0)
        except Exception:
            proc.kill()

    @staticmethod
    def _encode_image_rgb(image_rgb: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Failed to encode RGB image for YOLOE helper request.")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    @staticmethod
    def _decode_masks(response: dict[str, Any]) -> np.ndarray | None:
        packed_b64 = response.get("masks_packbits_base64")
        shape = response.get("mask_shape")
        if not packed_b64 or not isinstance(shape, list) or len(shape) != 3:
            return None
        raw = base64.b64decode(str(packed_b64))
        packed = np.frombuffer(raw, dtype=np.uint8)
        total = int(shape[0]) * int(shape[1]) * int(shape[2])
        bits = np.unpackbits(packed, bitorder="little")
        if bits.size < total:
            raise RuntimeError("YOLOE helper returned truncated packed masks.")
        masks = bits[:total].reshape((int(shape[0]), int(shape[1]), int(shape[2]))).astype(bool, copy=False)
        return masks

    def infer(
        self,
        image_rgb: np.ndarray,
        class_names: list[str],
        *,
        include_masks: bool,
    ) -> tuple[list[dict[str, Any]], np.ndarray | None]:
        if not class_names:
            return [], None
        proc = self._ensure_server()
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("Persistent YOLOE helper pipes are unavailable.")
        proc.stdin.write(
            json.dumps(
                {
                    "class_names": [str(name) for name in class_names if str(name).strip()],
                    "image_png_base64": self._encode_image_rgb(image_rgb),
                    "return_inline": True,
                    "include_masks": bool(include_masks),
                }
            )
            + "\n"
        )
        proc.stdin.flush()
        response_line = proc.stdout.readline().strip()
        if not response_line:
            raise RuntimeError("Persistent YOLOE helper exited while processing a frame.")
        response = json.loads(response_line)
        if response.get("status") != "ok":
            raise RuntimeError(f"YOLOE helper failed: {response.get('message', response)}")
        results = response.get("results", [])
        if not isinstance(results, list):
            results = []
        masks = self._decode_masks(response) if include_masks else None
        return [dict(item) for item in results], masks
