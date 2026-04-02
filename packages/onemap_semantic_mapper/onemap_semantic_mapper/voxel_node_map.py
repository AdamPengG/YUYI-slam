from __future__ import annotations

from pathlib import Path

import numpy as np


class VoxelNodeMap:
    """TSDF-style hash-grid geometry carrier with node ownership votes.

    This keeps geometry dense enough for export and online visualization while
    storing semantic ownership at the node level rather than per-point labels.
    """

    def __init__(
        self,
        voxel_size_m: float = 0.02,
        truncation_m: float | None = None,
        surface_band_m: float | None = None,
    ) -> None:
        self.voxel_size_m = max(float(voxel_size_m), 1e-3)
        self.truncation_m = max(float(truncation_m if truncation_m is not None else (self.voxel_size_m * 4.0)), self.voxel_size_m)
        self.surface_band_m = max(float(surface_band_m if surface_band_m is not None else (self.voxel_size_m * 1.5)), self.voxel_size_m * 0.5)
        self._voxels: dict[tuple[int, int, int], dict[str, object]] = {}

    def _voxel_key(self, xyz: np.ndarray) -> tuple[int, int, int]:
        coord = np.floor(np.asarray(xyz, dtype=np.float32) / self.voxel_size_m).astype(np.int32, copy=False)
        return int(coord[0]), int(coord[1]), int(coord[2])

    def _voxel_center(self, key: tuple[int, int, int]) -> np.ndarray:
        return (np.asarray(key, dtype=np.float32) + 0.5) * self.voxel_size_m

    def _empty_record(self, stamp_sec: float) -> dict[str, object]:
        return {
            "tsdf": 1.0,
            "tsdf_weight": 0.0,
            "surface_sum_xyz": np.zeros((3,), dtype=np.float64),
            "surface_weight": 0.0,
            "normal_sum": np.zeros((3,), dtype=np.float64),
            "node_votes": {},
            "last_seen_stamp": float(stamp_sec),
        }

    def _get_record(self, key: tuple[int, int, int], stamp_sec: float) -> dict[str, object]:
        record = self._voxels.get(key)
        if record is None:
            record = self._empty_record(stamp_sec)
            self._voxels[key] = record
        return record

    def _integrate_surface_only(
        self,
        xyz_world: np.ndarray,
        node_ids_arr: np.ndarray | None,
        stamp_sec: float,
    ) -> None:
        for idx, point in enumerate(np.asarray(xyz_world, dtype=np.float32)):
            key = self._voxel_key(point)
            record = self._get_record(key, stamp_sec)
            old_weight = float(record["tsdf_weight"])
            new_weight = old_weight + 1.0
            record["tsdf"] = float((float(record["tsdf"]) * old_weight) / max(new_weight, 1e-6))
            record["tsdf_weight"] = new_weight
            record["surface_sum_xyz"] = np.asarray(record["surface_sum_xyz"], dtype=np.float64) + point.astype(np.float64)
            record["surface_weight"] = float(record["surface_weight"]) + 1.0
            record["last_seen_stamp"] = float(stamp_sec)
            if node_ids_arr is not None and idx < node_ids_arr.shape[0]:
                node_id = str(node_ids_arr[idx])
                if node_id:
                    votes = dict(record.get("node_votes", {}))
                    votes[node_id] = float(votes.get(node_id, 0.0)) + 1.0
                    record["node_votes"] = votes

    def integrate(
        self,
        xyz_world: np.ndarray,
        node_ids: np.ndarray | None,
        stamp_sec: float,
        sensor_origin_world: np.ndarray | None = None,
    ) -> None:
        xyz_world = np.asarray(xyz_world, dtype=np.float32)
        if xyz_world.size == 0:
            return
        node_ids_arr = None
        if node_ids is not None:
            node_ids_arr = np.asarray(node_ids, dtype=object).reshape(-1)

        if sensor_origin_world is None:
            self._integrate_surface_only(xyz_world, node_ids_arr, stamp_sec)
            return

        sensor_origin = np.asarray(sensor_origin_world, dtype=np.float32).reshape(3)
        ray_step = self.voxel_size_m
        max_ray_steps = max(int(np.ceil((2.0 * self.truncation_m) / max(ray_step, 1e-6))) + 1, 5)

        for idx, point in enumerate(xyz_world):
            ray_vec = point - sensor_origin
            ray_dist = float(np.linalg.norm(ray_vec))
            if not np.isfinite(ray_dist) or ray_dist <= 1e-6:
                continue
            ray_dir = ray_vec / ray_dist
            sample_start = max(ray_dist - self.truncation_m, ray_step * 0.5)
            sample_end = ray_dist + self.truncation_m
            sample_count = max(int(np.ceil((sample_end - sample_start) / max(ray_step, 1e-6))) + 1, 3)
            sample_count = min(sample_count, max_ray_steps)
            sample_distances = np.linspace(sample_start, sample_end, num=sample_count, dtype=np.float32)

            node_id = ""
            if node_ids_arr is not None and idx < node_ids_arr.shape[0]:
                node_id = str(node_ids_arr[idx])

            for sample_distance in sample_distances.tolist():
                sample_xyz = sensor_origin + (ray_dir * float(sample_distance))
                sdf_m = float(ray_dist - float(sample_distance))
                tsdf = float(np.clip(sdf_m / max(self.truncation_m, 1e-6), -1.0, 1.0))
                key = self._voxel_key(sample_xyz)
                record = self._get_record(key, stamp_sec)
                old_weight = float(record["tsdf_weight"])
                sample_weight = 1.0 - min(abs(sdf_m) / max(self.truncation_m, 1e-6), 1.0) * 0.5
                sample_weight = max(sample_weight, 0.15)
                new_weight = old_weight + sample_weight
                record["tsdf"] = float((float(record["tsdf"]) * old_weight + (tsdf * sample_weight)) / max(new_weight, 1e-6))
                record["tsdf_weight"] = new_weight
                record["last_seen_stamp"] = float(stamp_sec)
                if abs(sdf_m) <= self.surface_band_m:
                    record["surface_sum_xyz"] = np.asarray(record["surface_sum_xyz"], dtype=np.float64) + point.astype(np.float64)
                    record["surface_weight"] = float(record["surface_weight"]) + 1.0
                    record["normal_sum"] = np.asarray(record["normal_sum"], dtype=np.float64) + ray_dir.astype(np.float64)
                    if node_id:
                        votes = dict(record.get("node_votes", {}))
                        votes[node_id] = float(votes.get(node_id, 0.0)) + 1.0
                        record["node_votes"] = votes

    def num_voxels(self) -> int:
        return len(self._voxels)

    def top_assignment(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self._voxels:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.str_),
                np.zeros((0,), dtype=np.float32),
            )
        xyz: list[np.ndarray] = []
        weight: list[float] = []
        node_id: list[str] = []
        node_conf: list[float] = []
        for key, record in self._voxels.items():
            tsdf_weight = float(record.get("tsdf_weight", 0.0))
            tsdf = float(record.get("tsdf", 1.0))
            surface_weight = float(record.get("surface_weight", 0.0))
            if tsdf_weight <= 0.0:
                continue
            if surface_weight <= 0.0 and abs(tsdf) > 0.20:
                continue
            if surface_weight > 0.0:
                surface_xyz = np.asarray(record["surface_sum_xyz"], dtype=np.float64) / max(surface_weight, 1e-6)
                xyz.append(surface_xyz.astype(np.float32))
            else:
                xyz.append(self._voxel_center(key).astype(np.float32))
            weight.append(tsdf_weight)
            votes = dict(record.get("node_votes", {}))
            if votes:
                best_node, best_score = max(votes.items(), key=lambda item: item[1])
                node_id.append(str(best_node))
                node_conf.append(float(best_score) / max(float(sum(votes.values())), 1e-6))
            else:
                node_id.append("")
                node_conf.append(0.0)
        if not xyz:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.str_),
                np.zeros((0,), dtype=np.float32),
            )
        return (
            np.vstack(xyz).astype(np.float32, copy=False),
            np.asarray(weight, dtype=np.float32),
            np.asarray(node_id, dtype=np.str_),
            np.asarray(node_conf, dtype=np.float32),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._voxels.keys())
        coords = np.asarray(keys, dtype=np.int32) if keys else np.zeros((0, 3), dtype=np.int32)
        tsdf = np.asarray([float(self._voxels[key]["tsdf"]) for key in keys], dtype=np.float32)
        tsdf_weight = np.asarray([float(self._voxels[key]["tsdf_weight"]) for key in keys], dtype=np.float32)
        surface_sum_xyz = np.asarray([np.asarray(self._voxels[key]["surface_sum_xyz"], dtype=np.float64) for key in keys], dtype=np.float32)
        surface_weight = np.asarray([float(self._voxels[key]["surface_weight"]) for key in keys], dtype=np.float32)
        normal_sum = np.asarray([np.asarray(self._voxels[key]["normal_sum"], dtype=np.float64) for key in keys], dtype=np.float32)
        voxel_parent_node_id: list[str] = []
        voxel_parent_node_conf: list[float] = []
        for key in keys:
            votes = dict(self._voxels[key].get("node_votes", {}))
            if votes:
                best_node, best_score = max(votes.items(), key=lambda item: item[1])
                voxel_parent_node_id.append(str(best_node))
                voxel_parent_node_conf.append(float(best_score) / max(float(sum(votes.values())), 1e-6))
            else:
                voxel_parent_node_id.append("")
                voxel_parent_node_conf.append(0.0)
        xyz, _weight, node_ids, node_conf = self.top_assignment()
        tmp_path = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            tmp_path,
            coords=coords,
            tsdf=tsdf,
            tsdf_weight=tsdf_weight,
            surface_sum_xyz=surface_sum_xyz,
            surface_weight=surface_weight,
            normal_sum=normal_sum,
            voxel_parent_node_id=np.asarray(voxel_parent_node_id, dtype=np.str_),
            voxel_parent_node_conf=np.asarray(voxel_parent_node_conf, dtype=np.float32),
            xyz=xyz,
            weight=_weight,
            parent_node_id=node_ids,
            parent_node_conf=node_conf,
            voxel_size_m=np.asarray(self.voxel_size_m, dtype=np.float32),
            truncation_m=np.asarray(self.truncation_m, dtype=np.float32),
            surface_band_m=np.asarray(self.surface_band_m, dtype=np.float32),
        )
        tmp_path.replace(path)

    @classmethod
    def load(cls, path: Path) -> "VoxelNodeMap":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            voxel_size_m = float(np.asarray(data["voxel_size_m"]).reshape(()))
            truncation_m = float(np.asarray(data["truncation_m"]).reshape(())) if "truncation_m" in data else None
            surface_band_m = float(np.asarray(data["surface_band_m"]).reshape(())) if "surface_band_m" in data else None
            store = cls(voxel_size_m=voxel_size_m, truncation_m=truncation_m, surface_band_m=surface_band_m)
            if "coords" in data:
                coords = np.asarray(data["coords"], dtype=np.int32)
                tsdf = np.asarray(data["tsdf"], dtype=np.float32).reshape(-1)
                tsdf_weight = np.asarray(data["tsdf_weight"], dtype=np.float32).reshape(-1)
                surface_sum_xyz = np.asarray(data["surface_sum_xyz"], dtype=np.float32)
                surface_weight = np.asarray(data["surface_weight"], dtype=np.float32).reshape(-1)
                normal_sum = np.asarray(data["normal_sum"], dtype=np.float32)
                voxel_parent_node_id = np.asarray(
                    data["voxel_parent_node_id"] if "voxel_parent_node_id" in data else data.get("parent_node_id", np.asarray([], dtype=np.str_)),
                    dtype=np.str_,
                ).reshape(-1)
                voxel_parent_node_conf = np.asarray(
                    data["voxel_parent_node_conf"] if "voxel_parent_node_conf" in data else data.get("parent_node_conf", np.asarray([], dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1)
                for idx, coord in enumerate(coords):
                    node_votes: dict[str, float] = {}
                    node_id = str(voxel_parent_node_id[idx]) if idx < voxel_parent_node_id.shape[0] else ""
                    if node_id:
                        conf = float(voxel_parent_node_conf[idx]) if idx < voxel_parent_node_conf.shape[0] else 1.0
                        node_votes[node_id] = max(conf, 1e-3)
                    store._voxels[(int(coord[0]), int(coord[1]), int(coord[2]))] = {
                        "tsdf": float(tsdf[idx]) if idx < tsdf.shape[0] else 1.0,
                        "tsdf_weight": float(tsdf_weight[idx]) if idx < tsdf_weight.shape[0] else 0.0,
                        "surface_sum_xyz": np.asarray(surface_sum_xyz[idx], dtype=np.float64) if idx < surface_sum_xyz.shape[0] else np.zeros((3,), dtype=np.float64),
                        "surface_weight": float(surface_weight[idx]) if idx < surface_weight.shape[0] else 0.0,
                        "normal_sum": np.asarray(normal_sum[idx], dtype=np.float64) if idx < normal_sum.shape[0] else np.zeros((3,), dtype=np.float64),
                        "node_votes": node_votes,
                        "last_seen_stamp": 0.0,
                    }
                return store

            # Backward compatibility for older occupancy-style exports.
            xyz = np.asarray(data["xyz"], dtype=np.float32)
            weight = np.asarray(data["weight"], dtype=np.float32).reshape(-1)
            parent_node_id = np.asarray(data["parent_node_id"], dtype=np.str_)
            coords = np.floor(xyz / voxel_size_m).astype(np.int32, copy=False)
            for idx, coord in enumerate(coords):
                node_votes: dict[str, float] = {}
                node_id = str(parent_node_id[idx])
                if node_id:
                    node_votes[node_id] = float(weight[idx])
                store._voxels[(int(coord[0]), int(coord[1]), int(coord[2]))] = {
                    "tsdf": 0.0,
                    "tsdf_weight": float(weight[idx]),
                    "surface_sum_xyz": xyz[idx].astype(np.float64) * float(weight[idx]),
                    "surface_weight": float(weight[idx]),
                    "normal_sum": np.zeros((3,), dtype=np.float64),
                    "node_votes": node_votes,
                    "last_seen_stamp": 0.0,
                }
            return store

    def summary(self) -> dict[str, object]:
        xyz, weight, node_id, node_conf = self.top_assignment()
        tsdf_weight = np.asarray([float(record.get("tsdf_weight", 0.0)) for record in self._voxels.values()], dtype=np.float32)
        surface_weight = np.asarray([float(record.get("surface_weight", 0.0)) for record in self._voxels.values()], dtype=np.float32)
        assigned = int(np.count_nonzero(node_id != ""))
        return {
            "geometry_type": "voxel_hash_tsdf",
            "voxel_size_m": float(self.voxel_size_m),
            "truncation_m": float(self.truncation_m),
            "surface_band_m": float(self.surface_band_m),
            "num_voxels": int(len(self._voxels)),
            "num_surface_voxels": int(xyz.shape[0]),
            "num_assigned_voxels": int(assigned),
            "mean_tsdf_weight": float(np.mean(tsdf_weight)) if tsdf_weight.size > 0 else 0.0,
            "mean_surface_weight": float(np.mean(surface_weight)) if surface_weight.size > 0 else 0.0,
            "mean_parent_conf": float(np.mean(node_conf)) if node_conf.size > 0 else 0.0,
        }
