# FAST-LIVO2 × OVO 语义观察器改造方案

## 1. 文档目的

本文档用于指导将当前：

`FAST-LIVO2 关键帧 + pose + RGBD -> OVO 自己累计几何/语义地图 -> 回投影检查`

改造为：

`FAST-LIVO2 作为唯一几何/位姿主权威 -> 导出当前关键帧可见点集 -> OVO 只做 2D 语义分割与点投票 -> 结果写入观测库与对象记忆 -> Final Consolidation 输出最终语义地图`

目标是解决当前系统中：

- OVO working map 回投影偏差明显，尤其在机器人旋转时更明显
- 几何误差、语义误差、实例漂移混在一起，难以定位
- OVO 既负责语义又负责几何累计，职责不清

---

## 2. 最终结论

### 2.1 保留什么
- **FAST-LIVO2**
  - 继续作为唯一几何主权威
  - 继续作为唯一位姿主权威
  - 继续负责图像与 LiDAR 点之间的高质量对应关系
- **OVO**
  - 保留 2D segmentation / 3D 点投票 / segment 更新 / 语义描述子这部分能力
  - 不再承担最终几何地图生成职责

### 2.2 改掉什么
- 不再使用 OVO 自己累计出的全局 `working semantic map` 作为最终真相
- 不再让 OVO 用 RGBD 自己重建一张长期几何图
- 不再通过“OVO working map 再投回原图”的方式判断 pose 或几何是否可靠

### 2.3 新原则
**图像-点云对应关系由 FAST-LIVO2 提供，语义解释由 OVO 提供，长期地图由对象记忆和最终整合模块输出。**

---

## 3. 为什么要这么改

### 3.1 当前问题的本质
当前链路里，OVO 同时承担了两件事：

1. 从关键帧/深度累计全局几何图
2. 在这张工作几何图上做语义实例更新

这会导致两个问题：

- 即使 FAST-LIVO2 的几何和 pose 很好，OVO 仍可能因为自己的全局几何累计方式造成回投影偏差
- 语义问题和几何问题耦合，排查困难

### 3.2 FAST-LIVO2 的优势
FAST-LIVO2 本身已经做了：
- 当前帧可见 LiDAR 点选择
- 图像 patch 与 LiDAR 点的高质量对应
- 视觉更新时的可见性、深度不连续、局部平面等处理

所以，最合理的做法是：

**直接复用 FAST-LIVO2 已经处理好的“关键帧图片 ↔ 当前帧可见 3D 点”的关系。**

### 3.3 OVO 的真正强项
OVO 的强项是：
- 2D mask 分割
- 3D point / segment 投票
- 3D instance / segment 更新
- 多视角语义描述子
- 最终实例融合

它的强项不是“长期几何真相维护”。

---

## 4. 新系统架构

```text
FAST-LIVO2
  ├─ pose authority
  ├─ map / local cloud authority
  └─ visible local cloud / visible map point selection
           ↓
KeyframePacket
  ├─ keyframe image
  ├─ keyframe pose
  ├─ local cloud reference
  └─ sensor config reference
           ↓
Semantic Adapter (OVO-style)
  ├─ 2D masks
  ├─ project visible local cloud to image
  ├─ mask-hit point indices
  └─ semantic observation generation
           ↓
ObservationStore
           ↓
ObjectMemory
           ↓
FinalConsolidation
           ↓
Final semantic map / Hydra input / retrieval index
```

---

## 5. 关键设计决策

### 5.1 不在关键帧里重复存内外参
关键帧不存相机内参、外参的完整副本。  
因为它们是固定不变的，应统一存放在固定配置文件中。

关键帧只记录：
- `sensor_config_id`
- `calib_version`

程序运行时按 ID 读取配置。

### 5.2 不把整张全局点云喂给 OVO
**只能给 OVO 当前关键帧真正可见的 local cloud / visible map points。**

不能直接把整张全局彩色点云拿去投到当前图像里做 mask 投票，因为这样会引入：
- 被遮挡点
- 背后点
- 远处不相关点
- 历史帧遗留的无关点

### 5.3 不直接把单帧语义永久写死到全局点云
必须分两层写入：

#### 观测层
记录：
- 当前帧看到了哪些点
- 哪些点命中了哪个 mask
- 当前帧给出的语义票数是什么

#### 全局层
综合多帧观测后，再更新：
- 全局点语义
- 3D object 语义
- 最终实例 ID

---

## 6. 数据结构设计

### 6.1 SensorConfig

```python
class SensorConfig:
    sensor_config_id: str
    calib_version: str
    camera_model: str
    intrinsics: dict
    t_lidar_cam: list[float]
    image_size: tuple[int, int]
    distortion_model: str
```

### 6.2 KeyframePacket

```python
class KeyframePacket:
    keyframe_id: int
    stamp_sec: float
    t_world_body: list[float]
    t_world_lidar: list[float]
    t_world_cam: list[float]
    rgb_path: str
    depth_path: str | None
    local_cloud_ref: str
    local_cloud_frame: str   # lidar/world
    sensor_config_id: str
    calib_version: str
    pose_source: str         # fast_livo2
    status: str              # raw / observed / fused / final
```

### 6.3 LocalCloudPacket

```python
class LocalCloudPacket:
    local_cloud_id: str
    source_scan_ids: list[str]
    stamp_start: float
    stamp_end: float
    frame: str
    cloud_path: str
    point_count: int
    parent_submap_id: str | None
```

### 6.4 ObservationLink

```python
class ObservationLink:
    keyframe_id: int
    mask_id: int
    local_cloud_id: str
    point_indices: list[int]
    semantic_label_candidates: list[str]
    semantic_scores: list[float]
    candidate_object_id: str | None
    vote_count: int
    visibility_score: float
    bbox_xyxy: list[int]
    mask_area: int
```

### 6.5 ObjectMemory

```python
class ObjectMemory:
    object_id: str
    point_support_refs: list[tuple[str, list[int]]]
    centroid_world: list[float]
    bbox_world: list[float]
    label_votes: dict[str, float]
    embedding_path: str | None
    best_view_keyframes: list[int]
    observation_count: int
    stability_score: float
    completeness_score: float
    dirty_flag: bool
    last_seen_stamp: float
```

---

## 7. 新流程（在线阶段）

### 7.1 Phase A：关键帧导出层

输入：
- FAST-LIVO2 当前关键帧
- FAST-LIVO2 当前时刻 pose
- FAST-LIVO2 对应 visible local cloud / visible map points

输出：
- `KeyframePacket`
- `LocalCloudPacket`

任务：
1. 扩展当前 exporter
2. 每个关键帧绑定一个 `local_cloud_ref`
3. 可选记录 `source_scan_ids`
4. 使用 `sensor_config_id` 引用固定配置文件

### 7.2 Phase B：单帧可见点投影层

新模块：
`visibility_projector.py`

输入：
- `KeyframePacket`
- `LocalCloudPacket`
- `SensorConfig`

输出：
- `visible_point_indices`
- `projected_uv`
- `projected_depth`
- `visibility_score`

核心要求：
1. 只投影当前关键帧真正可见的点
2. 优先复用 FAST-LIVO2 已有可见点筛选逻辑
3. 如暂时难复用，先做简化 z-buffer + FoV 剪裁

### 7.3 Phase C：语义观察层

新模块：
`semantic_observer.py`

输入：
- 当前关键帧 RGB
- 当前关键帧对应可见 local cloud 投影结果

处理：
1. 跑 2D segmentation
2. 对每个 mask：
   - 收集命中的 3D 点索引
   - 统计已有 object / segment 众数
   - 生成语义候选和分数
3. 产出 `ObservationLink`

重要原则：
- 先只生成 observation
- 暂时不要直接改全局 semantic cloud

### 7.4 Phase D：对象记忆更新层

新模块：
`object_memory.py`

输入：
- 一批 `ObservationLink`

处理：
1. 按几何关联找已有 object
2. 按语义关联筛选
3. 满足阈值则并入旧 object
4. 否则创建新 object

关键规则：
- 点太少不更新 object
- 小 mask 不更新 object
- 票数不稳定只记 observation，不改主标签
- 每个 object 最多保留 3–5 个最佳视角

---

## 8. Final Consolidation（最终地图输出）

触发时机：
- 房间探索完成
- 跑完一圈
- 或手动触发

新模块：
`final_consolidation.py`

动作：
1. 冻结关键帧集
2. 读取所有 ObservationLinks
3. 更新和清洗 ObjectMemory
4. 对每个 object 执行：
   - 主连通块保留
   - 小碎片剔除
   - 同类近邻实例融合
   - 标签投票稳定
5. 导出：
   - `final_semantic_cloud.ply`
   - `final_objects.json`
   - `hydra_nodes_edges.json`

---

## 9. 排查顺序（必须按这个顺序）

### Step 1：单帧闭环
只测试：

`当前关键帧 local cloud -> 当前图像`

如果这一步都不准，问题在：
- 时间对齐
- 坐标链
- 投影实现
- 可见性筛选

### Step 2：单帧语义附着
只测试：

`当前关键帧 RGB -> mask -> 当前关键帧 visible local cloud`

如果这一步偏，问题在：
- mask 质量
- 命中点选择
- 可见性筛选

### Step 3：多帧对象记忆
先不关心最终全局图，只看 object ID 稳不稳定。

如果这一步开始飘，问题在：
- observation 到 object 的关联
- vote update
- best-view 选择
- instance fusion

### Step 4：Final Consolidation
如果 online 过程中允许有些抖动，但 final map 稳定，说明系统路线正确。

---

## 10. Codex 实施顺序

### Task 1：新增数据结构与 manifest
新增文件：
- `onemap_semantic_mapper/data_types.py`
- `onemap_semantic_mapper/io/keyframe_manifest.py`
- `onemap_semantic_mapper/io/local_cloud_manifest.py`
- `onemap_semantic_mapper/io/sensor_config.py`

内容：
- `KeyframePacket`
- `LocalCloudPacket`
- `SensorConfig`
- `ObservationLink`
- `ObjectMemory`

### Task 2：扩展 exporter
优先基于现有 exporter 修改。

要求：
- 每个关键帧必须绑定一个 `local_cloud_ref`
- 必须能回溯来源 scan ids
- 不再在关键帧里重复写 intrinsics/extrinsics，只写配置引用

### Task 3：实现 `visibility_projector.py`
要求：
- 输入关键帧与 local cloud
- 输出当前帧可见点索引与 uv
- 先实现可用版本，再逐步复用 FAST-LIVO2 内部逻辑

### Task 4：实现 `semantic_observer.py`
要求：
- 输入 RGB + visible local cloud
- 复用现有 2D segmentation
- 输出 ObservationLinks
- 暂时不要写全局 semantic cloud

### Task 5：实现 `object_memory.py`
要求：
- 增量更新 object
- best-view 管理
- 低质量 observation 不更新主 object

### Task 6：实现 `final_consolidation.py`
要求：
- 从 ObservationLinks 与 ObjectMemory 输出 final map
- 输出 final objects 和 final semantic cloud
- 不依赖 OVO working map

---

## 11. 给 Codex 的执行提示词

```markdown
Read these first:
- .ai/PROJECT_FACTS.md
- .ai/INVARIANTS.md
- .ai/CURRENT_PLAN.md
- .ai/ATTEMPT_LOG.md

Goal:
Refactor the current FAST-LIVO2 -> OVO pipeline so that FAST-LIVO2 remains the only geometry/pose authority, while OVO is reduced to a semantic observation and association layer.

Core design:
1. Do NOT use OVO's accumulated global working map as final geometry authority.
2. Export per-keyframe visible local cloud / local submap references from FAST-LIVO2.
3. Camera intrinsics/extrinsics are constant; load them from shared config files, do not duplicate them in every keyframe record.
4. Run 2D segmentation on the RGB image, then project the FAST-LIVO2-visible local cloud into the image.
5. Persist semantic results as:
   - KeyframePacket
   - LocalCloudPacket
   - ObservationLink
   - ObjectMemory
6. Produce final maps only through a final consolidation step.

Implementation order:
1. data structures and manifests
2. exporter binding keyframes to local clouds
3. visibility_projector.py
4. semantic_observer.py
5. object_memory.py
6. final_consolidation.py

Acceptance criteria:
- Single-frame projection test passes on 20 sampled keyframes
- Rotation frames do not show large global semantic offset
- Semantic votes are attached to original FAST-LIVO2 local cloud points
- Final semantic map no longer depends on OVO global working cloud

After each task:
- update .ai/ATTEMPT_LOG.md
- update .ai/CURRENT_PLAN.md
- produce a small evidence bundle
```

---

## 12. 一句话总结

**FAST-LIVO2 负责“哪些点在那、这帧看到了谁”；OVO 负责“这些点属于什么语义/实例”；最终地图由对象记忆与最终整合模块输出。**

这才是当前系统最稳、最清晰、最适合继续往 Hydra / 检索 / 导航扩展的路线。
