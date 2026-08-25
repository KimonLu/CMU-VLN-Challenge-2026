# smart_vlm - CMU VLN Challenge 2026 AI Module

`smart_vlm` is a zero-training vision-language navigation pipeline. It combines
open-vocabulary YOLO-World detection, LiDAR depth fusion, an object-level
semantic map, deterministic spatial reasoning, and failover LLM/VLM APIs.
Runtime parameters and API provider settings are defined in
`config/params.yaml`.

## Module map

| Path | Responsibility |
| --- | --- |
| `smart_vlm/main_node.py` | ROS 2 state machine, subscriptions, publications, time budgets, and watchdogs |
| `smart_vlm/pose_buffer.py` | Timestamped pose and scan buffers for delayed panoramic perception |
| `smart_vlm/projection.py` | Equirectangular panorama and 3D geometry conversion |
| `smart_vlm/perception.py` | CPU/GPU-adaptive YOLO-World detection |
| `smart_vlm/semantic_map.py` | Multi-view object fusion, normalization, and attributes |
| `smart_vlm/exploration.py` | Occupancy grid, frontier exploration, A*, and waypoint selection |
| `smart_vlm/spatial_tools.py` | Deterministic spatial-relation filters |
| `smart_vlm/llm_client.py` | Multi-provider API failover, throttling, caching, and call budgets |
| `smart_vlm/answering.py` | Question parsing and the three challenge task pipelines |
| `config/params.yaml` | API providers, model paths, timing, mapping, and navigation thresholds |
| `tests/` | 144 offline pytest cases that do not require a ROS runtime |

## Test

From the repository root:

```bash
python -m pytest ai_module/src/smart_vlm/tests -q
```

The calibration utilities under `ai_module/tools/` can be used to verify the
panorama projection signs and the terrain intensity threshold.
