# AI Module — smart_vlm

Zero-training pipeline for the CMU VLN Challenge 2026: open-vocabulary
detection (YOLO-World, CPU-capable) + LiDAR depth fusion into an object-level
semantic map + deterministic spatial-relation toolbox + LLM/VLM API reasoning,
with a 10-minute budgeted state machine and hard-deadline fallback (a legal
answer is always published).

## Build

```bash
cd docker
docker compose -f compose.yml up --build -d      # or compose_gpu.yml
```

The image (`ai_module/docker/Dockerfile`) bakes in all Python dependencies,
CPU PyTorch, and YOLO-World + CLIP weights (no network needed at inference
time except for the LLM API). Both `dummy_vlm` and `smart_vlm` are built.

## Run

```bash
docker exec -it iros2026_ai_module bash
ros2 launch smart_vlm smart_vlm.launch.py
```

The node subscribes to `/challenge_question`, `/state_estimation`,
`/camera/image`, `/registered_scan`, `/terrain_map_ext` and publishes
`/numerical_response` (Int32), `/selected_object_marker` (Marker, CUBE in
map frame), `/way_point_with_heading` (Pose2D) depending on the question type.

API keys and all tunables live in `ai_module/src/smart_vlm/config/params.yaml`.

## Layout

- `src/smart_vlm/` — ROS 2 ament_python package (see its README for the
  module map); `src/smart_vlm/tests/` — 128 pytest cases, runnable without ROS
- `src/dummy_vlm/` — original dummy model, kept for reference/fallback
- `tools/` — development-only calibration and regression-evaluation scripts
