# smart_vlm — CMU VLN Challenge 2026 AI 模块

零训练方案:开放词汇检测(YOLO-World)+ 激光深度融合建物体库 + 空间关系工具箱 +
LLM/VLM API 推理。运行参数和 API 配置位于 `config/params.yaml`。

文件对应关系:

| 文件 | 职责 |
|---|---|
| `smart_vlm/main_node.py` | 状态机、订阅/发布、时间预算、看门狗 |
| `smart_vlm/pose_buffer.py` | 位姿时间缓冲(图像↔位姿同步)、关键帧判据 |
| `smart_vlm/projection.py` | 全景图⇄3D 几何、视图切片(wrap-aware) |
| `smart_vlm/perception.py` | YOLO-World 检测(CPU/GPU 自适应) |
| `smart_vlm/semantic_map.py` | 检测+点云→物体库(合并/同义词/颜色) |
| `smart_vlm/exploration.py` | 栅格图 + frontier 探索 + A* + 航点抽稀 |
| `smart_vlm/spatial_tools.py` | 空间关系工具箱(确定性几何) |
| `smart_vlm/llm_client.py` | 多供应商 API、failover、缓存、预算 |
| `smart_vlm/answering.py` | 问题解析 + 三题型 pipeline |
| `config/params.yaml` | API keys、模型路径、全部阈值 |
| `tests/` | 121 个单测,无 ROS 即可跑(pytest) |

两处 `TODO(calibrate)` 参数(投影符号、terrain 阈值)可使用
`ai_module/tools/` 中的标定工具确定。
