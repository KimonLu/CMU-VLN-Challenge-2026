FROM zhangjicmu/ubuntu24_ros:ai_module

# 基础镜像默认用户为 docker(uid 1001) 且未预装 pip → 切 root 补装
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip && \
    rm -rf /var/lib/apt/lists/*

# ---- Python 依赖：锁定已完成端到端验证的 CUDA 13 组合 ----
COPY src/smart_vlm/requirements.txt /tmp/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir \
      -i https://pypi.tuna.tsinghua.edu.cn/simple \
      torch==2.12.1 torchvision==0.27.1 && \
    pip3 install --break-system-packages --no-cache-dir \
      -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/requirements.txt && \
    pip3 install --break-system-packages --no-cache-dir \
      git+https://github.com/ultralytics/CLIP.git@f5e9fc6a896e66309f07747bd607704f6c3d282a

# ---- 预下载模型权重进镜像(评测机不现场联网下载) ----
# ultralytics 8.4 把 CLIP ViT-B/32 下到 WEIGHTS_DIR/clip,而 WEIGHTS_DIR 默认是
# 相对路径 "weights"(随进程 CWD 变!)→ 先把 weights_dir 设置成绝对路径写进
# settings.json(位于 $YOLO_CONFIG_DIR,烘焙进镜像),运行时从任意目录启动都能命中
ENV YOLO_CONFIG_DIR=/tmp/ultralytics
RUN mkdir -p /tmp/ultralytics /home/docker/ai_module/models && \
    cd /home/docker/ai_module/models && \
    python3 -c "\
from ultralytics import settings; \
settings.update({'weights_dir': '/home/docker/ai_module/models/weights'})" && \
    python3 -c "\
from ultralytics import YOLOWorld; \
m = YOLOWorld('yolov8s-worldv2.pt'); m.set_classes(['chair']); \
m = YOLOWorld('yolov8m-worldv2.pt'); m.set_classes(['chair'])" && \
    test -f yolov8s-worldv2.pt && test -f yolov8m-worldv2.pt && \
    test -f weights/clip/ViT-B-32.pt && \
    chown -R docker:docker /home/docker/ai_module && \
    chmod -R a+rwX /home/docker/ai_module/models /tmp/ultralytics

# ---- 构建 smart_vlm(保留 dummy_vlm 作对照/回退;以运行时用户 docker 构建) ----
COPY --chown=docker:docker src/dummy_vlm /home/docker/ai_module/src/dummy_vlm
COPY --chown=docker:docker src/smart_vlm /home/docker/ai_module/src/smart_vlm
COPY --chown=docker:docker docker/cyclonedds.xml /home/docker/cyclonedds.xml
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=file:///home/docker/cyclonedds.xml
USER docker
WORKDIR /home/docker/ai_module
RUN /bin/bash -c "\
    source /opt/ros/jazzy/setup.bash && \
    colcon build \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select dummy_vlm smart_vlm"

RUN echo "source /home/docker/ai_module/install/setup.bash" >> /home/docker/.bashrc
