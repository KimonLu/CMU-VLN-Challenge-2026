FROM zhangjicmu/ubuntu24_ros:ai_module

# Install the pinned Python runtime on the official challenge image.
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip && \
    rm -rf /var/lib/apt/lists/*
COPY src/smart_vlm/requirements.txt /tmp/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir \
      -i https://pypi.tuna.tsinghua.edu.cn/simple \
      torch==2.12.1 torchvision==0.27.1 && \
    pip3 install --break-system-packages --no-cache-dir \
      -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/requirements.txt && \
    pip3 install --break-system-packages --no-cache-dir \
      git+https://github.com/ultralytics/CLIP.git@f5e9fc6a896e66309f07747bd607704f6c3d282a
# Cache detector and CLIP weights for offline model startup.
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
# Build the reference package and the submitted smart_vlm package.
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
