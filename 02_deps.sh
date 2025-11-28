#!/bin/bash
# 02_deps.sh
# 容器名称 (需与 docker-compose.yml 一致)
CONTAINER="foundationpose_docker"

echo "🔧 正在安装系统库和 Python 依赖..."

docker exec -it $CONTAINER /bin/bash -c "
    set -e
    apt-get update
    # 安装 OpenCV 和编译必须的库
    apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev build-essential git cmake
    
    # 安装 Python 库
    pip install --no-cache-dir --upgrade pip
    pip install --no-cache-dir ultralytics opencv-python-headless lz4 scipy trimesh pyzed
"

echo "✅ 依赖安装完成。"