#!/bin/bash
# 04_compile_kaolin.sh
CONTAINER="foundationpose_docker"

echo "⚙️ [2/3] 正在编译 kaolin..."

docker exec -it $CONTAINER /bin/bash -c "
    set -e
    # 注意：镜像里的 kaolin 通常在根目录 /kaolin
    if [ -d '/kaolin' ]; then
        cd /kaolin
        echo '>>> Cleaning old build...'
        rm -rf build *.egg-info dist
        
        echo '>>> Installing kaolin...'
        pip install -e .
    else
        echo '❌ 错误: 找不到 /kaolin 目录，请检查镜像内容。'
        exit 1
    fi
"

echo "✅ kaolin 编译成功！"