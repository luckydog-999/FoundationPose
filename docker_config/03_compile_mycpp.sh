#!/bin/bash
# 03_compile_mycpp.sh
CONTAINER="foundationpose_docker"

echo "⚙️ [1/3] 正在编译 mycpp..."

docker exec -it $CONTAINER /bin/bash -c "
    set -e
    # 容器内的工作目录
    cd /workspace/mycpp
    
    # 清理旧的编译文件
    rm -rf build
    mkdir -p build && cd build
    
    # 编译命令
    echo '>>> Running CMake...'
    cmake .. -DPYTHON_EXECUTABLE=\$(which python)
    
    echo '>>> Running Make...'
    # 自动使用所有 CPU 核心加速
    make -j\$(nproc)
"

echo "✅ mycpp 编译成功！"