#!/bin/bash
# 05_compile_bundlesdf.sh
CONTAINER="foundationpose_docker"

echo "⚙️ [3/3] 正在编译 bundlesdf..."

docker exec -it $CONTAINER /bin/bash -c "
    set -e
    # 路径通常是 /workspace/bundlesdf/mycuda
    TARGET_DIR='/workspace/bundlesdf/mycuda'
    
    if [ -d \"\$TARGET_DIR\" ]; then
        cd \"\$TARGET_DIR\"
        echo '>>> Cleaning old build...'
        rm -rf build *.egg-info dist
        
        echo '>>> Installing bundlesdf...'
        pip install -e .
    else
        echo '❌ 错误: 找不到 bundlesdf 目录，请检查挂载路径。'
        exit 1
    fi
"

echo "✅ bundlesdf 编译成功！"