#!/bin/bash

# === 配置部分 ===
TARGET="./install/lib/librknn_cpp.so"                  # 你的可执行文件
SYSROOT="/home/yusio/rksysroot"              # 指向板端rootfs目录
RELEASE_DIR="./install"                      # 输出目录
CROSS_READ_ELF="aarch64-none-linux-gnu-readelf" # readelf路径（根据你的交叉编译工具链设置）

# === 初始化 ===
mkdir -p "${RELEASE_DIR}/lib"


# === 使用Set避免重复 ===
declare -A visited_libs

# === 递归函数：查找并复制so依赖 ===
resolve_deps() {
    local binary="$1"
    local level="${2:-0}"

    # 生成缩进
    local indent=""
    for ((i=0; i<level; i++)); do
        indent+="  "
    done

    # 使用 readelf 提取依赖库
    local needed_libs=$($CROSS_READ_ELF -d "$binary" 2>/dev/null | grep NEEDED | sed -n 's/.*\[\(.*\)\]/\1/p')

    for lib in $needed_libs; do
        # 如果已处理过该lib，则跳过
        if [[ -n "${visited_libs[$lib]}" ]]; then
            continue
        fi
        visited_libs[$lib]=1

        # 先在本地 ./libs 查找
        local lib_path=$(find ./libs -name "$lib" 2>/dev/null | head -n 1)
        # 如果本地没找到，再去 SYSROOT 查找
        if [[ -z "$lib_path" ]]; then
            lib_path=$(find "$SYSROOT" -name "$lib" 2>/dev/null | head -n 1)
        fi
        if [[ -z "$lib_path" ]]; then
            echo "${indent}警告: 在 ./libs 和 sysroot 中都找不到 $lib"
            continue
        fi

        echo "${indent}复制依赖库: $lib_path"
        cp "$lib_path" "$RELEASE_DIR/lib/" &

        # 递归解析它的依赖，层级+1
        resolve_deps "$lib_path" $((level+1))
    done
}

# === 启动递归 ===
resolve_deps "$TARGET" 0

# 等待所有后台 cp 结束
wait

# === 完成提示 ===
echo "所有依赖已复制到 $RELEASE_DIR/lib"
# === 移除所有 so 的 RPATH ===
echo "移除 $RELEASE_DIR/lib 下所有 so 的 RPATH..."
# patchelf --remove-rpath $RELEASE_DIR/lib/lib*