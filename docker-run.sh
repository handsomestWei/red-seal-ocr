#!/bin/bash

# OCR服务Docker运行脚本
# 使用方法: ./docker-run.sh

# 配置变量
IMAGE_NAME="red-seal-ocr"
IMAGE_TAG="latest"
CONTAINER_NAME="red-seal-ocr"
HOST_PORT=5000
CONTAINER_PORT=5000

# 资源限制配置（根据服务器资源调整）
CPU_LIMIT="4"          # CPU核心数
MEMORY_LIMIT="6g"     # 内存限制
MEMORY_SWAP="6g"      # 内存+交换空间（与内存相同，禁用swap）

# 宿主机目录配置（请根据实际情况修改）
HOST_LOGS_DIR="./logs"
HOST_MODEL_DIR="./model"

# 创建宿主机目录（如果不存在）
mkdir -p "${HOST_LOGS_DIR}"
mkdir -p "${HOST_MODEL_DIR}"

# 检查config.yaml是否存在
if [ ! -f "config.yaml" ]; then
    echo "错误: 无法找到config.yaml文件（应在项目根目录）"
    echo "当前目录: $(pwd)"
    exit 1
fi

# 获取绝对路径（确保挂载正确）
CONFIG_FILE="$(pwd)/config.yaml"
LOGS_DIR="$(pwd)/${HOST_LOGS_DIR}"
MODEL_DIR="$(pwd)/${HOST_MODEL_DIR}"

# 验证路径
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件路径无效: $CONFIG_FILE"
    exit 1
fi

echo "配置文件路径: $CONFIG_FILE"
echo "日志目录路径: $LOGS_DIR"
echo "模型目录路径: $MODEL_DIR"

# 停止并删除已存在的容器（如果存在）
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "停止并删除已存在的容器: ${CONTAINER_NAME}"
    docker stop ${CONTAINER_NAME} >/dev/null 2>&1
    docker rm ${CONTAINER_NAME} >/dev/null 2>&1
fi

# 运行容器
echo "启动OCR服务容器..."
echo "容器名称: ${CONTAINER_NAME}"
echo "服务地址: http://localhost:${HOST_PORT}"
echo "配置文件: $(pwd)/config.yaml"
echo "日志目录: ${HOST_LOGS_DIR}"
echo "模型目录: ${HOST_MODEL_DIR}"
echo "资源限制: CPU=${CPU_LIMIT}核, 内存=${MEMORY_LIMIT} "
echo ""

docker run -d \
    --name ${CONTAINER_NAME} \
    -p ${HOST_PORT}:${CONTAINER_PORT} \
    -v "${CONFIG_FILE}:/app/config.yaml:ro" \
    -v "${LOGS_DIR}:/app/logs" \
    -v "${MODEL_DIR}:/app/model" \
    --cpus=${CPU_LIMIT} \
    --memory=${MEMORY_LIMIT} \
    --memory-swap=${MEMORY_SWAP} \
    --restart unless-stopped \
    ${IMAGE_NAME}:${IMAGE_TAG}

# 检查容器是否启动成功
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 容器启动成功！"
    echo ""
    echo "查看日志: docker logs -f ${CONTAINER_NAME}"
    echo "停止容器: docker stop ${CONTAINER_NAME}"
    echo "重启容器: docker restart ${CONTAINER_NAME}"
    echo "进入容器: docker exec -it ${CONTAINER_NAME} /bin/bash"
    echo ""
    echo "服务接口:"
    echo "  健康检查: http://localhost:${HOST_PORT}/health"
    echo "  OCR识别: http://localhost:${HOST_PORT}/ocr"
    echo "  批量OCR: http://localhost:${HOST_PORT}/ocr/batch"
else
    echo ""
    echo "❌ 容器启动失败！"
    echo "请检查错误信息并重试"
    exit 1
fi

