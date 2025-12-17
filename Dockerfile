# 使用华为云镜像仓库的Python 3.13-slim镜像
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.13-slim

# 设置工作目录
WORKDIR /app

# 配置apt国内源（使用阿里云镜像）
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || \
    echo "deb https://mirrors.aliyun.com/debian/ bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian/ bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目文件
COPY . .

# 预下载OCR模型（在构建时执行）
# 使用check_paddleocr.py中的预下载函数
RUN python util/check_paddleocr.py --docker

# 创建logs和model目录（如果不存在）
RUN mkdir -p /app/logs && \
    mkdir -p /app/model

# 暴露端口
EXPOSE 5000

# 设置环境变量
ENV PYTHONUNBUFFERED=1
# 设置OpenCV相关环境变量（避免某些Docker环境下的问题）
ENV OPENCV_IO_ENABLE_OPENEXR=0
ENV QT_QPA_PLATFORM=offscreen
# 设置PaddlePaddle相关环境变量
ENV FLAGS_allocator_strategy=auto_growth
ENV FLAGS_fraction_of_gpu_memory_to_use=0.1

# 启动命令
CMD ["python", "api_server.py"]