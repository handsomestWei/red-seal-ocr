# red-seal-ocr

使用PaddleOCR识别图片中的红色印章文字。

## 功能特点

- 自动识别印章中的文字（如户口本等）
- 支持批量处理多张图片
- **可选图像预处理**：提取红色区域，提高印章识别准确率

## 环境要求

- Python 3.7+
- Windows/Linux/MacOS

## 安装步骤

1. 安装依赖包：
```bash
pip install -r requirements.txt
```

2. 首次运行时会自动下载PaddleOCR模型文件（约几百MB），请确保网络连接正常。

## 使用方法

1. 将需要识别的图片放入 `img/` 目录

2. 运行主程序：
```bash
python main.py
```

3. 程序会自动处理 `img/` 目录下的所有图片，并显示识别结果和有效性判断。

## 项目结构

```
red-seal-ocr/
├── img/                    # 图片文件夹（存放待识别的图片）
├── src/
│   ├── ocr_engine.py      # OCR识别引擎
│   └── preprocess.py      # 图像预处理模块（红色区域提取）
├── config.yaml            # 配置文件（控制预处理开关等）
├── main.py                # 主程序入口
├── check_paddleocr.py     # PaddleOCR安装检测工具
├── requirements.txt        # 依赖包列表
└── README.md              # 说明文档
```

## 配置说明

### 图像预处理配置

编辑 `config.json` 文件可以控制图像预处理功能：

```json
{
  "preprocessing": {
    "enabled": false,  // 是否启用图像预处理（默认关闭）
    "description": "是否启用图像预处理（提取红色区域）"
  }
}
```

**启用预处理的好处：**
- 只识别红色区域，减少干扰
- 提高印章识别准确率
- 减少处理时间

**如何启用：**
1. 打开 `config.json` 文件
2. 将 `"preprocessing.enabled"` 设置为 `true`
3. 保存文件后重新运行程序

### 红色提取参数调整

如果启用预处理后效果不理想，可以调整红色提取参数：

```json
{
  "red_extraction": {
    "hsv_lower": [0, 50, 50],      // 红色HSV下限（色相0-10）
    "hsv_upper": [10, 255, 255],
    "hsv_lower2": [170, 50, 50],   // 红色HSV下限（色相170-180）
    "hsv_upper2": [180, 255, 255]
  },
  "morphology": {
    "kernel_size": 3,              // 形态学操作核大小
    "iterations": 2                // 形态学操作迭代次数
  }
}
```

## 工具脚本

### 检测PaddleOCR安装

运行检测脚本检查PaddleOCR和模型是否已安装：

```bash
python check_paddleocr.py
```

脚本会自动检测：
- PaddleOCR包是否已安装
- PaddlePaddle包是否已安装
- 模型文件是否已下载
- 如果模型未下载，会自动下载

## 注意事项

- 首次运行需要下载模型，可能需要一些时间
- 识别准确率受图片质量影响
- 如果印章位置不固定，程序会扫描整张图片进行识别
- 新版本PaddleOCR（3.x）模型存储在 `~/.paddlex/` 目录
- 图像预处理功能默认关闭，可根据实际效果决定是否启用


## 构建Docker镜像

```bash
# 构建镜像（会在构建时预下载OCR模型）
docker build -t red-seal-ocr:latest .

# 注意：首次构建可能需要较长时间，因为需要下载OCR模型（约几百MB）
```