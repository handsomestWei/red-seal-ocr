"""
主程序入口
识别图片中的印章文字，判断页面是否有效
"""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from loguru import logger

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocess import load_config
from ocr_pool import create_ocr_engine_from_config
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocr_engine import OCREngine

def setup_logging(log_config: dict):
    """
    配置日志系统
    
    Args:
        log_config: 日志配置字典
    """
    # 移除默认的日志处理器
    logger.remove()
    
    # 获取配置
    log_file = log_config.get('log_file', 'logs/main.log')
    log_level = log_config.get('level', 'INFO')
    console_output = log_config.get('console_output', True)
    max_file_size_mb = log_config.get('max_file_size_mb', 100)
    retention_count = log_config.get('retention_count', 10)
    compression = log_config.get('compression', True)
    log_format = log_config.get('format', "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | [线程:{thread.id}] | {name}:{function}:{line} - {message}")
    
    # 确保日志目录存在
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 配置文件日志（支持轮转、压缩）
    logger.add(
        log_file,
        rotation=f"{max_file_size_mb} MB",  # 文件大小达到限制时轮转
        retention=retention_count,  # 保留的文件数量
        compression="zip" if compression else None,  # 压缩旧日志
        level=log_level,
        format=log_format,
        encoding="utf-8",
        enqueue=True,  # 异步写入，提高性能
        backtrace=True,  # 显示完整的错误堆栈
        diagnose=True  # 显示变量值
    )
    
    # 配置控制台输出（可选）
    if console_output:
        logger.add(
            sys.stderr,
            level=log_level,
            format=log_format,
            colorize=True  # 彩色输出
        )
    
    logger.info("日志系统初始化完成")


def process_image(image_path: str, ocr_engine: 'OCREngine', use_preprocessing: bool = False) -> dict:
    """
    处理单张图片
    
    Args:
        image_path: 图片路径
        ocr_engine: OCR引擎实例
        use_preprocessing: 是否使用预处理
    
    Returns:
        dict: 包含图片路径、识别结果、耗时等信息的字典
    """
    logger.info(f"处理图片: {image_path}")
    if use_preprocessing:
        logger.debug("⚠️  已启用图像预处理（红色区域提取）")
    
    result = {
        'image_path': image_path,
        'success': False,
        'results': [],
        'total': 0,
        'elapsed_time': 0.0,
        'error': None
    }
    
    try:
        # OCR识别
        logger.debug("正在进行OCR识别...")
        ocr_start_time = time.time()
        ocr_results = ocr_engine.recognize(image_path, use_preprocessing=use_preprocessing)
        ocr_elapsed_time = time.time() - ocr_start_time
        
        result['success'] = True
        result['results'] = ocr_results
        result['total'] = len(ocr_results)
        result['elapsed_time'] = ocr_elapsed_time
        
        logger.info(f"OCR识别完成，识别到 {len(ocr_results)} 处文字，耗时: {ocr_elapsed_time:.2f}秒")
        
        # 显示结果（详细信息使用debug级别）
        if ocr_results:
            logger.debug("识别结果详情:")
            for idx, result_item in enumerate(ocr_results, 1):
                text = result_item.get('text', '')
                confidence = result_item.get('confidence', 0.0)
                bbox = result_item.get('bbox', [])
                logger.debug(f"  {idx}、{text} (置信度: {confidence:.2f})")
                if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) > 0:
                    if isinstance(bbox[0], (list, tuple, np.ndarray)):
                        pts = np.array(bbox)
                        x_min = int(pts[:, 0].min())
                        y_min = int(pts[:, 1].min())
                        x_max = int(pts[:, 0].max())
                        y_max = int(pts[:, 1].max())
                        logger.debug(f"     位置: ({x_min}, {y_min}) - ({x_max}, {y_max})")
                    elif len(bbox) >= 4:
                        logger.debug(f"     位置: {bbox[:4]}")
        else:
            logger.warning("⚠️  未识别到任何文字")
        
    except Exception as e:
        error_msg = str(e)
        result['error'] = error_msg
        logger.error(f"❌ 处理图片时出错: {error_msg}", exc_info=True)
    
    return result


def main():
    """主函数"""
    # 先加载配置（用于日志配置）
    config_dir = Path(__file__).parent
    config_yaml = config_dir / 'config.yaml'
    config_json = config_dir / 'config.json'
    if config_yaml.exists():
        config = load_config(str(config_yaml))
    elif config_json.exists():
        config = load_config(str(config_json))
    else:
        config = load_config()  # 使用默认配置
    
    # 配置日志系统
    log_config = config.get('logging', {})
    # 如果未指定日志文件，使用main.log
    if 'log_file' not in log_config or not log_config['log_file']:
        log_config['log_file'] = 'logs/main.log'
    setup_logging(log_config)
    
    logger.info("印章OCR识别系统")
    
    # 检查是否启用预处理
    preprocessing_enabled = config.get('preprocessing', {}).get('enabled', False)
    
    if preprocessing_enabled:
        logger.debug("\n正在初始化图像预处理器...")
        logger.info("✅ 图像预处理器已初始化（红色区域提取）")
    
    # 获取图片优化配置
    optimization_config = config.get('image_optimization', {})
    optimization_enabled = optimization_config.get('enabled', True)
    
    # 获取图片增强配置
    enhancement_config = config.get('image_enhancement', {})
    
    # 初始化OCR引擎（使用封装好的工具函数）
    logger.debug("\n正在初始化PaddleOCR引擎（首次运行会下载模型，请稍候）...")
    ocr_engine = create_ocr_engine_from_config(config)
    
    # 打印OCR模型版本信息
    version_info = []
    if ocr_engine.paddleocr_version:
        version_info.append(f"PaddleOCR: {ocr_engine.paddleocr_version}")
    if ocr_engine.paddlepaddle_version:
        version_info.append(f"PaddlePaddle: {ocr_engine.paddlepaddle_version}")
    
    if version_info:
        logger.info(f"OCR模型版本: {', '.join(version_info)}")
    else:
        logger.info("OCR模型版本: 无法获取版本信息")
    
    logger.debug(f"\n配置信息详情:")
    logger.debug(f"  图像预处理: {'✅ 已启用' if preprocessing_enabled else '❌ 未启用'}")
    if preprocessing_enabled:
        logger.debug(f"    ⚠️  提示：启用预处理可以更好地识别红色印章文字")
    logger.debug(f"  图片压缩优化: {'✅ 已启用' if optimization_enabled else '❌ 未启用'}")
    if optimization_enabled:
        logger.debug(f"    最大尺寸: {optimization_config.get('max_size', 2000)}px")
        logger.debug(f"    JPEG质量: {optimization_config.get('jpeg_quality', 95)}")
    
    enhancement_enabled = enhancement_config.get('enabled', True)
    logger.debug(f"  图片增强（小质量图片）: {'✅ 已启用' if enhancement_enabled else '❌ 未启用'}")
    if enhancement_enabled:
        logger.debug(f"    质量阈值: 文件<{enhancement_config.get('size_threshold_kb', 200)}KB 或 分辨率<{enhancement_config.get('resolution_threshold', 1200)}px")
        skip_preprocessing = enhancement_config.get('skip_preprocessing_for_low_quality', False)
        if skip_preprocessing:
            logger.debug(f"    处理策略: 增强 → 直接OCR（跳过红色区域提取）")
        else:
            logger.debug(f"    处理策略: 增强 → 红色区域提取 → OCR（推荐）")
    
    # 获取图片目录
    img_dir = Path(__file__).parent / 'img'
    
    if not img_dir.exists():
        logger.error(f"❌ 图片目录不存在: {img_dir}")
        return
    
    # 获取所有图片文件（去重，因为Windows不区分大小写）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = set()  # 使用set自动去重
    for ext in image_extensions:
        image_files.update(img_dir.glob(f'*{ext}'))
        image_files.update(img_dir.glob(f'*{ext.upper()}'))
    
    # 转换为列表并排序
    image_files = sorted(list(image_files))
    
    if not image_files:
        logger.error(f"❌ 在 {img_dir} 目录下未找到图片文件")
        return
    
    logger.info(f"\n找到 {len(image_files)} 张图片，开始处理...\n")
    
    # 处理每张图片并收集结果
    total_start_time = time.time()
    all_results = []
    for idx, image_file in enumerate(image_files, 1):
        logger.debug(f"\n[进度: {idx}/{len(image_files)}]")
        result = process_image(str(image_file), ocr_engine, use_preprocessing=preprocessing_enabled)
        all_results.append(result)
    
    total_elapsed = time.time() - total_start_time
    avg_time = total_elapsed / len(image_files) if len(image_files) > 0 else 0
    logger.info(f"处理完成！共处理: {len(image_files)} 张图片，总耗时: {total_elapsed:.2f}秒，平均每张: {avg_time:.2f}秒")
    
    # 输出结果到txt文件
    output_file = Path(__file__).parent / 'ocr_results.txt'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("OCR识别结果汇总\n")
            f.write("=" * 80 + "\n")
            f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总图片数: {len(image_files)}\n")
            f.write(f"总耗时: {total_elapsed:.2f}秒\n")
            f.write(f"平均每张: {avg_time:.2f}秒\n")
            f.write("=" * 80 + "\n\n")
            
            # 统计信息
            success_count = sum(1 for r in all_results if r['success'])
            total_text_count = sum(r['total'] for r in all_results)
            f.write(f"成功处理: {success_count}/{len(image_files)} 张\n")
            f.write(f"识别到文字总数: {total_text_count} 处\n")
            f.write("=" * 80 + "\n\n")
            
            # 详细结果
            for idx, result in enumerate(all_results, 1):
                image_path = result['image_path']
                image_name = Path(image_path).name
                f.write(f"\n[{idx}/{len(image_files)}] {image_name}\n")
                f.write(f"文件路径: {image_path}\n")
                
                if result['success']:
                    f.write(f"状态: ✅ 成功 (耗时: {result['elapsed_time']:.2f}秒)\n")
                    if result['total'] > 0:
                        f.write(f"识别到 {result['total']} 处文字:\n")
                        for i, text_result in enumerate(result['results'], 1):
                            text = text_result.get('text', '')
                            confidence = text_result.get('confidence', 0.0)
                            f.write(f"  {i}. {text} (置信度: {confidence:.2f})\n")
                        
                    else:
                        f.write("识别结果: ⚠️ 未识别到任何文字\n")
                else:
                    f.write(f"状态: ❌ 失败\n")
                    f.write(f"错误信息: {result.get('error', '未知错误')}\n")
                
                f.write("-" * 80 + "\n")
        
        logger.info(f"识别结果已保存到: {output_file}")
    except Exception as e:
        logger.error(f"保存结果文件失败: {str(e)}", exc_info=True)


if __name__ == '__main__':
    main()

