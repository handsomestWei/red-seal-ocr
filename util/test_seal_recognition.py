"""
PaddleOCR印章文本识别模块测试工具
使用专门的印章识别产线进行测试
官方提供的模型，对圆形大印章有效果，对于小印章比如方形的效果较差，需要微调训练区域检测模块等小模型。

安装说明：
1. 如果使用 SealRecognition 产线，可能需要安装 PaddleX OCR 依赖组：
   pip install "paddlex[ocr]"
   
2. 模型会自动下载：
   - 首次使用时，模型会自动从 HuggingFace 下载（默认）
   - 可通过环境变量设置下载源：PADDLE_PDX_MODEL_SOURCE="bos" 或 "modelscope"
   - 模型会保存到用户目录下的 .paddlex 或 .paddleocr 目录

3. 印章检测-推理模型下载链接（两个源均可下载）
    - https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/seal_text_detection.html#_3
    - https://github.com/PaddlePaddle/PaddleOCR/blob/release/3.0/docs/version3.x/model_list.md

使用方法：
    python util/test_seal_recognition.py [图片路径]
    
    如果不提供图片路径，将尝试使用默认测试图片
"""
import sys
from pathlib import Path
import time
import numpy as np
from loguru import logger

# 添加项目根目录到路径（用于导入项目模块）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# 配置日志
logger.remove()
logger.add(
    sys.stderr,
    level="DEBUG",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)

# 导入PaddleOCR相关模块
try:
    from paddleocr import PaddleOCR
    # 尝试导入印章识别模块
    try:
        from paddleocr._pipelines.seal_recognition import SealRecognition
        SEAL_RECOGNITION_AVAILABLE = True
    except ImportError:
        SEAL_RECOGNITION_AVAILABLE = False
        logger.warning("无法导入SealRecognition模块")
        logger.warning("💡 提示: 可能需要安装 PaddleX OCR 依赖组")
        logger.warning("   请运行: pip install \"paddlex[ocr]\"")
except ImportError as e:
    logger.error(f"导入失败: {e}")
    logger.error("请确保已安装: pip install paddleocr")
    sys.exit(1)


def test_seal_recognition(image_path: str, model_dir: str = None):
    """
    测试使用PaddleOCR的印章文本识别产线
    
    Args:
        image_path: 图片路径
        model_dir: 本地模型目录路径（可选，如果不提供则使用默认路径）
    
    Returns:
        识别结果，如果失败返回None
    """
    logger.info(f"开始测试印章文本识别: {image_path}")
    
    if not SEAL_RECOGNITION_AVAILABLE:
        logger.error("❌ SealRecognition模块不可用")
        logger.error("请先安装: pip install \"paddlex[ocr]\"")
        return None
    
    try:
        logger.info("使用SealRecognition类进行印章识别...")
        
        # 确定模型路径
        if model_dir is None:
            # 使用默认路径：项目根目录下的 model/PP-OCRv4_server_seal_det
            model_dir = project_root / 'model' / 'PP-OCRv4_server_seal_det'
        else:
            model_dir = Path(model_dir)
        
        if not model_dir.exists():
            logger.error(f"本地模型路径不存在: {model_dir}")
            logger.error("请确保模型目录存在")
            logger.info("💡 提示: 模型目录应包含 inference.yml 等模型文件")
            return None
        
        seal_text_detection_model_dir = str(model_dir)
        logger.info(f"使用本地印章检测模型目录: {seal_text_detection_model_dir}")
        
        # 创建印章识别实例，直接指定本地模型路径
        seal_ocr = SealRecognition(
            seal_text_detection_model_dir=seal_text_detection_model_dir
        )
        logger.info("✅ 成功创建印章识别实例")
        
        # 进行识别
        start_time = time.time()
        result = seal_ocr.predict(image_path)
        elapsed_time = time.time() - start_time
        
        logger.info(f"识别完成，耗时: {elapsed_time:.2f}秒")
        logger.info(f"识别结果类型: {type(result)}")
        
        # 解析并打印识别结果（根据 PaddleX 文档格式）
        if result:
            # SealRecognition 返回的是列表，每个元素是一个结果字典
            if isinstance(result, list) and len(result) > 0:
                res_dict = result[0]
            elif isinstance(result, dict):
                res_dict = result
            else:
                logger.warning(f"未知的结果格式: {type(result)}")
                return result
            
            # 打印模型设置信息
            model_settings = res_dict.get('model_settings', {})
            if model_settings:
                logger.info("模型设置:")
                for key, value in model_settings.items():
                    logger.info(f"  - {key}: {value}")
            
            # 打印文档预处理结果（如果有）
            doc_preprocessor_res = res_dict.get('doc_preprocessor_res', {})
            if doc_preprocessor_res:
                angle = doc_preprocessor_res.get('angle', None)
                if angle is not None and angle != -1:
                    logger.info(f"文档方向: {angle}°")
            
            # 打印布局检测结果
            layout_det_res = res_dict.get('layout_det_res', {})
            if layout_det_res:
                boxes = layout_det_res.get('boxes', [])
                if boxes:
                    logger.info(f"布局检测结果: 检测到 {len(boxes)} 个区域")
                    for idx, box in enumerate(boxes, 1):
                        label = box.get('label', 'unknown')
                        score = box.get('score', 0.0)
                        cls_id = box.get('cls_id', -1)
                        coordinate = box.get('coordinate', [])
                        logger.info(f"  区域 {idx}: {label} (类别ID: {cls_id}, 置信度: {score:.4f})")
                        if coordinate:
                            logger.info(f"    坐标: {coordinate}")
            
            # 检查 seal_res_list（印章识别结果列表）
            seal_res_list = res_dict.get('seal_res_list', [])
            if seal_res_list:
                logger.info("="*60)
                logger.info(f"✅ 印章识别结果: 识别到 {len(seal_res_list)} 个印章区域")
                logger.info("="*60)
                
                for idx, seal_res in enumerate(seal_res_list, 1):
                    logger.info(f"\n印章区域 {idx}:")
                    
                    # 打印检测参数
                    text_det_params = seal_res.get('text_det_params', {})
                    if text_det_params:
                        logger.info(f"  检测参数: {text_det_params}")
                    
                    # 打印检测到的多边形框
                    dt_polys = seal_res.get('dt_polys', [])
                    if dt_polys:
                        logger.info(f"  检测到 {len(dt_polys)} 个文本区域")
                    
                    # 打印识别文本和置信度
                    rec_texts = seal_res.get('rec_texts', [])
                    rec_scores = seal_res.get('rec_scores', [])
                    
                    if rec_texts:
                        # 处理 rec_scores（可能是 numpy array 或 list）
                        if isinstance(rec_scores, np.ndarray):
                            rec_scores = rec_scores.tolist()
                        elif not isinstance(rec_scores, list):
                            rec_scores = []
                        
                        logger.info(f"  识别到的文字 ({len(rec_texts)} 条):")
                        for i, text in enumerate(rec_texts):
                            score = rec_scores[i] if i < len(rec_scores) else 0.0
                            # 格式化置信度为百分比
                            score_percent = score * 100 if score <= 1.0 else score
                            logger.info(f"    [{i+1}] {text} (置信度: {score_percent:.2f}%)")
                        
                        # 打印所有文字（合并）
                        all_text = "".join(rec_texts)
                        logger.info(f"  合并文字: {all_text}")
                    else:
                        logger.warning(f"  印章区域 {idx}: 未识别到文字")
                    
                    # 打印文本类型
                    text_type = seal_res.get('text_type', 'unknown')
                    if text_type:
                        logger.info(f"  文本类型: {text_type}")
            else:
                logger.warning("="*60)
                logger.warning("⚠️ 未识别到印章文字（seal_res_list 为空）")
                logger.warning("="*60)
                logger.info("提示: 可能的原因：")
                logger.info("  1. 图片中没有印章")
                logger.info("  2. 印章区域未被布局检测识别为 'seal' 类型")
                logger.info("  3. 印章检测阈值设置过高")
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ 印章识别失败: {error_msg}")
        import traceback
        logger.error("错误堆栈:")
        for line in traceback.format_exception(type(e), e, e.__traceback__):
            logger.error(line.rstrip())
        if "dependency" in error_msg.lower() or "依赖" in error_msg:
            logger.warning("💡 提示: 可能需要安装 PaddleX OCR 依赖组")
            logger.warning("   请运行: pip install \"paddlex[ocr]\"")
        return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试PaddleOCR印章文本识别模块')
    parser.add_argument(
        'image_path',
        nargs='?',
        help='图片路径（可选，如果不提供将尝试使用默认测试图片）'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='本地模型目录路径（可选，默认使用项目根目录下的 model/PP-OCRv4_server_seal_det）'
    )
    
    args = parser.parse_args()
    
    # 确定图片路径
    if args.image_path:
        image_path = Path(args.image_path)
        if not image_path.exists():
            logger.error(f"图片不存在: {image_path}")
            return 1
        image_path = str(image_path)
    else:
        # 尝试使用默认测试图片
        default_images = [
            project_root / 'img' / 'xxx.jpg'
        ]
        
        image_path = None
        for path in default_images:
            if path.exists():
                image_path = str(path)
                break
        
        if image_path is None:
            logger.error("未找到测试图片，请提供图片路径")
            logger.error("使用方法: python util/test_seal_recognition.py <图片路径>")
            return 1
    
    logger.info("="*60)
    logger.info("测试PaddleOCR印章文本识别模块")
    logger.info("="*60)
    logger.info(f"图片路径: {image_path}")
    if args.model_dir:
        logger.info(f"模型目录: {args.model_dir}")
    logger.info("")
    
    # 执行测试
    result = test_seal_recognition(image_path, args.model_dir)
    
    logger.info("\n" + "="*60)
    logger.info("测试总结")
    logger.info("="*60)
    if result:
        logger.info("✅ 印章识别测试成功")
        return 0
    else:
        logger.info("❌ 印章识别测试失败")
        return 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.error("\n❌ 用户中断了程序")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ 发生异常: {type(e).__name__}: {e}")
        import traceback
        for line in traceback.format_exception(type(e), e, e.__traceback__):
            logger.error(line.rstrip())
        sys.exit(1)

