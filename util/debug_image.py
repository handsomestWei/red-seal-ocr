"""
单图片调试脚本
合并了红色区域提取分析和OCR识别调试功能
用于分析为什么印章没有被识别
"""
import cv2
import numpy as np
from pathlib import Path
import sys
import hashlib
import gc

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocess import RedRegionExtractor, load_config
from ocr_pool import create_ocr_engine_from_config

# 要调试的图片路径（修改这里）
IMAGE_PATH = 'img/1761895215491_28330540b9ce4b5193dc2ae08fb9ff36.jpg'


def get_safe_filename(filepath: str) -> str:
    """
    从文件路径生成ASCII安全的文件名（处理中文文件名乱码问题）
    
    Args:
        filepath: 文件路径
        
    Returns:
        ASCII安全的文件名
    """
    source_path = Path(filepath)
    debug_stem = source_path.stem  # 获取文件名（不含扩展名）
    
    # 将中文文件名转换为ASCII安全的文件名（避免乱码）
    if any(ord(c) > 127 for c in debug_stem):
        # 如果文件名包含非ASCII字符，使用hash生成安全的文件名（仅使用hash，避免中文字符）
        debug_stem_safe = hashlib.md5(debug_stem.encode('utf-8')).hexdigest()
    else:
        # 对ASCII文件名做一次字符清理
        debug_stem_safe = ''.join(c if c.isalnum() or c in ('_', '-', '.') else '_' for c in debug_stem)
    
    return debug_stem_safe


def debug_image(image_path: str):
    """
    调试单张图片
    
    Args:
        image_path: 图片路径
    """
    print("=" * 60)
    print("单图片调试工具")
    print("=" * 60)
    
    # 检查文件是否存在
    if not Path(image_path).exists():
        print(f"❌ 文件不存在: {image_path}")
        return
    
    # 加载配置（优先使用config.yaml，如果不存在则使用config.json）
    config_dir = Path(__file__).parent.parent
    config_yaml = config_dir / 'config.yaml'
    config_json = config_dir / 'config.json'
    if config_yaml.exists():
        config = load_config(str(config_yaml))
    elif config_json.exists():
        config = load_config(str(config_json))
    else:
        config = load_config()  # 使用默认配置
    
    # 生成安全的调试文件名
    debug_stem_safe = get_safe_filename(image_path)
    debug_dir = Path('debug_output')
    debug_dir.mkdir(exist_ok=True)
    
    # ========== 2. 红色区域提取分析 ==========
    print(f"\n2. 红色区域提取分析:")
    
    # 初始化预处理器
    preprocessor = RedRegionExtractor(config)
    
    mask, processed_image, red_debug = preprocessor.extract_red_regions(image_path, return_debug=True)
    
    image_shape = red_debug.get('image_shape')
    file_size_kb = red_debug.get('file_size_kb')
    if image_shape:
        print(f"   图片尺寸: {image_shape}")
    if file_size_kb is not None:
        print(f"   图片大小: {file_size_kb:.1f}KB")
    print(f"   源文件: {image_path}")
    print(f"   调试文件名前缀: {debug_stem_safe}")
    
    hsv_range1 = red_debug.get('hsv_range1', ([0, 0, 0], [0, 0, 0]))
    hsv_range2 = red_debug.get('hsv_range2', ([0, 0, 0], [0, 0, 0]))
    print(f"   HSV范围1: {hsv_range1[0]} - {hsv_range1[1]}")
    print(f"   HSV范围2: {hsv_range2[0]} - {hsv_range2[1]}")
    print(f"   原始红色像素: {red_debug.get('raw_red_pixels', 0)} ({red_debug.get('raw_red_ratio', 0)*100:.2f}%)")
    print(f"   过滤后红色像素: {red_debug.get('filtered_red_pixels', 0)} ({red_debug.get('filtered_red_ratio', 0)*100:.2f}%)")
    print(f"   轮廓数量: {red_debug.get('total_contours', 0)}, 保留: {red_debug.get('kept_contours', 0)}")
    min_area = red_debug.get('min_area', 0)
    min_area_ratio = red_debug.get('min_area_ratio', 0)
    print(f"   最小面积阈值: {min_area:.0f} (比例: {min_area_ratio*100:.4f}%)")
    
    contours_info = red_debug.get('contours', [])
    for idx, info in enumerate(contours_info, 1):
        bbox = info.get('bbox', [0, 0, 0, 0])
        print(f"   轮廓 {idx}: 面积={info.get('area', 0):.0f}, 位置=({bbox[0]}, {bbox[1]}), 尺寸={bbox[2]}x{bbox[3]}, "
              f"宽高比={info.get('aspect_ratio', 0):.2f} [{'保留' if info.get('keep') else '过滤'}]")
    
    if np.sum(mask > 0) > 0:
        hsv_processed = None
        red_pixels = None
        red_pixels_hsv = None
        try:
            hsv_processed = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
            red_pixels = processed_image[mask > 0]
            red_pixels_hsv = hsv_processed[mask > 0]
            print(f"\n   红色区域颜色分析:")
            print(f"     BGR均值: ({np.mean(red_pixels[:, 0]):.1f}, {np.mean(red_pixels[:, 1]):.1f}, {np.mean(red_pixels[:, 2]):.1f})")
            print(f"     HSV均值: (H:{np.mean(red_pixels_hsv[:, 0]):.1f}, S:{np.mean(red_pixels_hsv[:, 1]):.1f}, V:{np.mean(red_pixels_hsv[:, 2]):.1f})")
            print(f"     HSV范围: H[{np.min(red_pixels_hsv[:, 0]):.0f}-{np.max(red_pixels_hsv[:, 0]):.0f}], "
                  f"S[{np.min(red_pixels_hsv[:, 1]):.0f}-{np.max(red_pixels_hsv[:, 1]):.0f}], "
                  f"V[{np.min(red_pixels_hsv[:, 2]):.0f}-{np.max(red_pixels_hsv[:, 2]):.0f}]")
        finally:
            # 释放临时数组
            if hsv_processed is not None:
                del hsv_processed
            if red_pixels is not None:
                del red_pixels
            if red_pixels_hsv is not None:
                del red_pixels_hsv
    
    # ========== 3. OCR识别 ==========
    print(f"\n3. OCR识别:")
    
    # 初始化OCR引擎（调试场景启用本地输出）
    debug_config = {
        'enabled': True,
        'save_red_full': True,
        'output_dir': str(debug_dir)
    }
    
    try:
        # 使用封装好的工具函数创建OCR引擎
        ocr_engine = create_ocr_engine_from_config(config, debug_config=debug_config)
        
        # 运行OCR（使用预处理）
        use_preprocessing = config.get('preprocessing', {}).get('enabled', False)
        print(f"   使用预处理: {'是' if use_preprocessing else '否'}")
        
        ocr_call = ocr_engine.recognize(image_path, use_preprocessing=use_preprocessing, return_debug=True)
        ocr_results, ocr_debug = ocr_call
        
        if not ocr_results:
            print(f"   ⚠️  OCR未识别到任何文字")
        else:
            print(f"   识别到 {len(ocr_results)} 处文字 (使用预处理: {'是' if ocr_debug.get('used_preprocessing') else '否'}, "
                  f"低质量: {'是' if ocr_debug.get('is_low_quality') else '否'})")
            for idx, result in enumerate(ocr_results, 1):
                text = result.get('text', '')
                confidence = result.get('confidence', 0.0)
                bbox = result.get('bbox', [])
                print(f"     {idx}、{text} (置信度: {confidence:.2f})")
                if bbox is not None and len(bbox) > 0:
                    if isinstance(bbox[0], (list, tuple, np.ndarray)):
                        pts = np.array(bbox)
                        x_min = int(pts[:, 0].min())
                        y_min = int(pts[:, 1].min())
                        x_max = int(pts[:, 0].max())
                        y_max = int(pts[:, 1].max())
                        print(f"        位置: ({x_min}, {y_min}) - ({x_max}, {y_max})")
                    elif len(bbox) >= 4:
                        print(f"        位置: {bbox[:4]}")
    except Exception as e:
        print(f"   ❌ OCR识别失败: {e}")
        import traceback
        traceback.print_exc()
        ocr_results = []
    
    # ========== 4. 保存调试图像 ==========
    print(f"\n4. 保存调试图像:")
    
    # 保存红色区域提取结果
    red_extracted_path = debug_dir / f"debug_{debug_stem_safe}_red_extracted.jpg"
    cv2.imwrite(str(red_extracted_path), processed_image)
    print(f"   红色区域提取: {red_extracted_path}")
    
    # 保存掩码图像
    mask_output = debug_dir / f"debug_{debug_stem_safe}_mask.jpg"
    cv2.imwrite(str(mask_output), mask)
    print(f"   掩码图像: {mask_output}")
    
    # 保存完整红色区域图像（加工后）
    red_region_path = debug_dir / f"debug_{debug_stem_safe}_red_region_full.jpg"
    cv2.imwrite(str(red_region_path), processed_image)
    print(f"   完整红色区域: {red_region_path}")
    
    print(f"\n{'='*60}")
    print("调试完成！请检查 debug_output 目录中的图像")
    print(f"{'='*60}")
    
    # 释放大数组，回收内存
    if 'mask' in locals():
        del mask
    if 'processed_image' in locals():
        del processed_image
    if 'ocr_engine' in locals():
        del ocr_engine
    gc.collect()  # 强制垃圾回收


if __name__ == '__main__':
    debug_image(IMAGE_PATH)

