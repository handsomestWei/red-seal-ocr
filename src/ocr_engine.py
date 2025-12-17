"""
OCR识别引擎模块
使用PaddleOCR进行文字识别
"""
from paddleocr import PaddleOCR
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import tempfile
import os
import time
from loguru import logger
from preprocess import ImageEnhancer
from PIL import Image


class OCREngine:
    """OCR识别引擎类"""
    
    def __init__(self, use_angle_cls=True, lang='ch', preprocessor=None, optimization_config=None, ocr_performance_config=None, enhancement_config=None, debug_config=None):
        """
        初始化OCR引擎
        
        Args:
            use_angle_cls: 是否使用角度分类器
            lang: 语言类型，'ch'表示中英文混合
            preprocessor: 图像预处理器（RedRegionExtractor实例），如果为None则不使用预处理
            optimization_config: 图片优化配置字典，包含enabled、max_size、jpeg_quality等
            ocr_performance_config: OCR性能优化配置字典
            enhancement_config: 图片增强配置字典，用于小质量图片增强
        """
        # OCR性能优化配置
        if ocr_performance_config is None:
            ocr_performance_config = {}
        
        # 保存配置以便重新初始化时使用
        self.lang = lang
        self.ocr_performance_config = ocr_performance_config
        self.use_angle_cls = use_angle_cls
        self.preprocessor = preprocessor
        self.optimization_config = optimization_config
        self.enhancement_config = enhancement_config
        self.debug_config = debug_config
        
        use_fast_model = ocr_performance_config.get('use_fast_model', True)
        
        # 获取性能参数（在初始化时使用）
        text_recognition_batch_size = ocr_performance_config.get('text_recognition_batch_size', 
                                                                 ocr_performance_config.get('rec_batch_num', 6))
        text_det_limit_side_len = ocr_performance_config.get('text_det_limit_side_len', None)
        use_textline_orientation = ocr_performance_config.get('use_textline_orientation', use_angle_cls)
        # OCR版本：PP-OCRv3, PP-OCRv4, PP-OCRv5（根据源码第51行）
        # 如果不指定，会根据语言自动选择（中文默认PP-OCRv5）
        ocr_version = ocr_performance_config.get('ocr_version', None)
        # 文本检测模型名称（可选，用于指定具体的检测模型）
        text_detection_model_name = ocr_performance_config.get('text_detection_model_name', None)
        # 文本检测模型目录（可选，用于使用本地模型）
        text_detection_model_dir = ocr_performance_config.get('text_detection_model_dir', None)
        # 文本识别模型名称（可选，用于指定具体的识别模型）
        text_recognition_model_name = ocr_performance_config.get('text_recognition_model_name', None)
        # 文本识别模型目录（可选，用于使用本地模型）
        text_recognition_model_dir = ocr_performance_config.get('text_recognition_model_dir', None)
        # CPU线程数（默认值：8）
        cpu_threads = ocr_performance_config.get('cpu_threads', None)
        # 计算精度（默认值：fp32，可选：fp16）
        precision = ocr_performance_config.get('precision', None)
        
        # 构建PaddleOCR初始化参数（根据源码第56-84行）
        ocr_params = {
            'lang': lang,
            'use_textline_orientation': use_textline_orientation,  # 新版本参数名
            'text_recognition_batch_size': text_recognition_batch_size,  # 识别批处理大小
        }
        
        # 添加文本检测模型名称（如果指定了，优先使用）
        if text_detection_model_name is not None:
            ocr_params['text_detection_model_name'] = text_detection_model_name
            logger.debug(f"指定文本检测模型名称: {text_detection_model_name}")
        
        # 添加文本检测模型目录（如果指定了）
        if text_detection_model_dir is not None:
            ocr_params['text_detection_model_dir'] = text_detection_model_dir
            logger.debug(f"指定文本检测模型目录: {text_detection_model_dir}")
        
        # 添加文本识别模型名称（如果指定了，优先使用）
        if text_recognition_model_name is not None:
            ocr_params['text_recognition_model_name'] = text_recognition_model_name
            logger.debug(f"指定文本识别模型名称: {text_recognition_model_name}")
        
        # 添加文本识别模型目录（如果指定了）
        if text_recognition_model_dir is not None:
            ocr_params['text_recognition_model_dir'] = text_recognition_model_dir
            logger.debug(f"指定文本识别模型目录: {text_recognition_model_dir}")
        
        # 添加OCR版本（如果指定了且未指定检测/识别模型名称）
        if ocr_version is not None and text_detection_model_name is None and text_recognition_model_name is None:
            ocr_params['ocr_version'] = ocr_version
            logger.debug(f"指定OCR版本: {ocr_version}")
        elif ocr_version is not None and (text_detection_model_name is not None or text_recognition_model_name is not None):
            logger.debug(f"已指定检测/识别模型名称，忽略 ocr_version 参数")
        else:
            logger.debug("未指定OCR版本，将根据语言自动选择（中文默认PP-OCRv5）")
        
        # 添加检测边长限制（如果配置了，可以大幅提升速度）
        if text_det_limit_side_len is not None:
            ocr_params['text_det_limit_side_len'] = text_det_limit_side_len
        
        # 添加CPU线程数（如果配置了）
        if cpu_threads is not None:
            ocr_params['cpu_threads'] = cpu_threads
            logger.debug(f"设置CPU线程数: {cpu_threads}")
        else:
            logger.debug("使用默认CPU线程数: 8")
        
        # 添加计算精度（如果配置了）
        # 注意：fp16 可能导致某些环境下崩溃，如果遇到问题可以尝试改为 fp32
        if precision is not None:
            ocr_params['precision'] = precision
            logger.debug(f"设置计算精度: {ocr_params.get('precision', 'fp32')}")
        else:
            logger.debug("使用默认计算精度: fp32")
        
        # 获取版本信息
        self.paddleocr_version = None
        self.paddlepaddle_version = None
        try:
            import paddleocr
            self.paddleocr_version = getattr(paddleocr, '__version__', None)
        except ImportError:
            pass
        
        try:
            import paddle
            self.paddlepaddle_version = getattr(paddle, '__version__', None)
        except ImportError:
            pass
        
        # 初始化PaddleOCR（根据源码，直接使用正确的参数名）
        logger.debug(f"初始化PaddleOCR: {ocr_params}")
        self.ocr = PaddleOCR(**ocr_params)
        logger.debug("PaddleOCR初始化成功")
        
        self.preprocessor = preprocessor
        logger.debug(f"预处理器: {'已配置' if preprocessor is not None else '未配置'}")
        
        # 图片优化配置
        if optimization_config is None:
            optimization_config = {}
        self.optimization_enabled = optimization_config.get('enabled', True)
        self.max_size = optimization_config.get('max_size', 2000)
        self.jpeg_quality = optimization_config.get('jpeg_quality', 95)
        logger.debug(f"图片优化配置: enabled={self.optimization_enabled}, max_size={self.max_size}, jpeg_quality={self.jpeg_quality}")
        
        # 图片增强配置（用于小质量图片）
        if enhancement_config is None:
            enhancement_config = {}
        enhancement_enabled = enhancement_config.get('enabled', True)
        if enhancement_enabled:
            # 合并所有配置，让ImageEnhancer可以访问
            full_config = {'image_enhancement': enhancement_config}
            self.image_enhancer = ImageEnhancer(full_config)
            # 质量检测阈值
            self.low_quality_size_threshold = enhancement_config.get('size_threshold_kb', 200)
            self.low_quality_resolution_threshold = enhancement_config.get('resolution_threshold', 1200)
            # 小质量图片是否跳过红色区域提取
            self.skip_preprocessing_for_low_quality = enhancement_config.get('skip_preprocessing_for_low_quality', True)
            logger.debug(f"图片增强配置: enabled=True, size_threshold={self.low_quality_size_threshold}KB, "
                        f"resolution_threshold={self.low_quality_resolution_threshold}, "
                        f"skip_preprocessing_for_low_quality={self.skip_preprocessing_for_low_quality}")
        else:
            self.image_enhancer = None
            logger.debug("图片增强配置: enabled=False")
        
        # 调试输出配置
        if debug_config is None:
            debug_config = {}
        self.debug_enabled = debug_config.get('enabled', False)
        self.debug_save_red_full = debug_config.get('save_red_full', False)
        self.debug_output_dir = Path(debug_config.get('output_dir', 'debug_output'))
        self._ensure_debug_dirs()
        
        # OCR性能参数（用于调用时传递）
        self.det_db_thresh = ocr_performance_config.get('det_db_thresh', 0.3)
        self.det_db_box_thresh = ocr_performance_config.get('det_db_box_thresh', 0.5)
        self.det_db_unclip_ratio = ocr_performance_config.get('det_db_unclip_ratio', 1.6)
        
        # OCR性能参数（用于初始化时设置）
        # text_recognition_batch_size: 识别批处理大小（旧名称rec_batch_num已弃用）
        self.text_recognition_batch_size = ocr_performance_config.get('text_recognition_batch_size', 
                                                                      ocr_performance_config.get('rec_batch_num', 6))
        # text_det_limit_side_len: 限制检测模型的输入边长，可以大幅提升速度
        self.text_det_limit_side_len = ocr_performance_config.get('text_det_limit_side_len', None)
        # use_textline_orientation: 是否使用文本行方向分类（如果不需要可以关闭以提升速度）
        self.use_textline_orientation = ocr_performance_config.get('use_textline_orientation', use_angle_cls)
        
        logger.debug(f"OCR性能参数（调用时）: det_db_thresh={self.det_db_thresh}, det_db_box_thresh={self.det_db_box_thresh}, "
                    f"det_db_unclip_ratio={self.det_db_unclip_ratio}")
        logger.debug(f"OCR性能参数（初始化时）: text_recognition_batch_size={self.text_recognition_batch_size}, "
                    f"text_det_limit_side_len={self.text_det_limit_side_len}, "
                    f"use_textline_orientation={self.use_textline_orientation}")
    
    def _reinitialize_ocr(self):
        """
        重新初始化OCR实例（用于处理假死情况）
        调用 ocr_pool.py 中的统一方法进行重新初始化
        """
        # 延迟导入，避免循环导入
        from ocr_pool import reinitialize_ocr_engine
        
        # 调用统一管理的重新初始化方法
        reinitialize_ocr_engine(self)
    
    def _read_image_with_chinese_path(self, image_path: str) -> Optional[np.ndarray]:
        """
        读取图片（支持中文路径）
        
        Args:
            image_path: 图片路径
            
        Returns:
            图像数组，如果读取失败返回None
        """
        try:
            # 使用PIL读取图片（支持中文路径）
            with Image.open(image_path) as img:
                # 转换为RGB格式（如果是RGBA等）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # 转换为numpy数组
                img_array = np.array(img)
                # PIL读取的是RGB，需要转换为BGR（OpenCV格式）
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                return img_bgr
        except Exception as e:
            # 如果PIL读取失败，尝试使用OpenCV（可能失败）
            try:
                return cv2.imread(image_path)
            except:
                return None
    
    def _ensure_debug_dirs(self):
        if not self.debug_enabled:
            return
        try:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    
    def _optimize_image_size(self, image: np.ndarray, max_size: int = None) -> np.ndarray:
        """
        优化图片大小，如果图片太大则缩放
        
        Args:
            image: 图像数组
            max_size: 最大尺寸，如果为None则使用配置中的max_size
            
        Returns:
            优化后的图像
        """
        if not self.optimization_enabled:
            return image
        
        if max_size is None:
            max_size = self.max_size
        
        h, w = image.shape[:2]
        if max(h, w) <= max_size:
            logger.debug(f"图片尺寸优化: {w}x{h} <= {max_size}, 无需缩放")
            return image
        
        # 计算缩放比例
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        logger.debug(f"图片尺寸优化: {w}x{h} -> {new_w}x{new_h}, 缩放比例={scale:.3f}")
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # 注意：如果原图很大，这里创建了新数组，原图应该由调用者管理释放
        return resized
    
    def _recognize_core(self, original_image: np.ndarray, use_preprocessing: bool, 
                       image_path: Optional[str] = None, return_debug: bool = False):
        """
        核心识别逻辑（内部方法，供recognize和recognize_from_array复用）
        
        Args:
            original_image: 原始图像数组（BGR格式）
            use_preprocessing: 是否使用预处理
            image_path: 图片路径（可选，用于文件大小检测和调试保存）
            return_debug: 是否返回调试信息
            
        Returns:
            OCR识别结果列表，如果return_debug=True则返回(results, debug_info)元组
        """
        if original_image is None or original_image.size == 0:
            return ([], {}) if return_debug else []
        
        debug_info = {
            'image_shape': original_image.shape,
            'used_preprocessing': False,
            'is_low_quality': False,
            'enhancement_applied': False,
            'scale_x': 1.0,
            'scale_y': 1.0,
            'red_ratio': None,
            'file_size_kb': None,
            'ocr_source': 'original'
        }
        
        # 检测是否为小质量图片
        is_low_quality = False
        if self.image_enhancer is not None:
            if image_path:
                # 有文件路径时，可以检测文件大小
                is_low_quality = ImageEnhancer.is_low_quality_image(
                    image_path, 
                    original_image,
                    size_threshold_kb=self.low_quality_size_threshold,
                    resolution_threshold=self.low_quality_resolution_threshold
                )
                logger.debug(f"图片质量检测（有路径）: is_low_quality={is_low_quality}, path={image_path}")
            else:
                # 无文件路径时，仅基于分辨率判断
                h, w = original_image.shape[:2]
                resolution = max(h, w)
                is_low_quality = resolution < self.low_quality_resolution_threshold
                logger.debug(f"图片质量检测（无路径）: 尺寸={w}x{h}, resolution={resolution}, "
                           f"threshold={self.low_quality_resolution_threshold}, is_low_quality={is_low_quality}")
        debug_info['is_low_quality'] = is_low_quality
        
        # 对于小质量图片，先进行增强
        enhancement_applied = False
        if is_low_quality and self.image_enhancer is not None:
            logger.debug("开始图片增强处理...")
            enhance_start = time.time()
            original_image = self.image_enhancer.enhance_image(original_image)
            enhance_elapsed = time.time() - enhance_start
            enhancement_applied = True
            logger.debug(f"图片增强完成，耗时: {enhance_elapsed:.3f}秒")
            # 小质量图片策略：增强 → 红色区域提取 → OCR
            if self.skip_preprocessing_for_low_quality:
                use_preprocessing = False
                logger.debug("低质量图片策略: 跳过红色区域提取（skip_preprocessing_for_low_quality=True）")
            elif self.preprocessor is not None:
                use_preprocessing = True
                logger.debug("低质量图片策略: 增强后继续红色区域提取")
        
        debug_info['enhancement_applied'] = enhancement_applied
        image_for_ocr: Optional[np.ndarray] = None
        
        # 如果启用预处理且有预处理器
        if use_preprocessing and self.preprocessor is not None:
            logger.debug("开始红色区域提取...")
            preprocess_start = time.time()
            debug_info['used_preprocessing'] = True
            mask = None
            processed_image = None
            try:
                # 提取红色区域（总是使用数组方式，更统一）
                extraction = self.preprocessor.extract_red_regions_from_array(original_image, return_debug=True)
                mask, processed_image, red_debug = extraction
                preprocess_elapsed = time.time() - preprocess_start
                
                debug_info['red_ratio'] = red_debug.get('filtered_red_ratio')
                debug_info['file_size_kb'] = red_debug.get('file_size_kb')
                
                # 格式化红色区域比例（处理None值）
                red_ratio_str = f"{debug_info['red_ratio']:.4f}" if debug_info['red_ratio'] is not None else 'N/A'
                # 格式化处理后尺寸
                processed_shape_str = str(processed_image.shape) if processed_image is not None else 'N/A'
                
                logger.debug(f"红色区域提取完成: 耗时={preprocess_elapsed:.3f}秒, "
                            f"红色区域比例={red_ratio_str}, 处理后尺寸={processed_shape_str}")
                
                if self.debug_enabled and self.debug_save_red_full and image_path:
                    self._save_red_full_image(image_path, processed_image)
                    logger.debug(f"调试图片已保存: {image_path}")
                
                image_for_ocr = processed_image
                debug_info['ocr_source'] = 'preprocessed'
            finally:
                # 释放临时变量（mask 在后续不需要）
                if mask is not None:
                    del mask
        else:
            # 直接使用原图进行OCR（可能是增强后的图片）
            image_for_ocr = original_image
            if is_low_quality and self.image_enhancer is not None:
                debug_info['ocr_source'] = 'enhanced'
                logger.debug("使用增强后的原图进行OCR（未进行红色区域提取）")
            else:
                debug_info['ocr_source'] = 'original'
                logger.debug("使用原始图片进行OCR（未进行预处理）")
        
        # 执行OCR识别
        if image_for_ocr is not None:
            h, w = image_for_ocr.shape[:2]
            logger.debug(f"开始OCR识别: 输入图片尺寸={w}x{h}, 来源={debug_info['ocr_source']}")
            ocr_start = time.time()
            
            # 记录OCR输入图片尺寸（用于调试）
            if return_debug:
                debug_info['ocr_input_shape'] = image_for_ocr.shape
                # 检查图片是否有效（不是全黑或全白）
                if len(image_for_ocr.shape) == 3:
                    gray = cv2.cvtColor(image_for_ocr, cv2.COLOR_BGR2GRAY)
                    mean_val = np.mean(gray)
                    std_val = np.std(gray)
                    debug_info['ocr_input_mean'] = float(mean_val)
                    debug_info['ocr_input_std'] = float(std_val)
                    logger.debug(f"OCR输入图片统计: 均值={mean_val:.1f}, 标准差={std_val:.1f}")
                    # 如果标准差很小，说明图片可能全黑或全白
                    if std_val < 5:
                        debug_info['ocr_input_warning'] = f'图片可能全黑/全白 (均值={mean_val:.1f}, 标准差={std_val:.1f})'
                        logger.warning(f"⚠️ 图片可能全黑/全白: 均值={mean_val:.1f}, 标准差={std_val:.1f}")
            try:
                ocr_results = self._perform_ocr_on_image(image_for_ocr)
                ocr_elapsed = time.time() - ocr_start
                logger.debug(f"OCR识别完成: 识别到{len(ocr_results)}处文字, 耗时={ocr_elapsed:.3f}秒")
            except Exception as e:
                logger.error(f"OCR识别过程出错: {e}", exc_info=True)
                ocr_results = []
                ocr_elapsed = time.time() - ocr_start
                logger.warning(f"OCR识别失败，返回空结果，耗时={ocr_elapsed:.3f}秒")
        else:
            # 如果image_path存在，尝试从文件路径OCR（兼容旧逻辑）
            if image_path:
                try:
                    result = self._run_paddle_ocr(image_path)
                    ocr_results = self._parse_ocr_result(result)
                except Exception as e:
                    logger.error(f"从文件路径OCR失败: {e}", exc_info=True)
                    ocr_results = []
            else:
                ocr_results = []
        
        if return_debug:
            debug_info['result_count'] = len(ocr_results)
            debug_info['texts'] = ocr_results
            logger.debug(f"识别完成（返回调试信息）: 结果数量={len(ocr_results)}, "
                        f"预处理={debug_info['used_preprocessing']}, 增强={debug_info['enhancement_applied']}, "
                        f"来源={debug_info['ocr_source']}")
            return ocr_results, debug_info
        
        logger.debug(f"识别完成: 结果数量={len(ocr_results)}, 预处理={debug_info['used_preprocessing']}, "
                    f"增强={debug_info['enhancement_applied']}, 来源={debug_info['ocr_source']}")
        return ocr_results
    
    def recognize(self, image_path: str, use_preprocessing: bool = False, return_debug: bool = False):
        """
        识别图片中的文字
        
        Args:
            image_path: 图片路径
            use_preprocessing: 是否使用预处理（提取红色区域）
            return_debug: 是否返回调试信息
            
        Returns:
            OCR识别结果列表，每个元素包含：
            - text: 识别的文字
            - confidence: 置信度
            - bbox: 边界框坐标
        """
        logger.debug(f"开始识别图片: path={image_path}, use_preprocessing={use_preprocessing}, return_debug={return_debug}")
        # 读取原图（用于质量检测和增强）
        original_image = self._read_image_with_chinese_path(image_path)
        if original_image is None:
            logger.warning(f"无法读取图片: {image_path}")
            return ([], {}) if return_debug else []
        
        h, w = original_image.shape[:2]
        logger.debug(f"图片读取成功: 尺寸={w}x{h}")
        
        # 调用核心识别逻辑
        return self._recognize_core(original_image, use_preprocessing, image_path, return_debug)
    
    def recognize_from_array(self, image: np.ndarray, use_preprocessing: bool = False, return_debug: bool = False):
        """
        从numpy数组识别图片中的文字（无需文件路径，适合流式处理）
        
        Args:
            image: BGR格式的图像数组（numpy.ndarray）
            use_preprocessing: 是否使用预处理（提取红色区域）
            return_debug: 是否返回调试信息
            
        Returns:
            OCR识别结果列表，每个元素包含：
            - text: 识别的文字
            - confidence: 置信度
            - bbox: 边界框坐标
        """
        if image is None:
            logger.warning("recognize_from_array: 输入图片为None")
            return ([], {}) if return_debug else []
        
        h, w = image.shape[:2]
        logger.debug(f"开始从数组识别: 尺寸={w}x{h}, use_preprocessing={use_preprocessing}, return_debug={return_debug}")
        
        # 调用核心识别逻辑（不传image_path，表示从数组识别）
        return self._recognize_core(image, use_preprocessing, image_path=None, return_debug=return_debug)
    
    def get_all_text(self, image_path: str, use_preprocessing: bool = False) -> str:
        """
        获取图片中所有识别到的文字（合并为字符串，仅OCR结果）
        
        Args:
            image_path: 图片路径
            use_preprocessing: 是否使用预处理
            
        Returns:
            所有识别文字的合并字符串
        """
        ocr_results = self.recognize(image_path, use_preprocessing=use_preprocessing)
        texts = [info.get('text', '') for info in ocr_results]
        return ''.join(texts)
    
    def get_texts_with_confidence(self, image_path: str, use_preprocessing: bool = False) -> List[Tuple[str, float]]:
        """
        获取文字及其置信度（仅OCR结果）
        
        Args:
            image_path: 图片路径
            use_preprocessing: 是否使用预处理
            
        Returns:
            (文字, 置信度) 元组列表
        """
        ocr_results = self.recognize(image_path, use_preprocessing=use_preprocessing)
        return [
            (info.get('text', ''), info.get('confidence', 0.0))
            for info in ocr_results
        ]
    
    def _perform_ocr_on_image(self, image: np.ndarray) -> List[Dict]:
        if image is None:
            logger.debug("OCR输入图片为None，返回空结果")
            return []
        scale_x = 1.0
        scale_y = 1.0
        prepared = image
        if self.optimization_enabled and image is not None:
            max_dim = max(image.shape[:2])
            if max_dim > self.max_size:
                original_h, original_w = image.shape[:2]
                prepared = self._optimize_image_size(image)
                new_h, new_w = prepared.shape[:2]
                scale_x = original_w / new_w if new_w else 1.0
                scale_y = original_h / new_h if new_h else 1.0
                logger.debug(f"OCR前图片缩放: {original_w}x{original_h} -> {new_w}x{new_h}, scale=({scale_x:.3f}, {scale_y:.3f})")
        
        # 记录OCR前的图片信息
        h, w = prepared.shape[:2]
        logger.debug(f"执行PaddleOCR: 输入尺寸={w}x{h}, 性能参数: det_db_thresh={self.det_db_thresh}, "
                    f"det_db_box_thresh={self.det_db_box_thresh}, text_recognition_batch_size={self.text_recognition_batch_size}")
        
        try:
            result = self._run_paddle_ocr(prepared)
        except Exception as e:
            # 异常已在_run_paddle_ocr中详细打印，这里只记录简要信息
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(f"PaddleOCR执行失败: {error_type}: {error_msg}")
            # 返回空结果，而不是抛出异常
            return []
        
        try:
            parsed_results = self._parse_ocr_result(result, scale_x=scale_x, scale_y=scale_y)
            logger.debug(f"OCR结果解析: 原始结果数量={len(result) if result else 0}, 解析后数量={len(parsed_results)}")
            return parsed_results
        except Exception as e:
            logger.error(f"OCR结果解析失败: {e}", exc_info=True)
            # 返回空结果，而不是抛出异常
            return []
    
    def _run_paddle_ocr(self, image_input):
        """
        调用PaddleOCR进行识别
        
        根据PaddleOCR源码（ocr.py第198-212行），predict()方法支持的参数：
        - text_det_thresh: 检测阈值（旧名称det_db_thresh已弃用）
        - text_det_box_thresh: 检测框阈值（旧名称det_db_box_thresh已弃用）
        - text_det_unclip_ratio: 检测框扩展比例（旧名称det_db_unclip_ratio已弃用）
        - text_det_limit_side_len: 检测输入边长限制（可在调用时覆盖初始化设置，重要性能参数）
        - text_rec_score_thresh: 识别分数阈值
        - return_word_box: 是否返回词框
        
        注意：text_recognition_batch_size、use_textline_orientation等参数只能在初始化时设置
        """
        try:
            # 构建参数字典，使用新版本的参数名（根据源码第207-209行）
            ocr_kwargs = {}
            
            # 检测相关参数（使用新参数名）
            if self.det_db_thresh is not None:
                ocr_kwargs['text_det_thresh'] = self.det_db_thresh
            if self.det_db_box_thresh is not None:
                ocr_kwargs['text_det_box_thresh'] = self.det_db_box_thresh
            if self.det_db_unclip_ratio is not None:
                ocr_kwargs['text_det_unclip_ratio'] = self.det_db_unclip_ratio
            # 检测边长限制（如果配置了，可以在调用时覆盖，这是重要的性能参数）
            if self.text_det_limit_side_len is not None:
                ocr_kwargs['text_det_limit_side_len'] = self.text_det_limit_side_len
            
            result = self.ocr.ocr(image_input, **ocr_kwargs)
            return result
        except Exception as e:
            # 捕获并打印详细异常信息
            error_type = type(e).__name__
            error_msg = str(e)
            import traceback
            error_traceback = traceback.format_exc()
            
            logger.error(f"PaddleOCR调用失败 - 异常类型: {error_type}, 异常信息: {error_msg}")
            logger.error(f"异常堆栈:\n{error_traceback}")
            
            # 如果是RuntimeError，可能是OCR实例假死，尝试重新初始化
            if error_type == 'RuntimeError':
                logger.warning("检测到RuntimeError，OCR实例可能已假死，尝试重新初始化...")
                try:
                    self._reinitialize_ocr()
                    logger.info("OCR实例重新初始化成功，将重试OCR调用")
                    # 重试一次OCR调用
                    result = self.ocr.ocr(image_input, **ocr_kwargs)
                    return result
                except Exception as reinit_error:
                    logger.error(f"OCR实例重新初始化失败: {reinit_error}", exc_info=True)
                    raise Exception(f"OCR调用失败且重新初始化失败: {error_msg}")
            
            raise
    
    def _parse_ocr_result(self, result, scale_x: float = 1.0, scale_y: float = 1.0) -> List[Dict]:
        ocr_results = []
        if result is None:
            # PaddleOCR返回None，可能是图片无效或OCR调用失败
            return ocr_results
        if isinstance(result, list) and len(result) == 0:
            # PaddleOCR返回空列表，表示未检测到任何文字区域
            return ocr_results
        if isinstance(result, list) and len(result) > 0 and result[0] is None:
            # 第一层是列表但第一个元素是None，可能是格式异常
            return ocr_results
        
        if isinstance(result, list):
            for item in result:
                if item is None:
                    continue
                try:
                    if isinstance(item, dict) and 'rec_texts' in item and 'rec_scores' in item:
                        texts = item['rec_texts']
                        scores = item['rec_scores']
                        bboxes = item.get('rec_polys', item.get('dt_polys', []))
                        if isinstance(texts, list) and isinstance(scores, list):
                            for i, text in enumerate(texts):
                                if text and str(text).strip():
                                    score = scores[i] if i < len(scores) else 1.0
                                    bbox = bboxes[i] if i < len(bboxes) else []
                                    bbox = self._scale_bbox(bbox, scale_x, scale_y)
                                    ocr_results.append({
                                        'text': str(text).strip(),
                                        'confidence': float(score) if score else 0.0,
                                        'bbox': bbox,
                                        'source': 'ocr'
                                    })
                        continue
                    
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        bbox = item[0]
                        text_info = item[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text, confidence = text_info[0], text_info[1]
                        elif isinstance(text_info, (list, tuple)) and len(text_info) == 1:
                            text, confidence = text_info[0], 1.0
                        elif isinstance(text_info, str):
                            text, confidence = text_info, 1.0
                        else:
                            continue
                        if text and str(text).strip():
                            bbox = self._scale_bbox(bbox, scale_x, scale_y)
                            ocr_results.append({
                                'text': str(text).strip(),
                                'confidence': float(confidence) if confidence else 0.0,
                                'bbox': bbox,
                                'source': 'ocr'
                            })
                    
                    elif isinstance(item, dict):
                        text = item.get('text', '') or item.get('txt', '')
                        confidence = item.get('confidence', 0.0) or item.get('score', 0.0)
                        bbox = item.get('bbox', []) or item.get('box', [])
                        if text and str(text).strip():
                            bbox = self._scale_bbox(bbox, scale_x, scale_y)
                            ocr_results.append({
                                'text': str(text).strip(),
                                'confidence': float(confidence) if confidence else 0.0,
                                'bbox': bbox,
                                'source': 'ocr'
                            })
                except Exception:
                    continue
        return ocr_results
    
    def _scale_bbox(self, bbox, scale_x: float, scale_y: float):
        if scale_x == 1.0 and scale_y == 1.0:
            return bbox
        if isinstance(bbox, list) and len(bbox) > 0:
            if isinstance(bbox[0], (list, tuple, np.ndarray)):
                scaled_bbox = []
                for point in bbox:
                    if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                        scaled_bbox.append([int(point[0] * scale_x), int(point[1] * scale_y)])
                    else:
                        scaled_bbox.append(point)
                return scaled_bbox
            elif len(bbox) >= 4:
                return [
                    int(bbox[0] * scale_x),
                    int(bbox[1] * scale_y),
                    int(bbox[2] * scale_x),
                    int(bbox[3] * scale_y),
                ]
        return bbox
    
    def _save_red_full_image(self, image_path: str, image: np.ndarray):
        if image is None or not self.debug_enabled or not self.debug_save_red_full:
            return
        filename = Path(image_path).stem
        path = self.debug_output_dir / f"{filename}_red_full.jpg"
        try:
            cv2.imwrite(str(path), image)
        except Exception:
            pass
