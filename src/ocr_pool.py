"""
OCR引擎实例池管理模块
提供线程安全的OCR实例池，支持并发访问和资源管理
"""
from typing import Dict
import threading
from queue import Queue, Empty
from contextlib import contextmanager
from loguru import logger

from preprocess import RedRegionExtractor
from ocr_engine import OCREngine


def create_ocr_engine_from_config(ocr_config: Dict, debug_config: Dict = None) -> OCREngine:
    """
    根据配置创建OCR引擎实例（工具函数）
    
    Args:
        ocr_config: OCR引擎初始化配置字典
        debug_config: 调试配置字典（可选）
        
    Returns:
        OCREngine: OCR引擎实例
    """
    # 检查是否启用预处理
    preprocessing_enabled = ocr_config.get('preprocessing', {}).get('enabled', False)
    
    # 初始化预处理器（如果需要）
    preprocessor = None
    if preprocessing_enabled:
        preprocessor = RedRegionExtractor(ocr_config)
    
    # 获取图片优化配置
    optimization_config = ocr_config.get('image_optimization', {})
    
    # 获取OCR性能优化配置
    ocr_performance_config = ocr_config.get('ocr_performance', {})
    
    # 获取图片增强配置
    enhancement_config = ocr_config.get('image_enhancement', {})
    
    # 初始化OCR引擎
    ocr_engine = OCREngine(
        lang='ch',
        preprocessor=preprocessor,
        optimization_config=optimization_config,
        ocr_performance_config=ocr_performance_config,
        enhancement_config=enhancement_config,
        debug_config=debug_config
    )
    
    return ocr_engine


def reinitialize_ocr_engine(ocr_engine: OCREngine):
    """
    重新初始化OCR引擎实例（用于处理假死情况）
    统一管理重新初始化逻辑
    
    Args:
        ocr_engine: 需要重新初始化的 OCREngine 实例
        
    Note:
        - 只重新初始化内部的 PaddleOCR 实例（self.ocr）
        - OCREngine 对象本身不变，可以正常放回实例池
        - 保留所有其他配置（预处理器、优化配置等）
        - 只处理传入的单个实例，不影响池中的其他实例
    """
    logger.warning("开始重新初始化OCR实例...")
    
    # 构建完整的配置字典（与初始化时使用的格式相同）
    ocr_config = {
        'preprocessing': {
            'enabled': ocr_engine.preprocessor is not None
        },
        'image_optimization': ocr_engine.optimization_config if ocr_engine.optimization_config else {},
        'ocr_performance': ocr_engine.ocr_performance_config if ocr_engine.ocr_performance_config else {},
        'image_enhancement': ocr_engine.enhancement_config if ocr_engine.enhancement_config else {}
    }
    
    # 先尝试释放旧的OCR实例
    try:
        del ocr_engine.ocr
    except:
        pass
    
    # 使用统一方法重新创建OCR引擎（只重新初始化PaddleOCR部分）
    # 注意：这里只重新初始化 ocr_engine.ocr，其他配置保持不变
    # OCREngine 对象本身不变，所以会正常放回实例池
    new_engine = create_ocr_engine_from_config(ocr_config, ocr_engine.debug_config)
    
    # 只替换OCR实例，保留其他配置
    # 重要：OCREngine 对象本身（ocr_engine）没有改变，只是内部的 ocr_engine.ocr 被替换
    # 因此这个实例仍然可以正常放回实例池
    ocr_engine.ocr = new_engine.ocr
    
    logger.warning("OCR实例重新初始化成功")


class OCREnginePool:
    """
    OCR引擎实例池管理器
    线程安全的资源池，支持并发访问
    """
    
    def __init__(self, pool_size: int, ocr_config: Dict):
        """
        初始化OCR实例池
        
        Args:
            pool_size: 实例池大小
            ocr_config: OCR引擎初始化配置
        """
        self.pool_size = pool_size
        self.ocr_config = ocr_config
        self.pool = Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        self.created_count = 0
        
        # 初始化所有OCR实例
        logger.info(f"开始初始化OCR实例池，池大小: {pool_size}")
        for i in range(pool_size):
            try:
                ocr_engine = self._create_ocr_engine()
                self.pool.put(ocr_engine)
                self.created_count += 1
                logger.info(f"OCR实例 {i+1}/{pool_size} 初始化完成")
            except Exception as e:
                logger.error(f"OCR实例 {i+1}/{pool_size} 初始化失败: {str(e)}", exc_info=True)
                raise
        
        logger.info(f"OCR实例池初始化完成，共 {self.created_count} 个实例")
    
    def _create_ocr_engine(self) -> OCREngine:
        """创建单个OCR引擎实例"""
        # 使用封装的工具函数
        return create_ocr_engine_from_config(self.ocr_config)
    
    @contextmanager
    def acquire(self, timeout: float = None):
        """
        获取OCR实例（上下文管理器，自动释放）
        
        Args:
            timeout: 超时时间（秒），None表示不超时，0表示立即返回（不等待）
            
        Yields:
            OCREngine: OCR引擎实例
            
        Raises:
            TimeoutError: 如果超时时间内无法获取实例
        """
        ocr_engine = None
        try:
            # 尝试从池中获取实例
            if timeout is None:
                ocr_engine = self.pool.get()
            elif timeout == 0:
                # 立即返回，不等待
                ocr_engine = self.pool.get_nowait()
            else:
                ocr_engine = self.pool.get(timeout=timeout)
            
            yield ocr_engine
            
        except Empty:
            raise TimeoutError(f"无法获取OCR实例，池中无可用资源（超时: {timeout}秒）")
        finally:
            # 确保释放实例回池
            # 注意：即使实例在使用过程中被重新初始化（_reinitialize_ocr），
            # OCREngine 对象本身没有改变，仍然可以正常放回池中
            if ocr_engine is not None:
                try:
                    self.pool.put_nowait(ocr_engine)
                except:
                    # 如果池已满（理论上不应该发生），记录错误
                    logger.error("OCR实例池已满，无法归还实例", exc_info=True)
    
    def get_available_count(self) -> int:
        """获取当前可用的实例数量"""
        return self.pool.qsize()
    
    def get_total_count(self) -> int:
        """获取实例池总大小"""
        return self.pool_size

