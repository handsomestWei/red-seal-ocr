"""
图像预处理模块
提取红色区域，用于提高印章识别准确率
支持图片质量检测和增强
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from PIL import Image
import os


class ImageEnhancer:
    """图片增强器，用于提升小质量图片的清晰度"""
    
    def __init__(self, config: Dict = None):
        """
        初始化图片增强器
        
        Args:
            config: 配置字典，包含增强参数
        """
        if config is None:
            config = {}
        
        enhance_config = config.get('image_enhancement', {})
        self.denoise_strength = enhance_config.get('denoise_strength', 3)  # 去噪强度
        self.sharpen_strength = enhance_config.get('sharpen_strength', 0.5)  # 锐化强度
        self.contrast_alpha = enhance_config.get('contrast_alpha', 1.1)  # 对比度增强系数
        self.brightness_beta = enhance_config.get('brightness_beta', 0)  # 亮度调整
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        增强图片质量
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            增强后的图像
        """
        enhanced = image.copy()
        
        # 1. 去噪（使用非局部均值去噪，对JPEG压缩噪声有效）
        # 注意：使用彩色去噪以保持颜色信息，这对后续的红色区域提取很重要
        if self.denoise_strength > 0:
            try:
                # 使用彩色去噪（虽然慢一些，但能保持颜色信息）
                # 如果图片太大，可以降级到灰度去噪
                h, w = enhanced.shape[:2]
                if h * w > 2000000:  # 如果图片很大（>200万像素），使用灰度去噪以节省时间
                    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                    denoised_gray = cv2.fastNlMeansDenoising(
                        gray, 
                        h=self.denoise_strength,
                        templateWindowSize=7,
                        searchWindowSize=21
                    )
                    denoised = cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2BGR)
                    # 混合原图和去噪图（避免过度去噪导致细节丢失）
                    enhanced = cv2.addWeighted(enhanced, 0.7, denoised, 0.3, 0)
                else:
                    # 使用彩色去噪，保持颜色信息
                    denoised = cv2.fastNlMeansDenoisingColored(
                        enhanced,
                        h=self.denoise_strength,
                        hColor=self.denoise_strength,
                        templateWindowSize=7,
                        searchWindowSize=21
                    )
                    # 混合原图和去噪图（避免过度去噪导致细节丢失）
                    enhanced = cv2.addWeighted(enhanced, 0.7, denoised, 0.3, 0)
            except:
                # 如果去噪失败，跳过
                pass
        
        # 2. 对比度和亮度调整（温和调整，避免大幅改变颜色）
        if self.contrast_alpha != 1.0 or self.brightness_beta != 0:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=self.contrast_alpha, beta=self.brightness_beta)
        
        # 3. 锐化（使用Unsharp Mask，增强文字边缘）
        # 注意：锐化不会改变颜色，只增强边缘，对HSV提取无影响
        if self.sharpen_strength > 0:
            # 创建高斯模糊
            blurred = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
            # Unsharp Mask: 原图 + (原图 - 模糊图) * 强度
            sharpened = cv2.addWeighted(enhanced, 1.0 + self.sharpen_strength, blurred, -self.sharpen_strength, 0)
            # 混合原图和锐化图（避免过度锐化）
            enhanced = cv2.addWeighted(enhanced, 0.6, sharpened, 0.4, 0)
        
        return enhanced
    
    @staticmethod
    def is_low_quality_image(image_path: str, image: np.ndarray = None, 
                             size_threshold_kb: float = 200, 
                             resolution_threshold: int = 1200) -> bool:
        """
        判断图片是否为低质量图片
        
        Args:
            image_path: 图片路径
            image: 图像数组（如果已加载）
            size_threshold_kb: 文件大小阈值（KB），小于此值认为是低质量
            resolution_threshold: 分辨率阈值（px），小于此值认为是低质量
            
        Returns:
            True表示低质量图片，False表示高质量图片
        """
        # 检查文件大小
        file_size_kb = os.path.getsize(image_path) / 1024
        if file_size_kb < size_threshold_kb:
            return True
        
        # 检查分辨率
        if image is None:
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    max_dimension = max(width, height)
            except:
                return False
        else:
            height, width = image.shape[:2]
            max_dimension = max(width, height)
        
        if max_dimension < resolution_threshold:
            return True
        
        return False


class RedRegionExtractor:
    """红色区域提取器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化红色区域提取器
        
        Args:
            config: 配置字典，包含HSV范围和形态学参数
        """
        if config is None:
            config = {}
        
        # 保存config用于后续使用
        self.config = config
        
        # HSV颜色范围（红色在色相环的两端）
        red_config = config.get('red_extraction', {})
        self.hsv_lower1 = np.array(red_config.get('hsv_lower', [0, 20, 20]))
        self.hsv_upper1 = np.array(red_config.get('hsv_upper', [20, 255, 255]))
        self.hsv_lower2 = np.array(red_config.get('hsv_lower2', [160, 20, 20]))
        self.hsv_upper2 = np.array(red_config.get('hsv_upper2', [180, 255, 255]))
        
        # 形态学操作参数
        morph_config = config.get('morphology', {})
        self.kernel_size = morph_config.get('kernel_size', 3)
        self.iterations = morph_config.get('iterations', 2)
        
        # 图片增强器
        self.enhancer = ImageEnhancer(config)
        
        # 红色增强配置
        red_enhance_config = config.get('red_enhancement', {})
        self.red_enhance_enabled = red_enhance_config.get('enabled', True)
        self.red_saturation_boost = red_enhance_config.get('red_saturation_boost', 1.3)  # 红色饱和度增强
        self.red_brightness_boost = red_enhance_config.get('red_brightness_boost', 1.1)  # 红色亮度增强
        self.value_gamma = red_enhance_config.get('value_gamma', 0.95)  # 亮度伽马调整（<1提亮）
        self.clahe_clip_limit = red_enhance_config.get('clahe_clip_limit', 2.5)  # CLAHE对比度
        tile_grid = red_enhance_config.get('clahe_tile_grid', [8, 8])
        if isinstance(tile_grid, (list, tuple)) and len(tile_grid) == 2:
            self.clahe_tile_grid = (int(tile_grid[0]), int(tile_grid[1]))
        else:
            self.clahe_tile_grid = (8, 8)
        self.black_threshold = red_enhance_config.get('black_threshold', 60)  # 黑色阈值（低于此值认为是黑色）
        self.black_saturation_threshold = red_enhance_config.get('black_saturation_threshold', 70)  # 黑色判断时允许的饱和度
        self.preserve_dark_red = red_enhance_config.get('preserve_dark_red', True)
        self.remove_black = red_enhance_config.get('remove_black', True)  # 是否去除黑色
        self.remove_gray = red_enhance_config.get('remove_gray', True)  # 是否去除灰色（表格线）
        self.gray_threshold = red_enhance_config.get('gray_threshold', 0.3)  # 灰色判断阈值（饱和度低于此值）
    
    def _enhance_red_and_remove_black(self, image: np.ndarray, red_mask: np.ndarray, config: Dict = None) -> np.ndarray:
        """
        在红色区域内增强红色，并去除黑色文字和灰色表格线
        
        Args:
            image: 输入图像（BGR格式，非红色区域已变白）
            red_mask: 红色区域掩码
            config: 配置字典（可选，如果不提供则使用实例配置）
            
        Returns:
            处理后的图像
        """
        # 默认使用实例配置
        saturation_boost = self.red_saturation_boost
        brightness_boost = self.red_brightness_boost
        value_gamma = self.value_gamma
        clahe_clip_limit = self.clahe_clip_limit
        clahe_tile_grid = self.clahe_tile_grid
        black_threshold = self.black_threshold
        black_saturation_threshold = self.black_saturation_threshold
        preserve_dark_red = self.preserve_dark_red
        remove_black = self.remove_black
        remove_gray = self.remove_gray
        gray_threshold = self.gray_threshold
        
        if config is not None:
            saturation_boost = config.get('red_saturation_boost', saturation_boost)
            brightness_boost = config.get('red_brightness_boost', brightness_boost)
            value_gamma = config.get('value_gamma', value_gamma)
            clahe_clip_limit = config.get('clahe_clip_limit', clahe_clip_limit)
            tile_grid = config.get('clahe_tile_grid', clahe_tile_grid)
            if isinstance(tile_grid, (list, tuple)) and len(tile_grid) == 2:
                clahe_tile_grid = (int(tile_grid[0]), int(tile_grid[1]))
            black_threshold = config.get('black_threshold', black_threshold)
            black_saturation_threshold = config.get('black_saturation_threshold', black_saturation_threshold)
            preserve_dark_red = config.get('preserve_dark_red', preserve_dark_red)
            remove_black = config.get('remove_black', remove_black)
            remove_gray = config.get('remove_gray', remove_gray)
            gray_threshold = config.get('gray_threshold', gray_threshold)
        
        result = image.copy()
        
        # 只在红色区域内处理
        if np.sum(red_mask) == 0:
            return result
        
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        # 创建红色区域内的处理掩码
        red_region_mask = (red_mask > 0)
        
        # 1. 增强红色像素的饱和度和亮度
        if saturation_boost > 1.0 or brightness_boost > 1.0:
            red_h_mask = ((h < 20) | (h > 160)) & red_region_mask
            s[red_h_mask] = np.clip(s[red_h_mask] * saturation_boost, 0, 255)
            v[red_h_mask] = np.clip(v[red_h_mask] * brightness_boost, 0, 255)
        
        # 2. 局部对比和亮度调整，让浅色红章更明显
        if clahe_clip_limit and clahe_clip_limit > 0:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
            v_uint8 = np.clip(v, 0, 255).astype(np.uint8)
            v_clahe = clahe.apply(v_uint8)
            v[red_region_mask] = v_clahe[red_region_mask]
        
        if abs(value_gamma - 1.0) > 0.01:
            v_norm = np.clip(v / 255.0, 0, 1)
            v_gamma = (np.power(v_norm, value_gamma) * 255.0).astype(np.float32)
            v[red_region_mask] = v_gamma[red_region_mask]
        
        # 转换回BGR用于后续处理（去黑/灰等）
        hsv_enhanced = cv2.merge([np.clip(h, 0, 255), np.clip(s, 0, 255), np.clip(v, 0, 255)])
        result = cv2.cvtColor(hsv_enhanced.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # 2. 去除黑色文字和线条
        if remove_black:
            # 黑色判断：亮度（V）很低，同时饱和度也低（避免误删暗红文字）
            black_mask = (v < black_threshold) & red_region_mask
            if preserve_dark_red:
                black_mask = black_mask & (s < black_saturation_threshold)
            # 将黑色像素设为白色
            result[black_mask] = [255, 255, 255]
        
        # 3. 去除灰色表格线（低饱和度）
        if remove_gray:
            # 灰色判断：饱和度很低，且不是红色
            gray_mask = (s < gray_threshold * 255) & red_region_mask
            # 排除红色像素（红色即使饱和度低也要保留）
            h_normalized = h / 180.0  # 归一化到0-1
            not_red_mask = ~((h_normalized < 20/180.0) | (h_normalized > 160/180.0))
            gray_mask = gray_mask & not_red_mask
            # 将灰色像素设为白色
            result[gray_mask] = [255, 255, 255]
        
        return result
    
    def _apply_additional_red_enhancement(self, processed_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        应用额外的红色区域增强（LAB颜色空间、HSV增强、锐化）
        这些增强步骤特别针对小文字印章，提升对比度和清晰度
        
        Args:
            processed_image: 已经过基础增强的图像
            mask: 红色区域掩码
            
        Returns:
            进一步增强后的图像
        """
        if np.sum(mask) == 0:
            return processed_image
        
        # 提取红色区域
        red_region = cv2.bitwise_and(processed_image, processed_image, mask=mask)
        
        # 方法1：使用LAB颜色空间进行对比度增强
        lab = cv2.cvtColor(red_region, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # 使用更强的CLAHE参数，提升小文字的清晰度
        # clipLimit增加到2.0，tileGridSize减小到8x8，使增强更局部化，更适合小文字
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        # 合并通道并转换回BGR
        enhanced_red = cv2.merge([l, a, b])
        enhanced_red = cv2.cvtColor(enhanced_red, cv2.COLOR_LAB2BGR)
        
        # 方法2：额外的对比度和亮度调整，特别针对模糊的红色印章
        # 转换为HSV，增强饱和度和明度
        hsv_red = cv2.cvtColor(enhanced_red, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_red)
        # 增强饱和度（使红色更鲜艳）
        s = cv2.multiply(s, 1.2)
        s = np.clip(s, 0, 255).astype(np.uint8)
        # 增强明度（使文字更清晰）
        v = cv2.multiply(v, 1.1)
        v = np.clip(v, 0, 255).astype(np.uint8)
        enhanced_red = cv2.merge([h, s, v])
        enhanced_red = cv2.cvtColor(enhanced_red, cv2.COLOR_HSV2BGR)
        
        # 方法3：轻微锐化，提升文字边缘清晰度
        kernel_sharpen = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]]) * 0.3  # 降低锐化强度，避免过度
        sharpened = cv2.filter2D(enhanced_red, -1, kernel_sharpen)
        # 混合原图和锐化图（70%原图 + 30%锐化图）
        enhanced_red = cv2.addWeighted(enhanced_red, 0.7, sharpened, 0.3, 0)
        
        # 将增强后的红色区域放回原图
        processed_image[mask > 0] = enhanced_red[mask > 0]
        
        return processed_image
    
    def _read_image_with_chinese_path(self, image_path: str) -> np.ndarray:
        """
        读取图片（支持中文路径）
        
        Args:
            image_path: 图片路径
            
        Returns:
            图像数组（BGR格式）
        """
        try:
            # 使用PIL读取图片（支持中文路径）
            with Image.open(image_path) as img:
                # 转换为RGB格式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # 转换为numpy数组
                img_array = np.array(img)
                # PIL读取的是RGB，需要转换为BGR（OpenCV格式）
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                return img_bgr
        except Exception:
            # 如果PIL读取失败，尝试使用OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            return image
    
    def extract_red_regions(self, image_path: str, return_debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取图像中的红色区域
        
        Args:
            image_path: 图片路径
            
        Returns:
            (mask, processed_image) 元组
            - mask: 红色区域的掩码（二值图像）
            - processed_image: 处理后的图像（红色区域保留，其他区域变白）
        """
        # 读取图像（支持中文路径）
        image = self._read_image_with_chinese_path(image_path)
        
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 创建红色掩码（红色在色相环的两端：0-15 和 165-180，扩大范围以识别更多红色）
        mask1 = cv2.inRange(hsv, self.hsv_lower1, self.hsv_upper1)
        mask2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
        initial_mask = cv2.bitwise_or(mask1, mask2)
        mask = initial_mask.copy()
        
        # 最小化形态学操作，保持文字完整性
        # 只做非常轻微的去噪，避免破坏文字结构
        kernel_small = np.ones((2, 2), np.uint8)
        # 只去除极小的噪声点，不进行闭运算（避免连接独立的文字）
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # 查找轮廓，保留所有红色区域（不过度过滤）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 根据图片大小动态调整最小面积要求（非常宽松，确保小文字不被过滤）
        img_size = max(image.shape[:2])
        if img_size < 1000:
            min_area_ratio = 0.00002  # 0.002%，小图片非常宽松
        elif img_size < 2000:
            min_area_ratio = 0.00005  # 0.005%，中等图片
        else:
            min_area_ratio = 0.0001   # 0.01%，大图片（非常宽松以保留所有文字）
        
        min_area = (image.shape[0] * image.shape[1]) * min_area_ratio
        filtered_mask = np.zeros_like(mask)
        contour_infos = []
        kept_contours = 0
        
        # 保留所有大于最小面积的区域，不过度过滤
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            keep = area >= min_area
            if keep:
                kept_contours += 1
                # 直接保留，不过度检查矩形度（文字可能不是完美矩形）
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
            contour_infos.append({
                "area": float(area),
                "bbox": [int(x), int(y), int(w), int(h)],
                "aspect_ratio": float(aspect_ratio),
                "keep": bool(keep)
            })
        
        mask = filtered_mask
        
        # 如果过滤后没有红色区域，使用原始掩码（不做任何处理）
        if np.sum(mask) == 0:
            mask = initial_mask
        
        # 创建处理后的图像：红色区域保留，其他区域变白
        processed_image = image.copy()
        # 将非红色区域变为白色
        processed_image[mask == 0] = [255, 255, 255]
        
        # 在红色区域内，增强红色并去除黑色干扰
        red_enhance_config = self.config.get('red_enhancement', {})
        if red_enhance_config.get('enabled', True):
            processed_image = self._enhance_red_and_remove_black(processed_image, mask, red_enhance_config)
        
        # 增强红色区域的对比度和清晰度，特别针对小文字印章
        if np.sum(mask) > 0:
            processed_image = self._apply_additional_red_enhancement(processed_image, mask)
        
        if return_debug:
            total_pixels = image.shape[0] * image.shape[1]
            raw_pixels = int(np.sum(initial_mask > 0))
            raw_ratio = raw_pixels / total_pixels if total_pixels else 0
            filtered_pixels = int(np.sum(mask > 0))
            filtered_ratio = filtered_pixels / total_pixels if total_pixels else 0
            debug_info = {
                "image_shape": image.shape,
                "file_size_kb": os.path.getsize(image_path) / 1024 if os.path.exists(image_path) else None,
                "hsv_range1": (self.hsv_lower1.tolist(), self.hsv_upper1.tolist()),
                "hsv_range2": (self.hsv_lower2.tolist(), self.hsv_upper2.tolist()),
                "raw_red_pixels": raw_pixels,
                "raw_red_ratio": raw_ratio,
                "filtered_red_pixels": filtered_pixels,
                "filtered_red_ratio": filtered_ratio,
                "total_contours": len(contours),
                "kept_contours": kept_contours,
                "min_area": float(min_area),
                "min_area_ratio": min_area_ratio,
                "contours": contour_infos
            }
            return mask, processed_image, debug_info
        
        return mask, processed_image
    
    def extract_red_regions_from_array(self, image: np.ndarray, return_debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        从numpy数组提取红色区域
        
        Args:
            image: BGR格式的图像数组
            
        Returns:
            (mask, processed_image) 元组
        """
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 创建红色掩码
        mask1 = cv2.inRange(hsv, self.hsv_lower1, self.hsv_upper1)
        mask2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
        initial_mask = cv2.bitwise_or(mask1, mask2)
        mask = initial_mask.copy()
        
        # 最小化形态学操作，保持文字完整性
        kernel_small = np.ones((2, 2), np.uint8)
        # 只去除极小的噪声点，不进行闭运算
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # 查找轮廓，保留所有红色区域（不过度过滤）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 根据图片大小动态调整最小面积要求（非常宽松）
        img_size = max(image.shape[:2])
        if img_size < 1000:
            min_area_ratio = 0.00002
        elif img_size < 2000:
            min_area_ratio = 0.00005
        else:
            min_area_ratio = 0.0001
        
        min_area = (image.shape[0] * image.shape[1]) * min_area_ratio
        filtered_mask = np.zeros_like(mask)
        contour_infos = []
        kept_contours = 0
        
        # 保留所有大于最小面积的区域
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            keep = area >= min_area
            if keep:
                kept_contours += 1
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
            contour_infos.append({
                "area": float(area),
                "bbox": [int(x), int(y), int(w), int(h)],
                "aspect_ratio": float(aspect_ratio),
                "keep": bool(keep)
            })
        
        mask = filtered_mask
        
        # 如果过滤后没有红色区域，使用原始掩码
        if np.sum(mask) == 0:
            mask = initial_mask
        
        # 创建处理后的图像
        processed_image = image.copy()
        processed_image[mask == 0] = [255, 255, 255]
        
        # 在红色区域内，增强红色并去除黑色干扰
        red_enhance_config = self.config.get('red_enhancement', {})
        if red_enhance_config.get('enabled', True):
            processed_image = self._enhance_red_and_remove_black(processed_image, mask, red_enhance_config)
        
        # 应用额外的红色区域增强（LAB、HSV、锐化）
        if np.sum(mask) > 0:
            processed_image = self._apply_additional_red_enhancement(processed_image, mask)
        
        if return_debug:
            total_pixels = image.shape[0] * image.shape[1]
            raw_pixels = int(np.sum(initial_mask > 0))
            raw_ratio = raw_pixels / total_pixels if total_pixels else 0
            filtered_pixels = int(np.sum(mask > 0))
            filtered_ratio = filtered_pixels / total_pixels if total_pixels else 0
            debug_info = {
                "image_shape": image.shape,
                "file_size_kb": None,
                "hsv_range1": (self.hsv_lower1.tolist(), self.hsv_upper1.tolist()),
                "hsv_range2": (self.hsv_lower2.tolist(), self.hsv_upper2.tolist()),
                "raw_red_pixels": raw_pixels,
                "raw_red_ratio": raw_ratio,
                "filtered_red_pixels": filtered_pixels,
                "filtered_red_ratio": filtered_ratio,
                "total_contours": len(contours),
                "kept_contours": kept_contours,
                "min_area": float(min_area),
                "min_area_ratio": min_area_ratio,
                "contours": contour_infos
            }
            return mask, processed_image, debug_info
        
        return mask, processed_image
    
    def save_debug_image(self, image_path: str, output_path: str = None):
        """
        保存调试图像（用于查看预处理效果）
        
        Args:
            image_path: 原始图片路径
            output_path: 输出路径，如果为None则自动生成
        """
        mask, processed_image = self.extract_red_regions(image_path)
        
        if output_path is None:
            original_path = Path(image_path)
            output_path = original_path.parent / f"{original_path.stem}_red_extracted{original_path.suffix}"
        
        cv2.imwrite(str(output_path), processed_image)
        print(f"调试图像已保存: {output_path}")
        
        return str(output_path)


def load_config(config_path: str = None) -> Dict:
    """
    加载配置文件（支持YAML和JSON格式）
    
    Args:
        config_path: 配置文件路径，如果为None则自动查找config.yaml或config.json
    
    Returns:
        配置字典
    """
    import json
    from pathlib import Path
    
    # 如果未指定路径，自动查找config.yaml或config.json
    if config_path is None:
        config_path = "config.yaml"
        config_file = Path(config_path)
        if not config_file.exists():
            config_path = "config.json"
            config_file = Path(config_path)
    else:
        config_file = Path(config_path)
    
    if not config_file.exists():
        # 如果配置文件不存在，返回默认配置
        return {
            "preprocessing": {"enabled": False},
            "red_extraction": {
                "hsv_lower": [0, 50, 50],
                "hsv_upper": [10, 255, 255],
                "hsv_lower2": [170, 50, 50],
                "hsv_upper2": [180, 255, 255]
            },
            "morphology": {
                "kernel_size": 3,
                "iterations": 2
            }
        }
    
    # 根据文件扩展名选择解析方式
    if config_file.suffix.lower() in ['.yaml', '.yml']:
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except ImportError:
            raise ImportError("需要安装PyYAML库才能读取YAML配置文件：pip install pyyaml")
        except Exception as e:
            raise ValueError(f"无法解析YAML配置文件 {config_path}: {e}")
    else:
        # 默认使用JSON格式
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    return config

