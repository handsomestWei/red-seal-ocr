"""
图片下载工具模块
从URL下载图片到内存（流式传递，不落盘）
支持多种图片格式，自动处理内存管理和资源释放
"""
import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional
from loguru import logger


def download_image_to_array(
    url: str, 
    connect_timeout: int = 10, 
    read_timeout: int = 30, 
    verify_ssl: bool = False,
    max_size_mb: int = 20
) -> Tuple[np.ndarray, int, float]:
    """
    从URL下载图片到内存（流式传递，不落盘）
    
    Args:
        url: 图片URL（支持minio等）
        connect_timeout: 连接服务器超时时间（秒）
        read_timeout: 读取数据超时时间（秒）
        verify_ssl: 是否验证SSL证书（False表示跳过验证）
        max_size_mb: 最大文件大小限制（MB），默认20MB
        
    Returns:
        tuple: (图像数组（BGR格式，numpy.ndarray）, 文件大小（字节）, 下载耗时（秒）)
        
    Raises:
        Exception: 下载失败时抛出异常
    """
    import time
    
    download_start_time = time.time()
    
    try:
        # 设置请求头，模拟浏览器请求（避免被服务器拒绝）
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive',
        }
        
        # 下载图片到内存（使用流式下载）
        response = requests.get(
            url, 
            headers=headers,
            timeout=(connect_timeout, read_timeout),  # (连接超时, 读取超时)
            stream=True,
            verify=verify_ssl,  # SSL验证（可配置）
            allow_redirects=True  # 允许重定向
        )
        response.raise_for_status()
        
        # 检查Content-Type是否为图片（某些服务器可能不返回正确的Content-Type）
        content_type = response.headers.get('Content-Type', '').lower()
        if content_type and not content_type.startswith('image/'):
            logger.warning(f"Content-Type不是图片类型: {content_type}，但继续尝试解析")
        
        # 读取图片数据到内存（流式读取，避免内存占用过大）
        with BytesIO() as image_data:
            total_size = 0
            max_size = max_size_mb * 1024 * 1024  # 转换为字节
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # 过滤掉keep-alive的空块
                    image_data.write(chunk)
                    total_size += len(chunk)
                    if total_size > max_size:
                        raise ValueError(f"图片文件过大（超过{max_size_mb}MB），已停止下载")
            
            if total_size == 0:
                raise ValueError("下载的图片数据为空")
            image_data.seek(0)
            
            # 使用PIL读取图片（支持多种格式）
            with Image.open(image_data) as img:
                # 转换为RGB格式（如果是RGBA等）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 转换为numpy数组
                img_array = np.array(img)
                # PIL读取的是RGB，需要转换为BGR（OpenCV格式）
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            download_elapsed_time = time.time() - download_start_time
            
            return img_bgr, total_size, download_elapsed_time
            
    except requests.exceptions.Timeout as e:
        error_detail = str(e)
        if 'Read timed out' in error_detail:
            raise Exception(f"下载图片读取超时（已超过{read_timeout}秒），可能是图片文件过大或网络较慢")
        elif 'Connect timeout' in error_detail:
            raise Exception(f"连接服务器超时（已超过{connect_timeout}秒），请检查网络连接或URL是否正确")
        else:
            raise Exception(f"下载图片超时: {error_detail}")
    except requests.exceptions.SSLError as e:
        raise Exception(f"SSL连接错误: {str(e)}")
    except requests.exceptions.ConnectionError as e:
        raise Exception(f"连接错误: {str(e)}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"下载图片失败: {str(e)}")
    except Exception as e:
        raise Exception(f"处理图片失败: {str(e)}")

