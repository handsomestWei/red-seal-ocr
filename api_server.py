"""
OCR识别HTTP服务
提供HTTP接口，接收图片URL，返回OCR识别结果
"""
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from pathlib import Path
import sys
from typing import Dict, Any, Optional, List, Tuple
import traceback
import time
import json
import numpy as np
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加src和util目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'util'))

# 导入src模块（日志配置会在main函数中根据配置文件更新）
from preprocess import load_config
from ocr_pool import OCREnginePool
from image_downloader import download_image_to_array

app = Flask(__name__)
# 配置JSON编码：不将非ASCII字符编码为Unicode转义序列，直接输出中文
app.config['JSON_AS_ASCII'] = False
CORS(app)  # 允许跨域请求

# 全局变量
ocr_pool = None
config = None


def json_response(data: Dict, status_code: int = 200):
    """
    自定义JSON响应函数，确保中文直接输出而不是Unicode转义序列
    
    Args:
        data: 要返回的数据字典
        status_code: HTTP状态码
        
    Returns:
        Flask Response对象，包含UTF-8编码的JSON数据
    """
    response = Response(
        json.dumps(data, ensure_ascii=False, indent=None),
        mimetype='application/json; charset=utf-8',
        status=status_code
    )
    return response


def convert_to_json_serializable(obj):
    """
    递归转换numpy数组和numpy标量为Python原生类型，确保JSON可序列化
    
    Args:
        obj: 需要转换的对象（可能是numpy数组、numpy标量、字典、列表等）
        
    Returns:
        转换后的Python原生类型对象
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def setup_logging(log_config: Dict):
    """
    配置日志系统
    
    Args:
        log_config: 日志配置字典
    """
    # 移除所有现有的日志处理器（包括默认的和之前配置的）
    logger.remove()
    
    # 获取配置
    log_file = log_config.get('log_file', 'logs/api_server.log')
    log_level = log_config.get('level', 'INFO')
    # 确保日志级别是大写字符串（loguru要求）
    if isinstance(log_level, str):
        log_level = log_level.upper().strip()
    else:
        log_level = str(log_level).upper().strip()
    
    # 验证日志级别是否有效
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if log_level not in valid_levels:
        print(f"警告: 无效的日志级别 '{log_level}'，使用默认值 'INFO'")
        log_level = 'INFO'
    
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
    
    # 输出日志配置信息（使用print确保在日志系统配置前也能看到）
    print(f"日志系统配置: 级别={log_level}, 文件={log_file}, 控制台输出={console_output}")
    logger.info(f"日志系统初始化完成，日志级别: {log_level}")


def init_ocr_pool():
    """初始化OCR实例池"""
    global ocr_pool, config
    
    # 如果config未加载，则加载配置（main函数中可能已经加载）
    if config is None:
        config_dir = Path(__file__).parent
        config_yaml = config_dir / 'config.yaml'
        config_json = config_dir / 'config.json'
        if config_yaml.exists():
            config = load_config(str(config_yaml))
        elif config_json.exists():
            config = load_config(str(config_json))
        else:
            config = load_config()  # 使用默认配置
    
    # 获取OCR实例池配置
    server_config = config.get('server', {})
    ocr_pool_config = server_config.get('ocr_pool', {})
    pool_size = ocr_pool_config.get('pool_size', 3)
    
    if pool_size <= 0:
        raise ValueError(f"OCR实例池大小必须大于0，当前配置: {pool_size}")
    
    # 初始化OCR实例池
    ocr_pool = OCREnginePool(pool_size=pool_size, ocr_config=config)
    
    # 打印OCR模型版本信息（从第一个实例获取）
    with ocr_pool.acquire() as ocr_engine:
        version_info = []
        if ocr_engine.paddleocr_version:
            version_info.append(f"PaddleOCR: {ocr_engine.paddleocr_version}")
        if ocr_engine.paddlepaddle_version:
            version_info.append(f"PaddlePaddle: {ocr_engine.paddlepaddle_version}")
        
        if version_info:
            logger.info(f"OCR模型版本: {', '.join(version_info)}")
        else:
            logger.info("OCR模型版本: 无法获取版本信息")
    
    logger.info(f"OCR实例池初始化完成，池大小: {pool_size}")




@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({
        'success': False,
        'message': '接口不存在'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """405错误处理（方法不允许）"""
    return jsonify({
        'success': False,
        'message': 'HTTP方法不允许'
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """500错误处理（服务器内部错误）"""
    logger.error(f"服务器内部错误: {str(error)}", exc_info=True)
    return jsonify({
        'success': False,
        'message': '服务器内部错误，请稍后重试'
    }), 500


@app.errorhandler(Exception)
def handle_exception(e):
    """全局异常处理器，捕获所有未处理的异常"""
    logger.error(f"未处理的异常: {str(e)}", exc_info=True)
    return jsonify({
        'success': False,
        'message': f'服务器错误: {str(e)}'
    }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        # 检查OCR实例池状态
        if ocr_pool is None:
            return jsonify({
                'status': 'error',
                'message': 'OCR实例池未初始化'
            }), 500
        
        available_count = ocr_pool.get_available_count()
        total_count = ocr_pool.get_total_count()
        
        # 记录OCR实例池状态
        logger.info(f"健康检查: OCR实例池状态 - 总数: {total_count}, 空闲数: {available_count}, 使用中: {total_count - available_count}")
        
        return jsonify({
            'status': 'ok',
            'message': 'OCR服务运行正常',
            'ocr_pool': {
                'available': available_count,
                'total': total_count
            }
        })
    except Exception as e:
        logger.error(f"健康检查接口出错: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': '服务异常'
        }), 500


@app.route('/ocr', methods=['POST'])
def ocr_recognize():
    """
    OCR识别接口
    
    请求格式:
    {
        "image_url": "http://example.com/image.jpg"
    }
    
    返回格式:
    {
        "success": true,
        "data": {
            "results": [
                {
                    "text": "识别文字",
                    "confidence": 0.95,
                }
            ],
            "total": 10
        },
        "message": "识别成功"
    }
    """
    try:
        # 检查OCR实例池是否已初始化
        if ocr_pool is None:
            return jsonify({
                'success': False,
                'message': 'OCR实例池未初始化'
            }), 500
        
        # 获取请求数据
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': '请求数据格式错误，需要JSON格式'
            }), 400
        
        image_url = data.get('image_url')
        if not image_url:
            return jsonify({
                'success': False,
                'message': '缺少参数: image_url'
            }), 400
        
        # 使用配置文件中的预处理设置
        use_preprocessing = config.get('preprocessing', {}).get('enabled', False)
        
        # 获取下载超时配置
        download_timeout_config = config.get('server', {}).get('download_timeout', {})
        connect_timeout = download_timeout_config.get('connect_timeout', 10)
        read_timeout = download_timeout_config.get('read_timeout', 30)
        verify_ssl = config.get('server', {}).get('verify_ssl', False)
        
        # 先申请OCR资源（不设置超时，立即返回）
        ocr_resource = None
        ocr_engine = None
        try:
            ocr_resource = ocr_pool.acquire(timeout=0)
            ocr_engine = ocr_resource.__enter__()
        except TimeoutError:
            # 无可用OCR实例
            return jsonify({
                'success': False,
                'message': 'OCR服务繁忙，无可用资源，请稍后重试'
            }), 503
        
        # 确保image_array在OCR调用期间不会被垃圾回收（多线程场景下很重要）
        image_array = None
        try:
            # 下载图片到内存（流式传递，不落盘）
            download_start_time = time.time()
            image_array, file_size, download_elapsed_time = download_image_to_array(image_url, connect_timeout=connect_timeout, read_timeout=read_timeout, verify_ssl=verify_ssl)
            
            # 获取图片尺寸
            h, w = image_array.shape[:2]
            image_shape = f"{w}x{h}"
            file_size_mb = file_size / 1024 / 1024
            
            # 执行OCR识别（直接使用内存中的图像数组）
            # 注意：保持image_array的强引用，防止在OCR调用过程中被垃圾回收
            ocr_start_time = time.time()
            ocr_results = ocr_engine.recognize_from_array(image_array, use_preprocessing=use_preprocessing)
            ocr_elapsed_time = time.time() - ocr_start_time
        except Exception as e:
            # 捕获所有异常，确保资源释放
            error_type = type(e).__name__
            error_msg = str(e) if str(e) else f"{error_type}异常"
            error_detail = f"{error_type}: {error_msg}"
            logger.error(f"OCR处理出错: {error_detail}", exc_info=True)
            raise
        finally:
            # 确保释放OCR资源
            if ocr_resource is not None:
                try:
                    ocr_resource.__exit__(None, None, None)
                except Exception as release_error:
                    logger.error(f"释放OCR资源时出错: {release_error}", exc_info=True)
        
        # 打印图片信息和OCR结果
        logger.info(f"图片URL: {image_url}, 大小: {file_size_mb:.2f}MB, 尺寸: {image_shape}, 下载耗时: {download_elapsed_time:.2f}秒")
        
        if len(ocr_results) == 0:
            logger.warning(f"未识别到任何文字")
        
        logger.info(f"OCR识别完成，识别到 {len(ocr_results)} 处文字，耗时: {ocr_elapsed_time:.2f}秒")
        
        # 格式化结果（只返回文字和置信度，不返回bbox）
        formatted_results = []
        for result in ocr_results:
            formatted_results.append({
                'text': result.get('text', ''),
                'confidence': convert_to_json_serializable(result.get('confidence', 0.0))
            })
        
        return json_response({
            'success': True,
            'data': {
                'results': formatted_results,
                'total': len(formatted_results)
            },
            'message': '识别成功'
        })
    
    except ValueError as e:
        return json_response({
            'success': False,
            'message': str(e)
        }, 400)
    
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else f"{error_type}异常"
        error_detail = f"{error_type}: {error_msg}"
        logger.error(f"OCR识别出错: {error_detail}", exc_info=True)
        return json_response({
            'success': False,
            'message': f'识别失败: {error_detail}'
        }, 500)


def download_single_image(image_url: str, idx: int, total: int, connect_timeout: int, read_timeout: int, verify_ssl: bool) -> Tuple[int, str, Optional[np.ndarray], Optional[Dict], Optional[str]]:
    """
    下载单张图片（用于线程池）
    
    Args:
        image_url: 图片URL
        idx: 图片索引（从0开始）
        total: 总图片数量
        connect_timeout: 连接超时
        read_timeout: 读取超时
        verify_ssl: SSL验证
        
    Returns:
        (索引, URL, 图片数组, 图片信息字典, 错误信息)
    """
    try:
        image_array, file_size, download_elapsed_time = download_image_to_array(
            image_url, 
            connect_timeout=connect_timeout, 
            read_timeout=read_timeout, 
            verify_ssl=verify_ssl
        )
        
        # 获取图片尺寸
        h, w = image_array.shape[:2]
        image_shape = f"{w}x{h}"
        file_size_mb = file_size / 1024 / 1024
        
        image_info = {
            'file_size_mb': file_size_mb,
            'shape': image_shape,
            'download_time': download_elapsed_time
        }
        
        logger.info(f"[{idx+1}/{total}] 图片下载完成: {image_url}, 大小: {file_size_mb:.2f}MB, 尺寸: {image_shape}, 耗时: {download_elapsed_time:.2f}秒")
        
        return (idx, image_url, image_array, image_info, None)
        
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"[{idx+1}/{total}] 图片下载失败 {image_url}: {error_msg}", exc_info=True)
        return (idx, image_url, None, None, error_msg)


@app.route('/ocr/batch', methods=['POST'])
def ocr_recognize_batch():
    """
    批量OCR识别接口
    
    请求格式:
    {
        "image_urls": [
            "http://example.com/image1.jpg",
            "http://example.com/image2.jpg"
        ]
    }
    
    返回格式:
    {
        "success": true,
        "data": [
            {
                "image_url": "http://example.com/image1.jpg",
                "results": [...],
                "total": 10
            },
            ...
        ],
        "message": "批量识别完成"
    }
    """
    try:
        # 检查OCR实例池是否已初始化
        if ocr_pool is None:
            return jsonify({
                'success': False,
                'message': 'OCR实例池未初始化'
            }), 500
        
        # 获取请求数据
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': '请求数据格式错误，需要JSON格式'
            }), 400
        
        image_urls = data.get('image_urls')
        if not image_urls or not isinstance(image_urls, list):
            return jsonify({
                'success': False,
                'message': '缺少参数: image_urls（需要数组格式）'
            }), 400
        
        # 检查最大数量限制
        batch_config = config.get('server', {}).get('batch_ocr', {})
        max_images = batch_config.get('max_images', 50)
        
        if len(image_urls) > max_images:
            return jsonify({
                'success': False,
                'message': f'图片数量超过限制，最多支持 {max_images} 张，当前 {len(image_urls)} 张'
            }), 400
        
        # 使用配置文件中的预处理设置
        use_preprocessing = config.get('preprocessing', {}).get('enabled', False)
        
        # 获取下载超时配置
        download_timeout_config = config.get('server', {}).get('download_timeout', {})
        connect_timeout = download_timeout_config.get('connect_timeout', 10)
        read_timeout = download_timeout_config.get('read_timeout', 30)
        verify_ssl = config.get('server', {}).get('verify_ssl', False)
        
        # 获取下载线程池大小配置
        download_threads = batch_config.get('download_threads', 10)
        
        logger.info(f"开始批量OCR处理，共 {len(image_urls)} 张图片，下载线程数: {download_threads}")
        
        batch_start_time = time.time()
        
        # 第一步：先申请OCR资源（确保有可用资源）
        ocr_resource = None
        ocr_engine = None
        try:
            ocr_resource = ocr_pool.acquire(timeout=0)
            ocr_engine = ocr_resource.__enter__()
        except TimeoutError:
            # 无可用OCR实例，直接返回错误
            return jsonify({
                'success': False,
                'message': 'OCR服务繁忙，无可用资源，请稍后重试'
            }), 503
        
        try:
            # 第二步：创建线程池并行下载所有图片
            download_start_time = time.time()
            downloaded_images = {}  # {idx: (image_url, image_array, image_info, error)}
            
            with ThreadPoolExecutor(max_workers=download_threads) as executor:
                # 提交所有下载任务
                future_to_idx = {
                    executor.submit(
                        download_single_image,
                        image_url,
                        idx,
                        len(image_urls),
                        connect_timeout,
                        read_timeout,
                        verify_ssl
                    ): idx
                    for idx, image_url in enumerate(image_urls)
                }
                
                # 等待所有下载任务完成
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        downloaded_images[idx] = result
                    except Exception as e:
                        logger.error(f"下载任务异常: {str(e)}", exc_info=True)
                        downloaded_images[idx] = (idx, image_urls[idx], None, None, str(e))
            
            download_elapsed_time = time.time() - download_start_time
            logger.info(f"所有图片下载完成，总耗时: {download_elapsed_time:.2f}秒")
            
            # 第三步：使用申请到的OCR资源，串行识别
            batch_results = []
            for idx in range(len(image_urls)):
                result = downloaded_images.get(idx)
                if result is None:
                    # 下载结果缺失
                    batch_results.append({
                        'image_url': image_urls[idx],
                        'success': False,
                        'results': [],
                        'total': 0,
                        'error': '下载结果缺失'
                    })
                    continue
                
                _, image_url, image_array, image_info, download_error = result
                
                # 如果下载失败，直接记录错误
                if download_error is not None or image_array is None:
                    batch_results.append({
                        'image_url': image_url,
                        'success': False,
                        'results': [],
                        'total': 0,
                        'error': download_error or '图片下载失败'
                    })
                    continue
                
                try:
                    # 执行OCR识别
                    ocr_start_time = time.time()
                    ocr_results = ocr_engine.recognize_from_array(image_array, use_preprocessing=use_preprocessing)
                    ocr_elapsed_time = time.time() - ocr_start_time
                    
                    logger.info(f"[{idx+1}/{len(image_urls)}] OCR识别完成，识别到 {len(ocr_results)} 处文字，耗时: {ocr_elapsed_time:.2f}秒")
                    
                    # 格式化结果（只返回文字和置信度，不返回bbox）
                    formatted_results = []
                    for result in ocr_results:
                        formatted_results.append({
                            'text': result.get('text', ''),
                            'confidence': convert_to_json_serializable(result.get('confidence', 0.0))
                        })
                    
                    batch_results.append({
                        'image_url': image_url,
                        'success': True,
                        'results': formatted_results,
                        'total': len(formatted_results)
                    })
                    
                except Exception as e:
                    # OCR处理失败
                    error_type = type(e).__name__
                    error_msg = str(e) if str(e) else f"{error_type}异常"
                    error_detail = f"{error_type}: {error_msg}"
                    
                    logger.warning(f"[{idx+1}/{len(image_urls)}] OCR识别失败 {image_url}: {error_detail}", exc_info=True)
                    batch_results.append({
                        'image_url': image_url,
                        'success': False,
                        'results': [],
                        'total': 0,
                        'error': error_detail
                    })
        finally:
            # 确保释放OCR资源
            if ocr_resource is not None:
                try:
                    ocr_resource.__exit__(None, None, None)
                except Exception as release_error:
                    logger.error(f"释放OCR资源时出错: {release_error}", exc_info=True)
        
        total_elapsed_time = time.time() - batch_start_time
        
        logger.info(f"批量OCR处理完成，总耗时: {total_elapsed_time:.2f}秒")
        
        return json_response({
            'success': True,
            'data': batch_results,
            'message': f'批量识别完成，共处理 {len(image_urls)} 张图片'
        })
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"批量OCR识别出错: {error_msg}", exc_info=True)
        return json_response({
            'success': False,
            'message': f'批量识别失败: {error_msg}'
        }, 500)


def main():
    """启动HTTP服务"""
    global config
    
    try:
        # 加载配置
        config_dir = Path(__file__).parent
        config_yaml = config_dir / 'config.yaml'
        config_json = config_dir / 'config.json'
        if config_yaml.exists():
            config = load_config(str(config_yaml))
        elif config_json.exists():
            config = load_config(str(config_json))
        else:
            config = load_config()  # 使用默认配置
        
        # 根据配置文件更新日志系统配置
        # 注意：loguru的logger是全局单例，重新配置会影响所有已导入的模块
        log_config = config.get('logging', {})
        setup_logging(log_config)
        
        # 初始化OCR实例池
        try:
            init_ocr_pool()
        except Exception as e:
            logger.error(f"OCR实例池初始化失败: {str(e)}", exc_info=True)
            logger.error("服务启动失败，退出")
            sys.exit(1)
        
        # 从配置文件读取服务器配置
        server_config = config.get('server', {})
        host = server_config.get('host', '0.0.0.0')
        port = server_config.get('port', 5000)
        debug = server_config.get('debug', False)
        
        logger.info("OCR识别HTTP服务")
        logger.info(f"服务地址: http://{host}:{port}")
        logger.info(f"健康检查接口: http://{host}:{port}/health")
        logger.info(f"OCR识别接口: http://{host}:{port}/ocr")
        logger.info(f"批量OCR接口: http://{host}:{port}/ocr/batch")
        
        # 启动服务
        app.run(host=host, port=port, debug=debug, use_reloader=False)
    
    except KeyboardInterrupt:
        logger.info("服务被用户中断")
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

