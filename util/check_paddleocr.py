"""
PaddleOCR安装检测脚本
检测本地是否已安装PaddleOCR引擎和模型
"""
import sys
from pathlib import Path
import time


def check_paddleocr_installation():
    """
    检测PaddleOCR和模型是否已安装
    
    Returns:
        dict: 检测结果，包含：
            - paddleocr_installed: PaddleOCR包是否已安装
            - paddlepaddle_installed: PaddlePaddle包是否已安装
            - models_downloaded: 模型是否已下载
            - paddleocr_version: PaddleOCR版本（如果已安装）
            - paddlepaddle_version: PaddlePaddle版本（如果已安装）
            - model_path: 模型路径（如果已下载）
            - model_details: 模型详细信息
    """
    result = {
        'paddleocr_installed': False,
        'paddlepaddle_installed': False,
        'models_downloaded': False,
        'paddleocr_version': None,
        'paddlepaddle_version': None,
        'model_path': None,
        'model_details': []
    }
    
    # 检测PaddleOCR包
    print("正在检测PaddleOCR包...")
    try:
        import paddleocr
        result['paddleocr_installed'] = True
        try:
            result['paddleocr_version'] = paddleocr.__version__
            print(f"  ✅ PaddleOCR已安装，版本: {paddleocr.__version__}")
        except AttributeError:
            print("  ✅ PaddleOCR已安装（无法获取版本号）")
    except ImportError:
        print("  ❌ PaddleOCR未安装")
    
    # 检测PaddlePaddle包
    print("\n正在检测PaddlePaddle包...")
    try:
        import paddle
        result['paddlepaddle_installed'] = True
        try:
            result['paddlepaddle_version'] = paddle.__version__
            print(f"  ✅ PaddlePaddle已安装，版本: {paddle.__version__}")
        except AttributeError:
            print("  ✅ PaddlePaddle已安装（无法获取版本号）")
    except ImportError:
        print("  ❌ PaddlePaddle未安装")
    
    # 检测模型文件
    # 新版本PaddleOCR (3.x) 使用 .paddlex 目录
    # 旧版本使用 .paddleocr 目录
    print("\n正在检测模型文件...")
    home_dir = Path.home()
    
    # 检查新版本模型目录 (.paddlex)
    paddlex_dir = home_dir / '.paddlex' / 'official_models'
    # 检查旧版本模型目录 (.paddleocr)
    paddleocr_dir = home_dir / '.paddleocr'
    
    found_models = []
    model_path_used = None
    
    # 优先检查新版本模型目录
    if paddlex_dir.exists():
        print(f"  检测到新版本模型目录: {paddlex_dir}")
        model_path_used = paddlex_dir
        
        # 新版本PaddleOCR使用的模型名称
        new_version_models = {
            'PP-OCRv5_server_det': '文字检测模型',
            'PP-OCRv5_server_rec': '文字识别模型',
            'PP-LCNet_x1_0_textline_ori': '文字方向分类模型',
            'PP-LCNet_x1_0_doc_ori': '文档方向分类模型',
            'UVDoc': '文档理解模型'
        }
        
        for model_dir, model_name in new_version_models.items():
            model_path = paddlex_dir / model_dir
            if model_path.exists():
                # 检查是否有模型文件（可能是.pdmodel、.pdiparams或其他格式）
                model_files = list(model_path.rglob('*.pdmodel')) + list(model_path.rglob('*.onnx'))
                param_files = list(model_path.rglob('*.pdiparams'))
                config_files = list(model_path.rglob('*.yaml')) + list(model_path.rglob('*.yml'))
                
                # 只要有模型文件、参数文件或配置文件，就认为模型存在
                if model_files or param_files or config_files or any(model_path.iterdir()):
                    found_models.append(model_dir)
                    model_info = {
                        'name': model_name,
                        'path': str(model_path),
                        'has_model': len(model_files) > 0,
                        'has_params': len(param_files) > 0
                    }
                    result['model_details'].append(model_info)
                    print(f"  ✅ {model_name}已下载")
                    print(f"     路径: {model_path}")
    
    # 如果新版本目录没有找到模型，检查旧版本目录
    if not found_models and paddleocr_dir.exists():
        print(f"  检测到旧版本模型目录: {paddleocr_dir}")
        model_path_used = paddleocr_dir
        
        # 旧版本PaddleOCR使用的模型名称
        old_version_models = {
            'ch_ppocr_det': '文字检测模型',
            'ch_ppocr_rec': '文字识别模型',
            'ch_ppocr_cls': '文字方向分类模型'
        }
        
        for model_dir, model_name in old_version_models.items():
            model_path = paddleocr_dir / model_dir
            if model_path.exists():
                model_files = list(model_path.glob('*.pdmodel'))
                param_files = list(model_path.glob('*.pdiparams'))
                
                if model_files or param_files:
                    found_models.append(model_dir)
                    model_info = {
                        'name': model_name,
                        'path': str(model_path),
                        'has_model': len(model_files) > 0,
                        'has_params': len(param_files) > 0
                    }
                    result['model_details'].append(model_info)
                    print(f"  ✅ {model_name}已下载")
                    print(f"     路径: {model_path}")
    
    if found_models:
        result['models_downloaded'] = True
        result['model_path'] = str(model_path_used)
        print(f"\n  ✅ 共找到 {len(found_models)} 个模型")
    else:
        if not paddlex_dir.exists() and not paddleocr_dir.exists():
            print(f"  ❌ 模型目录不存在")
            print(f"     新版本目录: {paddlex_dir}")
            print(f"     旧版本目录: {paddleocr_dir}")
        else:
            print(f"\n  ❌ 未找到已下载的模型文件")
    
    return result


def download_models(use_angle_cls=True, lang='ch'):
    """
    自动下载PaddleOCR模型
    
    Args:
        use_angle_cls: 是否使用角度分类器
        lang: 语言类型，'ch'表示中英文混合
        
    Returns:
        bool: 下载是否成功
    """
    print("\n" + "=" * 60)
    print("开始下载PaddleOCR模型")
    print("=" * 60)
    print("\n⚠️  注意：模型文件较大（约几百MB），下载可能需要几分钟时间")
    print("   请确保网络连接正常，不要中断下载过程...\n")
    
    try:
        from paddleocr import PaddleOCR
        
        print("正在初始化PaddleOCR（这将触发模型自动下载）...")
        print("请稍候，正在下载模型文件...\n")
        
        # 初始化PaddleOCR，这会自动下载模型
        # 注意：新版本PaddleOCR不再支持use_gpu参数
        # use_angle_cls 在新版本中已弃用，改用 use_textline_orientation
        try:
            # 尝试使用新参数
            ocr = PaddleOCR(
                use_textline_orientation=use_angle_cls,
                lang=lang
            )
        except TypeError:
            # 如果新参数不支持，使用旧参数（兼容旧版本）
            ocr = PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang
            )
        
        print("\n✅ 模型下载完成！")
        return True
        
    except ImportError:
        print("❌ 错误：PaddleOCR未安装，无法下载模型")
        print("   请先运行: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"\n❌ 下载模型时出错: {str(e)}")
        print("   可能的原因：")
        print("   1. 网络连接问题")
        print("   2. 磁盘空间不足")
        print("   3. 权限问题")
        import traceback
        traceback.print_exc()
        return False


def print_summary(result):
    """打印检测结果摘要"""
    print("\n" + "=" * 60)
    print("检测结果摘要")
    print("=" * 60)
    
    print(f"\nPaddleOCR包: {'✅ 已安装' if result['paddleocr_installed'] else '❌ 未安装'}")
    if result['paddleocr_version']:
        print(f"  版本: {result['paddleocr_version']}")
    
    print(f"\nPaddlePaddle包: {'✅ 已安装' if result['paddlepaddle_installed'] else '❌ 未安装'}")
    if result['paddlepaddle_version']:
        print(f"  版本: {result['paddlepaddle_version']}")
    
    print(f"\n模型文件: {'✅ 已下载' if result['models_downloaded'] else '❌ 未下载'}")
    if result['model_path']:
        print(f"  路径: {result['model_path']}")
        if result['model_details']:
            print(f"  已下载的模型:")
            for detail in result['model_details']:
                print(f"    - {detail['name']}")
    
    print("\n" + "=" * 60)
    
    # 给出建议
    if not result['paddleocr_installed'] or not result['paddlepaddle_installed']:
        print("\n⚠️  建议：请先安装依赖包")
        print("   pip install -r requirements.txt")
    
    if result['paddleocr_installed'] and result['paddlepaddle_installed'] and not result['models_downloaded']:
        print("\n⚠️  模型文件未下载")
        print("   提示：可以运行此脚本自动下载模型")
    
    if result['paddleocr_installed'] and result['paddlepaddle_installed'] and result['models_downloaded']:
        print("\n✅ 所有组件已就绪，可以正常运行主程序！")
    
    print("=" * 60)
    
    return result


def pre_download_models_for_docker(use_angle_cls=True, lang='ch'):
    """
    为Docker构建预下载PaddleOCR模型
    这个函数专门用于Docker构建时调用，会显示详细的下载信息
    
    Args:
        use_angle_cls: 是否使用角度分类器
        lang: 语言类型，'ch'表示中英文混合
        
    Returns:
        bool: 下载是否成功
    """
    print("开始预下载PaddleOCR模型...")
    print("这可能需要几分钟时间，请耐心等待...\n")
    
    try:
        from paddleocr import PaddleOCR
        import pathlib
        
        # 初始化PaddleOCR，这会自动下载模型
        # 使用与生产环境相同的配置
        try:
            ocr = PaddleOCR(
                use_textline_orientation=use_angle_cls,
                lang=lang
            )
        except TypeError:
            # 兼容旧版本
            ocr = PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang
            )
        
        print("\n✅ PaddleOCR模型预下载完成！")
        
        # 显示模型路径和大小
        home = pathlib.Path.home()
        paddlex_dir = home / ".paddlex"
        paddleocr_dir = home / ".paddleocr"
        
        if paddlex_dir.exists():
            print(f"模型存储路径: {paddlex_dir}")
            # 计算模型文件总大小
            total_size = sum(f.stat().st_size for f in paddlex_dir.rglob('*') if f.is_file())
            print(f"模型总大小: {total_size / 1024 / 1024:.2f} MB")
        elif paddleocr_dir.exists():
            print(f"模型存储路径: {paddleocr_dir}")
            total_size = sum(f.stat().st_size for f in paddleocr_dir.rglob('*') if f.is_file())
            print(f"模型总大小: {total_size / 1024 / 1024:.2f} MB")
        
        return True
        
    except ImportError:
        print("❌ 错误：PaddleOCR未安装，无法下载模型")
        print("   请先运行: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"\n❌ 模型下载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("PaddleOCR安装检测工具")
    print("=" * 60)
    print()
    
    result = check_paddleocr_installation()
    summary_result = print_summary(result)
    
    # 如果包已安装但模型未下载，自动下载模型
    if (result['paddleocr_installed'] and 
        result['paddlepaddle_installed'] and 
        not result['models_downloaded']):
        
        print("\n" + "=" * 60)
        print("检测到模型未下载，开始自动下载...")
        print("=" * 60)
        
        download_success = download_models(use_angle_cls=True, lang='ch')
        
        if download_success:
            # 重新检测模型
            print("\n正在重新检测模型文件...")
            time.sleep(1)  # 等待文件系统更新
            result = check_paddleocr_installation()
            print_summary(result)
        else:
            print("\n❌ 模型下载失败，请检查网络连接后重试")
    
    return result


if __name__ == '__main__':
    # 如果通过命令行参数指定为Docker构建模式
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--docker':
        # Docker构建模式：直接预下载模型
        success = pre_download_models_for_docker(use_angle_cls=True, lang='ch')
        sys.exit(0 if success else 1)
    else:
        # 正常模式：运行主函数
        main()

