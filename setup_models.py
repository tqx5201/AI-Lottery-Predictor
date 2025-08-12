#!/usr/bin/env python3
"""
模型安装脚本
自动安装和配置所需的NLP模型
"""

import os
import sys
import subprocess
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_spacy_models():
    """安装SpaCy语言模型"""
    models_to_install = [
        "en_core_web_sm",  # 英文模型
        "zh_core_web_sm"   # 中文模型
    ]
    
    for model in models_to_install:
        try:
            logger.info(f"正在安装SpaCy模型: {model}")
            result = subprocess.run([
                sys.executable, "-m", "spacy", "download", model
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✅ SpaCy模型 {model} 安装成功")
            else:
                logger.warning(f"⚠️ SpaCy模型 {model} 安装失败: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ SpaCy模型 {model} 安装超时")
        except Exception as e:
            logger.error(f"❌ SpaCy模型 {model} 安装异常: {e}")

def check_pytorch_version():
    """检查PyTorch版本"""
    try:
        import torch
        version = torch.__version__
        logger.info(f"当前PyTorch版本: {version}")
        
        # 检查版本是否满足要求
        major, minor = map(int, version.split('.')[:2])
        if major < 2 or (major == 2 and minor < 1):
            logger.warning(f"⚠️ PyTorch版本 {version} 可能不满足要求，建议升级到 >= 2.1.0")
            logger.info("升级命令: pip install torch>=2.1.0 --upgrade")
        else:
            logger.info(f"✅ PyTorch版本满足要求")
            
    except ImportError:
        logger.error("❌ PyTorch未安装")
        logger.info("安装命令: pip install torch>=2.1.0")

def check_transformers():
    """检查Transformers库"""
    try:
        import transformers
        version = transformers.__version__
        logger.info(f"Transformers版本: {version}")
        logger.info("✅ Transformers可用")
        
        # 测试基本功能
        from transformers import pipeline
        logger.info("✅ Transformers pipeline可用")
        
    except ImportError as e:
        logger.error(f"❌ Transformers导入失败: {e}")
        logger.info("安装命令: pip install transformers>=4.20.0")

def check_nltk_data():
    """检查NLTK数据"""
    try:
        import nltk
        
        # 检查必要的数据包
        required_data = ['punkt', 'stopwords', 'wordnet']
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}' if data_name == 'punkt' 
                              else f'corpora/{data_name}')
                logger.info(f"✅ NLTK数据 {data_name} 已安装")
            except LookupError:
                logger.info(f"正在下载NLTK数据: {data_name}")
                nltk.download(data_name, quiet=True)
                logger.info(f"✅ NLTK数据 {data_name} 下载完成")
                
    except ImportError:
        logger.error("❌ NLTK未安装")
        logger.info("安装命令: pip install nltk>=3.7")

def main():
    """主函数"""
    logger.info("🚀 开始模型安装和配置")
    logger.info("=" * 50)
    
    # 1. 检查PyTorch
    logger.info("1. 检查PyTorch...")
    check_pytorch_version()
    
    # 2. 检查Transformers
    logger.info("\n2. 检查Transformers...")
    check_transformers()
    
    # 3. 检查和下载NLTK数据
    logger.info("\n3. 检查NLTK数据...")
    check_nltk_data()
    
    # 4. 安装SpaCy模型
    logger.info("\n4. 安装SpaCy模型...")
    install_spacy_models()
    
    logger.info("\n" + "=" * 50)
    logger.info("🎉 模型安装和配置完成！")
    logger.info("现在可以运行测试脚本: python test_core_functionality.py")

if __name__ == "__main__":
    main()
