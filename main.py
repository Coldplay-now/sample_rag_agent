#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Agent 主程序入口
提供命令行参数解析和程序启动功能
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
from loguru import logger

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, load_config, setup_directories, setup_logging
from chat_interface import ChatInterface
from document_processor import DocumentProcessor
from vector_store import VectorStore


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="RAG Agent - 智能文档问答助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py                          # 使用默认配置启动
  python main.py --docs ./my_docs         # 指定文档目录
  python main.py --rebuild                # 重建索引后启动
  python main.py --config                 # 显示配置信息
  python main.py --test                   # 测试模式
  python main.py --debug                  # 调试模式

配置文件:
  程序会自动查找 .env 文件来加载配置
  也可以通过环境变量设置配置项

更多信息请参考项目文档。
        """
    )
    
    # 基本选项
    parser.add_argument(
        "--docs", "-d",
        type=str,
        help="文档目录路径（覆盖配置文件设置）"
    )
    
    parser.add_argument(
        "--index-dir",
        type=str,
        help="索引目录路径（覆盖配置文件设置）"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="嵌入模型名称（覆盖配置文件设置）"
    )
    
    # 操作选项
    parser.add_argument(
        "--rebuild", "-r",
        action="store_true",
        help="重建文档索引"
    )
    
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="测试模式（验证配置和连接）"
    )
    
    parser.add_argument(
        "--config", "-c",
        action="store_true",
        help="显示当前配置信息"
    )
    
    parser.add_argument(
        "--info", "-i",
        action="store_true",
        help="显示系统信息"
    )
    
    parser.add_argument(
        "--chat",
        action="store_true",
        help="启动交互式聊天界面"
    )
    
    # 调试选项
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式"
    )
    
    # 版本信息
    parser.add_argument(
        "--version",
        action="version",
        version="RAG Agent v1.0.0"
    )
    
    return parser.parse_args()


def setup_environment(args: argparse.Namespace) -> Config:
    """
    设置运行环境
    
    Args:
        args: 命令行参数
    
    Returns:
        Config: 配置对象
    """
    # 加载配置
    config = load_config()
    
    # 应用命令行参数覆盖
    if args.docs:
        config.documents_dir = Path(args.docs).resolve()
    
    if args.index_dir:
        config.index_dir = Path(args.index_dir).resolve()
    
    if args.model:
        config.embedding_model = args.model
    
    if args.debug:
        config.debug_mode = True
    
    # 设置日志级别
    if args.quiet:
        log_level = "ERROR"
    elif args.verbose or args.debug:
        log_level = "DEBUG"
    else:
        log_level = "INFO"
    
    # 设置日志和目录
    setup_logging(config)
    setup_directories(config)
    
    logger.info(f"RAG Agent 启动 - 配置加载完成")
    logger.debug(f"文档目录: {config.documents_dir}")
    logger.debug(f"索引目录: {config.index_dir}")
    
    return config


def show_config_info(config: Config) -> None:
    """
    显示配置信息
    
    Args:
        config: 配置对象
    """
    print("\n" + "="*60)
    print("⚙️  RAG Agent 配置信息")
    print("="*60)
    
    print("\n📁 路径配置:")
    print(f"  文档目录: {config.documents_dir}")
    print(f"  索引目录: {config.index_dir}")
    print(f"  缓存目录: {config.cache_dir}")
    print(f"  日志目录: {config.logs_dir}")
    
    print("\n🤖 模型配置:")
    print(f"  嵌入模型: {config.embedding_model}")
    print(f"  LLM模型: {config.deepseek_model}")
    print(f"  API地址: {config.deepseek_base_url}")
    
    print("\n📖 文档处理:")
    print(f"  分块大小: {config.chunk_size}")
    print(f"  分块重叠: {config.chunk_overlap}")
    
    print("\n🔍 检索配置:")
    print(f"  检索数量: {config.top_k}")
    print(f"  相似度阈值: {config.similarity_threshold}")
    print(f"  查询扩展: {'✅ 启用' if config.enable_query_expansion else '❌ 禁用'}")
    print(f"  重排序: {'✅ 启用' if config.enable_rerank else '❌ 禁用'}")
    
    print("\n💬 对话配置:")
    print(f"  最大Token: {config.max_tokens}")
    print(f"  温度参数: {config.temperature}")
    print(f"  历史记录: {config.max_history}")
    print(f"  流式响应: {'✅ 启用' if config.enable_streaming else '❌ 禁用'}")
    
    print("\n🔧 其他配置:")
    print(f"  调试模式: {'✅ 启用' if config.debug_mode else '❌ 禁用'}")
    
    print("\n" + "="*60)


def show_system_info(config: Config) -> None:
    """
    显示系统信息
    
    Args:
        config: 配置对象
    """
    import platform
    import psutil
    
    print("\n" + "="*60)
    print("🔧 RAG Agent 系统信息")
    print("="*60)
    
    print("\n💻 系统环境:")
    print(f"  操作系统: {platform.system()} {platform.release()}")
    print(f"  Python版本: {platform.python_version()}")
    print(f"  CPU核心数: {psutil.cpu_count()}")
    print(f"  内存总量: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"  可用内存: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    print("\n📦 依赖检查:")
    dependencies = [
        ('faiss-cpu', 'faiss'),
        ('sentence-transformers', 'sentence_transformers'),
        ('openai', 'openai'),
        ('loguru', 'loguru'),
        ('pydantic', 'pydantic'),
        ('python-dotenv', 'dotenv')
    ]
    
    for pkg_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"  ✅ {pkg_name}")
        except ImportError:
            print(f"  ❌ {pkg_name} (未安装)")
    
    print("\n📁 目录状态:")
    directories = [
        ('文档目录', config.documents_dir),
        ('索引目录', config.index_dir),
        ('缓存目录', config.cache_dir),
        ('日志目录', config.logs_dir)
    ]
    
    for name, path in directories:
        if path.exists():
            if path.is_dir():
                file_count = len(list(path.rglob('*'))) if path.exists() else 0
                print(f"  ✅ {name}: {path} ({file_count} 个文件)")
            else:
                print(f"  ⚠️  {name}: {path} (不是目录)")
        else:
            print(f"  ❌ {name}: {path} (不存在)")
    
    print("\n" + "="*60)


def test_system(config: Config) -> bool:
    """
    测试系统功能
    
    Args:
        config: 配置对象
    
    Returns:
        bool: 测试是否通过
    """
    print("\n" + "="*60)
    print("🧪 RAG Agent 系统测试")
    print("="*60)
    
    success = True
    
    try:
        # 测试配置加载
        print("\n1️⃣ 测试配置加载...")
        if not config.deepseek_api_key:
            print("  ❌ DeepSeek API密钥未配置")
            success = False
        else:
            print("  ✅ 配置加载成功")
        
        # 测试目录创建
        print("\n2️⃣ 测试目录创建...")
        setup_directories(config)
        print("  ✅ 目录创建成功")
        
        # 测试文档处理
        print("\n3️⃣ 测试文档处理...")
        doc_processor = DocumentProcessor(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        documents_path = Path(config.documents_dir)
        if documents_path.exists():
            chunks = doc_processor.process_documents(str(documents_path))
            print(f"  ✅ 文档处理成功，找到 {len(chunks)} 个分块")
        else:
            print(f"  ⚠️  文档目录不存在: {documents_path}")
        
        # 测试向量存储
        print("\n4️⃣ 测试向量存储...")
        vector_store = VectorStore(
            model_name=config.embedding_model,
            index_dir=config.index_dir
        )
        print("  ✅ 向量存储初始化成功")
        
        # 测试LLM连接
        print("\n5️⃣ 测试LLM连接...")
        from llm_client import DeepSeekClient
        
        llm_client = DeepSeekClient(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            model=config.deepseek_model
        )
        
        if llm_client.test_connection():
            print("  ✅ LLM连接测试成功")
        else:
            print("  ❌ LLM连接测试失败")
            success = False
        
    except Exception as e:
        logger.error(f"系统测试失败: {e}")
        print(f"  ❌ 测试过程中发生错误: {e}")
        success = False
    
    print("\n" + "="*60)
    if success:
        print("🎉 所有测试通过，系统运行正常")
    else:
        print("❌ 部分测试失败，请检查配置和环境")
    print("="*60)
    
    return success


def rebuild_index(config: Config) -> bool:
    """
    重建文档索引
    
    Args:
        config: 配置对象
    
    Returns:
        bool: 重建是否成功
    """
    print("\n" + "="*60)
    print("🔨 重建文档索引")
    print("="*60)
    
    try:
        # 初始化组件
        print("\n📚 初始化文档处理器...")
        doc_processor = DocumentProcessor(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        print("🔍 初始化向量存储...")
        vector_store = VectorStore(
            model_name=config.embedding_model,
            index_dir=config.index_dir
        )
        
        # 处理文档
        documents_dir = Path(config.documents_dir).resolve()
        print(f"\n📁 扫描文档目录: {documents_dir}")
        if not documents_dir.exists():
            print(f"❌ 文档目录不存在: {documents_dir}")
            return False
        
        chunks = doc_processor.process_documents(str(documents_dir))
        if not chunks:
            print("⚠️  未找到任何文档")
            return False
        
        print(f"📄 找到 {len(chunks)} 个文档分块")
        
        # 清空现有索引
        print("🗑️  清空现有索引...")
        vector_store.clear_index()
        
        # 构建新索引
        print("🔨 构建向量索引...")
        vector_store.build_index(chunks)
        
        # 保存索引
        print("💾 保存索引...")
        vector_store.save_index()
        
        print("\n✅ 索引重建完成")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"索引重建失败: {e}")
        print(f"❌ 索引重建失败: {e}")
        print("="*60)
        return False


def main() -> None:
    """
    主函数
    """
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 设置环境
        config = setup_environment(args)
        
        # 处理特殊命令
        if args.config:
            show_config_info(config)
            return
        
        if args.info:
            show_system_info(config)
            return
        
        if args.test:
            success = test_system(config)
            sys.exit(0 if success else 1)
        
        if args.rebuild:
            success = rebuild_index(config)
            if not success:
                sys.exit(1)
            
            # 如果只是重建索引，则退出
            if not any([args.config, args.info, args.test, args.chat]):
                print("\n🎉 索引重建完成，可以启动聊天界面了")
                return
        
        if args.chat:
            # 启动聊天界面
            print("\n🚀 启动RAG Agent聊天界面...")
            chat = ChatInterface(config)
            chat.run()
            return
        
        # 默认启动聊天界面（如果没有指定其他操作）
        if not any([args.config, args.info, args.test, args.rebuild]):
            print("\n🚀 启动RAG Agent聊天界面...")
            chat = ChatInterface(config)
            chat.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序运行异常: {e}")
        print(f"❌ 程序运行失败: {e}")
        if args.debug if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()