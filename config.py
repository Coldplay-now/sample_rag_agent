#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块
负责加载和验证环境变量配置
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
from loguru import logger


class Config(BaseModel):
    """配置类，使用 Pydantic 进行配置验证"""
    
    # API 配置
    deepseek_api_key: str = Field(..., description="DeepSeek API 密钥")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", description="DeepSeek API 基础URL")
    deepseek_model: str = Field(default="deepseek-chat", description="DeepSeek 模型名称")
    
    # 文档预处理配置
    chunk_size: int = Field(default=500, description="文档分块大小")
    chunk_overlap: int = Field(default=50, description="分块重叠大小")
    max_chunk_size: int = Field(default=1000, description="最大分块大小")
    
    # 检索配置
    retrieval_top_k: int = Field(default=5, description="检索返回的文档数量")
    similarity_threshold: float = Field(default=0.7, description="相似度阈值")
    max_context_length: int = Field(default=4000, description="最大上下文长度")
    
    # 向量模型配置
    embedding_model: str = Field(default="BAAI/bge-small-zh", description="向量化模型名称")
    embedding_device: str = Field(default="cpu", description="向量化设备")
    embedding_batch_size: int = Field(default=32, description="向量化批处理大小")
    
    # 文件路径配置
    data_dir: str = Field(default="./data", description="数据目录路径")
    documents_dir: str = Field(default="./data/documents", description="文档目录路径")
    index_dir: str = Field(default="./index", description="索引目录路径")
    cache_dir: str = Field(default="./cache", description="缓存目录路径")
    logs_dir: str = Field(default="./logs", description="日志目录路径")
    
    # 检索配置扩展
    top_k: int = Field(default=5, description="检索返回的文档数量")
    enable_query_expansion: bool = Field(default=False, description="是否启用查询扩展")
    enable_rerank: bool = Field(default=True, description="是否启用重排序")
    
    # LLM 配置
    max_tokens: int = Field(default=2048, description="最大生成token数")
    temperature: float = Field(default=0.7, description="生成温度参数")
    max_history: int = Field(default=10, description="最大对话历史数")
    enable_streaming: bool = Field(default=True, description="是否启用流式响应")
    
    # 系统配置
    log_level: str = Field(default="INFO", description="日志级别")
    max_memory_mb: int = Field(default=2048, description="最大内存使用量(MB)")
    debug_mode: bool = Field(default=False, description="是否启用调试模式")
    stream_response: bool = Field(default=True, description="是否启用流式响应")
    
    @validator('chunk_size')
    def validate_chunk_size(cls, v):
        """验证分块大小"""
        if v <= 0 or v > 2000:
            raise ValueError("分块大小必须在 1-2000 之间")
        return v
    
    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        """验证分块重叠大小"""
        if v < 0:
            raise ValueError("分块重叠大小不能为负数")
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError("分块重叠大小不能大于等于分块大小")
        return v
    
    @validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        """验证相似度阈值"""
        if not 0 <= v <= 1:
            raise ValueError("相似度阈值必须在 0-1 之间")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """验证日志级别"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"日志级别必须是 {valid_levels} 中的一个")
        return v.upper()


def load_config(env_file: Optional[str] = None) -> Config:
    """
    加载配置
    
    Args:
        env_file: .env 文件路径，默认为当前目录下的 .env
    
    Returns:
        Config: 配置对象
    """
    # 确定 .env 文件路径
    if env_file is None:
        env_file = Path.cwd() / ".env"
    else:
        env_file = Path(env_file)
    
    # 加载环境变量
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"已加载配置文件: {env_file}")
    else:
        logger.warning(f"配置文件不存在: {env_file}，将使用环境变量和默认值")
    
    # 从环境变量构建配置
    config_data = {
        # API 配置
        'deepseek_api_key': os.getenv('DEEPSEEK_API_KEY', ''),
        'deepseek_base_url': os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
        'deepseek_model': os.getenv('DEEPSEEK_MODEL', 'deepseek-chat'),
        
        # 文档预处理配置
        'chunk_size': int(os.getenv('CHUNK_SIZE', '500')),
        'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '50')),
        'max_chunk_size': int(os.getenv('MAX_CHUNK_SIZE', '1000')),
        
        # 检索配置
        'retrieval_top_k': int(os.getenv('RETRIEVAL_TOP_K', '5')),
        'similarity_threshold': float(os.getenv('SIMILARITY_THRESHOLD', '0.7')),
        'max_context_length': int(os.getenv('MAX_CONTEXT_LENGTH', '4000')),
        
        # 向量模型配置
        'embedding_model': os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-zh'),
        'embedding_device': os.getenv('EMBEDDING_DEVICE', 'cpu'),
        'embedding_batch_size': int(os.getenv('EMBEDDING_BATCH_SIZE', '32')),
        
        # 文件路径配置
        'data_dir': os.getenv('DATA_DIR', './data'),
        'documents_dir': os.getenv('DOCUMENTS_DIR', './data/documents'),
        'index_dir': os.getenv('INDEX_DIR', './index'),
        'cache_dir': os.getenv('CACHE_DIR', './cache'),
        'logs_dir': os.getenv('LOGS_DIR', './logs'),
        
        # 检索配置扩展
        'top_k': int(os.getenv('TOP_K', '5')),
        'enable_query_expansion': os.getenv('ENABLE_QUERY_EXPANSION', 'false').lower() == 'true',
        'enable_rerank': os.getenv('ENABLE_RERANK', 'true').lower() == 'true',
        
        # LLM 配置
        'max_tokens': int(os.getenv('MAX_TOKENS', '2048')),
        'temperature': float(os.getenv('TEMPERATURE', '0.7')),
        'max_history': int(os.getenv('MAX_HISTORY', '10')),
        'enable_streaming': os.getenv('ENABLE_STREAMING', 'true').lower() == 'true',
        
        # 系统配置
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'max_memory_mb': int(os.getenv('MAX_MEMORY_MB', '2048')),
        'debug_mode': os.getenv('DEBUG_MODE', 'false').lower() == 'true',
        'stream_response': os.getenv('STREAM_RESPONSE', 'true').lower() == 'true',
    }
    
    try:
        config = Config(**config_data)
        logger.info("配置加载成功")
        return config
    except Exception as e:
        logger.error(f"配置验证失败: {e}")
        raise


def setup_directories(config: Config) -> None:
    """
    创建必要的目录
    
    Args:
        config: 配置对象
    """
    directories = [
        config.data_dir,
        config.documents_dir,
        config.index_dir,
        config.cache_dir,
        config.logs_dir
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"确保目录存在: {directory}")


def setup_logging(config: Config) -> None:
    """
    设置日志配置
    
    Args:
        config: 配置对象
    """
    # 移除默认的日志处理器
    logger.remove()
    
    # 添加控制台日志
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=config.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 添加文件日志
    logger.add(
        "./logs/rag_agent.log",
        level=config.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )
    
    logger.info(f"日志系统初始化完成，级别: {config.log_level}")


if __name__ == "__main__":
    # 测试配置加载
    try:
        config = load_config()
        setup_directories(config)
        setup_logging(config)
        logger.info("配置模块测试成功")
    except Exception as e:
        print(f"配置模块测试失败: {e}")