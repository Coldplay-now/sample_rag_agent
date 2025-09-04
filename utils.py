#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Agent 工具函数模块
提供通用的辅助功能
"""

import os
import re
import time
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from functools import wraps
from loguru import logger


def timer(func):
    """
    计时装饰器
    
    Args:
        func: 被装饰的函数
    
    Returns:
        装饰后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"{func.__name__} 执行时间: {execution_time:.3f}秒")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff: 延迟倍数
    
    Returns:
        装饰器函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} 第{attempt + 1}次尝试失败: {e}, "
                            f"{current_delay:.1f}秒后重试"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} 所有重试均失败")
            
            raise last_exception
        return wrapper
    return decorator


def safe_execute(func, default=None, log_error: bool = True):
    """
    安全执行函数，捕获异常并返回默认值
    
    Args:
        func: 要执行的函数
        default: 异常时返回的默认值
        log_error: 是否记录错误日志
    
    Returns:
        函数执行结果或默认值
    """
    try:
        return func()
    except Exception as e:
        if log_error:
            logger.error(f"函数执行失败: {e}")
        return default


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    计算文件哈希值
    
    Args:
        file_path: 文件路径
        algorithm: 哈希算法（md5, sha1, sha256）
    
    Returns:
        str: 文件哈希值
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def calculate_text_hash(text: str, algorithm: str = "md5") -> str:
    """
    计算文本哈希值
    
    Args:
        text: 文本内容
        algorithm: 哈希算法
    
    Returns:
        str: 文本哈希值
    """
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(text.encode('utf-8'))
    return hash_obj.hexdigest()


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 字节数
    
    Returns:
        str: 格式化后的大小字符串
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """
    格式化时间间隔
    
    Args:
        seconds: 秒数
    
    Returns:
        str: 格式化后的时间字符串
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m{secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h{minutes}m"


def clean_text(text: str) -> str:
    """
    清理文本内容
    
    Args:
        text: 原始文本
    
    Returns:
        str: 清理后的文本
    """
    if not text:
        return ""
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除首尾空白
    text = text.strip()
    
    # 移除特殊字符（保留中文、英文、数字、常用标点）
    text = re.sub(r'[^\u4e00-\u9fff\w\s.,!?;:()\[\]{}"\'-]', '', text)
    
    return text


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    截断文本
    
    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 截断后缀
    
    Returns:
        str: 截断后的文本
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    提取文本关键词（简单实现）
    
    Args:
        text: 文本内容
        max_keywords: 最大关键词数量
    
    Returns:
        List[str]: 关键词列表
    """
    if not text:
        return []
    
    # 简单的关键词提取：基于词频
    # 移除标点符号
    clean_text_content = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', text)
    
    # 分词（简单按空格分割）
    words = clean_text_content.split()
    
    # 过滤短词和常见停用词
    stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
    
    words = [word for word in words if len(word) > 1 and word not in stop_words]
    
    # 统计词频
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # 按频率排序
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # 返回前N个关键词
    return [word for word, freq in sorted_words[:max_keywords]]


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    验证文件路径
    
    Args:
        file_path: 文件路径
        must_exist: 是否必须存在
    
    Returns:
        Path: 验证后的路径对象
    
    Raises:
        ValueError: 路径无效
        FileNotFoundError: 文件不存在（当must_exist=True时）
    """
    if not file_path:
        raise ValueError("文件路径不能为空")
    
    path = Path(file_path).resolve()
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    
    return path


def ensure_directory(dir_path: Union[str, Path]) -> Path:
    """
    确保目录存在，不存在则创建
    
    Args:
        dir_path: 目录路径
    
    Returns:
        Path: 目录路径对象
    """
    path = Path(dir_path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json_file(file_path: Union[str, Path], default: Any = None) -> Any:
    """
    加载JSON文件
    
    Args:
        file_path: 文件路径
        default: 加载失败时的默认值
    
    Returns:
        JSON数据或默认值
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载JSON文件失败 {file_path}: {e}")
        return default


def save_json_file(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    保存JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        indent: 缩进空格数
    
    Returns:
        bool: 是否保存成功
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        
        return True
    except Exception as e:
        logger.error(f"保存JSON文件失败 {file_path}: {e}")
        return False


def get_file_stats(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    获取文件统计信息
    
    Args:
        file_path: 文件路径
    
    Returns:
        Dict[str, Any]: 文件统计信息
    """
    path = Path(file_path)
    
    if not path.exists():
        return {}
    
    stat = path.stat()
    
    return {
        'size': stat.st_size,
        'size_formatted': format_file_size(stat.st_size),
        'created': datetime.fromtimestamp(stat.st_ctime),
        'modified': datetime.fromtimestamp(stat.st_mtime),
        'accessed': datetime.fromtimestamp(stat.st_atime),
        'is_file': path.is_file(),
        'is_dir': path.is_dir(),
        'extension': path.suffix.lower(),
        'name': path.name,
        'stem': path.stem
    }


def find_files(directory: Union[str, Path], 
               pattern: str = "*", 
               recursive: bool = True,
               include_dirs: bool = False) -> List[Path]:
    """
    查找文件
    
    Args:
        directory: 搜索目录
        pattern: 文件模式
        recursive: 是否递归搜索
        include_dirs: 是否包含目录
    
    Returns:
        List[Path]: 找到的文件列表
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    if recursive:
        files = directory.rglob(pattern)
    else:
        files = directory.glob(pattern)
    
    result = []
    for file in files:
        if include_dirs or file.is_file():
            result.append(file)
    
    return sorted(result)


def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """
    创建进度条字符串
    
    Args:
        current: 当前进度
        total: 总数
        width: 进度条宽度
    
    Returns:
        str: 进度条字符串
    """
    if total == 0:
        return "[" + "="*width + "] 100%"
    
    percentage = min(100, int((current / total) * 100))
    filled = int((current / total) * width)
    bar = "=" * filled + "-" * (width - filled)
    
    return f"[{bar}] {percentage}% ({current}/{total})"


def normalize_query(query: str) -> str:
    """
    标准化查询字符串
    
    Args:
        query: 原始查询
    
    Returns:
        str: 标准化后的查询
    """
    if not query:
        return ""
    
    # 转换为小写
    query = query.lower()
    
    # 移除多余空白
    query = re.sub(r'\s+', ' ', query)
    
    # 移除首尾空白
    query = query.strip()
    
    return query


def split_text_by_sentences(text: str, max_length: int = 1000) -> List[str]:
    """
    按句子分割文本，确保每段不超过最大长度
    
    Args:
        text: 原始文本
        max_length: 最大长度
    
    Returns:
        List[str]: 分割后的文本段落
    """
    if not text:
        return []
    
    # 按句号、问号、感叹号分割
    sentences = re.split(r'[。！？.!?]', text)
    
    result = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # 如果当前句子本身就超过最大长度，强制分割
        if len(sentence) > max_length:
            if current_chunk:
                result.append(current_chunk)
                current_chunk = ""
            
            # 按字符强制分割长句子
            for i in range(0, len(sentence), max_length):
                result.append(sentence[i:i + max_length])
        else:
            # 检查添加当前句子是否会超过长度限制
            if len(current_chunk) + len(sentence) + 1 > max_length:
                if current_chunk:
                    result.append(current_chunk)
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
    
    # 添加最后一个分块
    if current_chunk:
        result.append(current_chunk)
    
    return result


def calculate_similarity_score(text1: str, text2: str) -> float:
    """
    计算两个文本的相似度（简单实现）
    
    Args:
        text1: 文本1
        text2: 文本2
    
    Returns:
        float: 相似度分数（0-1）
    """
    if not text1 or not text2:
        return 0.0
    
    # 简单的基于词汇重叠的相似度计算
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


class PerformanceMonitor:
    """
    性能监控器
    """
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str) -> None:
        """
        开始计时
        
        Args:
            name: 计时器名称
        """
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """
        结束计时
        
        Args:
            name: 计时器名称
        
        Returns:
            float: 执行时间（秒）
        """
        if name not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[name]
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(duration)
        del self.start_times[name]
        
        return duration
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """
        获取统计信息
        
        Args:
            name: 计时器名称
        
        Returns:
            Dict[str, float]: 统计信息
        """
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        times = self.metrics[name]
        
        return {
            'count': len(times),
            'total': sum(times),
            'average': sum(times) / len(times),
            'min': min(times),
            'max': max(times)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        获取所有统计信息
        
        Returns:
            Dict[str, Dict[str, float]]: 所有统计信息
        """
        return {name: self.get_stats(name) for name in self.metrics}
    
    def reset(self) -> None:
        """
        重置所有统计信息
        """
        self.metrics.clear()
        self.start_times.clear()


# 全局性能监控器实例
performance_monitor = PerformanceMonitor()


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试工具函数...")
    
    # 测试文本处理
    test_text = "这是一个测试文本。它包含多个句子！还有一些特殊字符@#$%。"
    print(f"原文本: {test_text}")
    print(f"清理后: {clean_text(test_text)}")
    print(f"关键词: {extract_keywords(test_text)}")
    
    # 测试时间格式化
    print(f"\n时间格式化测试:")
    print(f"0.5秒: {format_duration(0.5)}")
    print(f"65秒: {format_duration(65)}")
    print(f"3661秒: {format_duration(3661)}")
    
    # 测试文件大小格式化
    print(f"\n文件大小格式化测试:")
    print(f"1024字节: {format_file_size(1024)}")
    print(f"1048576字节: {format_file_size(1048576)}")
    
    # 测试进度条
    print(f"\n进度条测试:")
    print(create_progress_bar(25, 100))
    print(create_progress_bar(50, 100))
    print(create_progress_bar(100, 100))
    
    print("\n✅ 工具函数测试完成")