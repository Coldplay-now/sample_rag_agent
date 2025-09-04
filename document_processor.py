#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档处理模块
负责扫描、读取和分块处理 Markdown 文档
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
import markdown


@dataclass
class DocumentChunk:
    """文档分块数据类"""
    content: str  # 分块内容
    metadata: Dict[str, Any]  # 元数据
    chunk_id: str  # 分块唯一标识
    file_path: str  # 文件路径
    chunk_index: int  # 分块索引


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, max_chunk_size: int = 1000):
        """
        初始化文档处理器
        
        Args:
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
            max_chunk_size: 最大分块大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_size = max_chunk_size
        self.md_parser = markdown.Markdown()
        
        logger.info(f"文档处理器初始化完成 - 分块大小: {chunk_size}, 重叠: {chunk_overlap}")
    
    def scan_documents(self, documents_dir: str) -> List[str]:
        """
        扫描文档目录，获取所有 .md 文件
        
        Args:
            documents_dir: 文档目录路径
        
        Returns:
            List[str]: Markdown 文件路径列表
        """
        documents_path = Path(documents_dir)
        
        if not documents_path.exists():
            logger.warning(f"文档目录不存在: {documents_dir}")
            return []
        
        # 只扫描当前目录下的 .md 文件，不包括子目录
        md_files = []
        for file_path in documents_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == '.md':
                md_files.append(str(file_path.absolute()))
        
        logger.info(f"扫描到 {len(md_files)} 个 Markdown 文件")
        return sorted(md_files)
    
    def read_document(self, file_path: str) -> Optional[str]:
        """
        读取文档内容
        
        Args:
            file_path: 文件路径
        
        Returns:
            Optional[str]: 文档内容，读取失败返回 None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                logger.warning(f"文档为空: {file_path}")
                return None
            
            logger.debug(f"成功读取文档: {file_path}, 长度: {len(content)}")
            return content
        
        except Exception as e:
            logger.error(f"读取文档失败 {file_path}: {e}")
            return None
    
    def clean_markdown_content(self, content: str) -> str:
        """
        清理 Markdown 内容
        
        Args:
            content: 原始内容
        
        Returns:
            str: 清理后的内容
        """
        # 移除多余的空行
        lines = content.split('\n')
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            line = line.strip()
            if not line:
                if not prev_empty:
                    cleaned_lines.append('')
                prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False
        
        # 移除开头和结尾的空行
        while cleaned_lines and not cleaned_lines[0]:
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)
    
    def split_text_by_sentences(self, text: str) -> List[str]:
        """
        按句子分割文本
        
        Args:
            text: 输入文本
        
        Returns:
            List[str]: 句子列表
        """
        # 简单的中英文句子分割
        import re
        
        # 按句号、问号、感叹号分割
        sentences = re.split(r'[。！？.!?]', text)
        
        # 清理空句子并保留标点
        result = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence:
                # 为非最后一个句子添加标点
                if i < len(sentences) - 1:
                    # 找到原文中的标点
                    original_pos = text.find(sentence) + len(sentence)
                    if original_pos < len(text):
                        punct = text[original_pos]
                        if punct in '。！？.!?':
                            sentence += punct
                result.append(sentence)
        
        return result
    
    def create_chunks(self, content: str, file_path: str) -> List[DocumentChunk]:
        """
        将文档内容分块
        
        Args:
            content: 文档内容
            file_path: 文件路径
        
        Returns:
            List[DocumentChunk]: 文档分块列表
        """
        # 清理内容
        cleaned_content = self.clean_markdown_content(content)
        
        # 按句子分割
        sentences = self.split_text_by_sentences(cleaned_content)
        
        if not sentences:
            logger.warning(f"文档没有有效句子: {file_path}")
            return []
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            # 检查添加当前句子后是否超过分块大小
            potential_chunk = current_chunk + ("\n" if current_chunk else "") + sentence
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # 当前分块已满，创建分块
                if current_chunk:
                    chunk = self._create_chunk(
                        content=current_chunk,
                        file_path=file_path,
                        chunk_index=chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # 处理重叠
                if self.chunk_overlap > 0 and current_chunk:
                    # 从当前分块末尾取重叠部分
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + "\n" + sentence if overlap_text else sentence
                else:
                    current_chunk = sentence
        
        # 处理最后一个分块
        if current_chunk:
            chunk = self._create_chunk(
                content=current_chunk,
                file_path=file_path,
                chunk_index=chunk_index
            )
            chunks.append(chunk)
        
        logger.info(f"文档 {Path(file_path).name} 分块完成: {len(chunks)} 个分块")
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """
        获取重叠文本
        
        Args:
            text: 原始文本
        
        Returns:
            str: 重叠文本
        """
        if len(text) <= self.chunk_overlap:
            return text
        
        # 从末尾取指定长度的文本
        overlap_text = text[-self.chunk_overlap:]
        
        # 尝试在句子边界处截断
        sentences = self.split_text_by_sentences(overlap_text)
        if len(sentences) > 1:
            # 取最后几个完整句子
            return '\n'.join(sentences[-2:])
        
        return overlap_text
    
    def _create_chunk(self, content: str, file_path: str, chunk_index: int) -> DocumentChunk:
        """
        创建文档分块
        
        Args:
            content: 分块内容
            file_path: 文件路径
            chunk_index: 分块索引
        
        Returns:
            DocumentChunk: 文档分块对象
        """
        # 生成分块唯一标识
        chunk_id = self._generate_chunk_id(file_path, chunk_index, content)
        
        # 创建元数据
        file_path_obj = Path(file_path)
        metadata = {
            'file_name': file_path_obj.name,
            'file_path': str(file_path_obj.absolute()),
            'chunk_index': chunk_index,
            'chunk_length': len(content),
            'file_size': file_path_obj.stat().st_size if file_path_obj.exists() else 0,
            'file_modified': file_path_obj.stat().st_mtime if file_path_obj.exists() else 0,
        }
        
        return DocumentChunk(
            content=content,
            metadata=metadata,
            chunk_id=chunk_id,
            file_path=str(file_path_obj.absolute()),
            chunk_index=chunk_index
        )
    
    def _generate_chunk_id(self, file_path: str, chunk_index: int, content: str) -> str:
        """
        生成分块唯一标识
        
        Args:
            file_path: 文件路径
            chunk_index: 分块索引
            content: 分块内容
        
        Returns:
            str: 分块唯一标识
        """
        # 使用文件路径、分块索引和内容哈希生成唯一ID
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        file_name = Path(file_path).stem
        return f"{file_name}_{chunk_index}_{content_hash}"
    
    def process_documents(self, documents_dir: str) -> List[DocumentChunk]:
        """
        处理文档目录中的所有文档
        
        Args:
            documents_dir: 文档目录路径
        
        Returns:
            List[DocumentChunk]: 所有文档分块列表
        """
        logger.info(f"开始处理文档目录: {documents_dir}")
        
        # 扫描文档
        md_files = self.scan_documents(documents_dir)
        
        if not md_files:
            logger.warning("没有找到 Markdown 文档")
            return []
        
        all_chunks = []
        processed_files = 0
        
        for file_path in md_files:
            try:
                # 读取文档
                content = self.read_document(file_path)
                if content is None:
                    continue
                
                # 分块处理
                chunks = self.create_chunks(content, file_path)
                all_chunks.extend(chunks)
                processed_files += 1
                
                logger.debug(f"处理完成: {Path(file_path).name} ({len(chunks)} 个分块)")
                
            except Exception as e:
                logger.error(f"处理文档失败 {file_path}: {e}")
                continue
        
        logger.info(f"文档处理完成: {processed_files}/{len(md_files)} 个文件, 共 {len(all_chunks)} 个分块")
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        获取分块统计信息
        
        Args:
            chunks: 文档分块列表
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.content) for chunk in chunks]
        file_counts = {}
        
        for chunk in chunks:
            file_name = chunk.metadata['file_name']
            file_counts[file_name] = file_counts.get(file_name, 0) + 1
        
        stats = {
            'total_chunks': len(chunks),
            'total_files': len(file_counts),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'file_chunk_counts': file_counts
        }
        
        return stats


if __name__ == "__main__":
    # 测试文档处理器
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=30)
    
    # 创建测试文档目录
    test_dir = "./test_docs"
    Path(test_dir).mkdir(exist_ok=True)
    
    # 创建测试文档
    test_content = """
# 测试文档

这是一个测试文档。它包含多个段落和句子。

## 第一节

这是第一节的内容。我们在这里测试文档分块功能。每个分块应该包含适当数量的文本。

## 第二节

这是第二节的内容。分块之间应该有适当的重叠。这样可以保证上下文的连续性。

测试结束。
"""
    
    with open(f"{test_dir}/test.md", 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    # 处理文档
    chunks = processor.process_documents(test_dir)
    
    # 显示结果
    print(f"\n处理结果: {len(chunks)} 个分块")
    for i, chunk in enumerate(chunks):
        print(f"\n分块 {i+1}:")
        print(f"ID: {chunk.chunk_id}")
        print(f"长度: {len(chunk.content)}")
        print(f"内容: {chunk.content[:100]}...")
    
    # 显示统计信息
    stats = processor.get_chunk_stats(chunks)
    print(f"\n统计信息: {stats}")
    
    # 清理测试文件
    import shutil
    shutil.rmtree(test_dir)
    print("\n测试完成")