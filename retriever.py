#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索模块
负责查询处理、上下文构建和检索优化
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from vector_store import VectorStore


@dataclass
class RetrievalResult:
    """检索结果数据类"""
    query: str  # 原始查询
    expanded_query: str  # 扩展后的查询
    chunks: List[Dict[str, Any]]  # 检索到的文档分块
    context: str  # 构建的上下文
    total_length: int  # 上下文总长度
    retrieval_time: float  # 检索耗时


class QueryExpander:
    """查询扩展器"""
    
    def __init__(self):
        """初始化查询扩展器"""
        # 中文同义词映射（简化版）
        self.synonyms = {
            '如何': ['怎么', '怎样', '方法'],
            '什么': ['啥', '何'],
            '为什么': ['为啥', '原因'],
            '问题': ['疑问', '困难', '难题'],
            '解决': ['处理', '解答', '应对'],
            '方法': ['方式', '途径', '办法'],
            '技术': ['技能', '工艺', '科技'],
            '系统': ['体系', '平台', '框架'],
            '功能': ['作用', '特性', '能力'],
            '配置': ['设置', '配备', '安装'],
        }
    
    def expand_query(self, query: str) -> str:
        """
        扩展查询文本
        
        Args:
            query: 原始查询
        
        Returns:
            str: 扩展后的查询
        """
        if not query.strip():
            return query
        
        expanded_terms = [query]  # 包含原始查询
        
        # 添加同义词
        for word, synonyms in self.synonyms.items():
            if word in query:
                for synonym in synonyms:
                    if synonym not in query:
                        expanded_terms.append(query.replace(word, synonym))
        
        # 提取关键词（简单的中文分词）
        keywords = self._extract_keywords(query)
        expanded_terms.extend(keywords)
        
        # 去重并合并
        unique_terms = list(set(expanded_terms))
        expanded_query = ' '.join(unique_terms)
        
        logger.debug(f"查询扩展: '{query}' -> '{expanded_query}'")
        return expanded_query
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        提取关键词（简化版中文分词）
        
        Args:
            text: 输入文本
        
        Returns:
            List[str]: 关键词列表
        """
        # 移除标点符号
        cleaned_text = re.sub(r'[^\w\s]', ' ', text)
        
        # 按空格分割
        words = cleaned_text.split()
        
        # 过滤短词和停用词
        stop_words = {'的', '了', '在', '是', '有', '和', '与', '或', '但', '而', '也', '都', '要', '会', '能', '可以'}
        keywords = [word for word in words if len(word) > 1 and word not in stop_words]
        
        return keywords


class ContextBuilder:
    """上下文构建器"""
    
    def __init__(self, max_context_length: int = 4000):
        """
        初始化上下文构建器
        
        Args:
            max_context_length: 最大上下文长度
        """
        self.max_context_length = max_context_length
    
    def build_context(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        include_metadata: bool = True
    ) -> str:
        """
        构建检索上下文
        
        Args:
            chunks: 检索到的文档分块
            query: 查询文本
            include_metadata: 是否包含元数据
        
        Returns:
            str: 构建的上下文
        """
        if not chunks:
            return ""
        
        context_parts = []
        current_length = 0
        
        # 添加查询信息
        query_info = f"用户查询: {query}\n\n相关文档:\n\n"
        context_parts.append(query_info)
        current_length += len(query_info)
        
        # 按相似度排序（已经排序）并添加文档分块
        for i, chunk in enumerate(chunks):
            # 构建分块信息
            chunk_info = self._format_chunk(
                chunk,
                index=i + 1,
                include_metadata=include_metadata
            )
            
            # 检查长度限制
            if current_length + len(chunk_info) > self.max_context_length:
                logger.debug(f"达到上下文长度限制，截断在第 {i} 个分块")
                break
            
            context_parts.append(chunk_info)
            current_length += len(chunk_info)
        
        context = ''.join(context_parts)
        logger.debug(f"构建上下文完成 - 长度: {len(context)}, 分块数: {len(chunks)}")
        
        return context
    
    def _format_chunk(
        self,
        chunk: Dict[str, Any],
        index: int,
        include_metadata: bool = True
    ) -> str:
        """
        格式化文档分块
        
        Args:
            chunk: 文档分块
            index: 分块序号
            include_metadata: 是否包含元数据
        
        Returns:
            str: 格式化的分块信息
        """
        content = chunk['content']
        score = chunk.get('score', 0.0)
        
        # 基本格式
        formatted = f"[文档 {index}] (相似度: {score:.3f})\n{content}\n\n"
        
        # 添加元数据
        if include_metadata and 'metadata' in chunk:
            metadata = chunk['metadata']
            file_name = metadata.get('file_name', '未知文件')
            formatted = f"[文档 {index}] 来源: {file_name} (相似度: {score:.3f})\n{content}\n\n"
        
        return formatted
    
    def optimize_context_diversity(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        优化上下文多样性（去重相似内容）
        
        Args:
            chunks: 原始分块列表
        
        Returns:
            List[Dict[str, Any]]: 优化后的分块列表
        """
        if len(chunks) <= 1:
            return chunks
        
        optimized_chunks = [chunks[0]]  # 保留第一个（最相似的）
        
        for chunk in chunks[1:]:
            # 检查与已选择分块的相似性
            is_similar = False
            for selected_chunk in optimized_chunks:
                similarity = self._calculate_text_similarity(
                    chunk['content'],
                    selected_chunk['content']
                )
                
                # 如果内容过于相似，跳过
                if similarity > 0.8:
                    is_similar = True
                    break
            
            if not is_similar:
                optimized_chunks.append(chunk)
        
        logger.debug(f"多样性优化: {len(chunks)} -> {len(optimized_chunks)} 个分块")
        return optimized_chunks
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似性（简化版）
        
        Args:
            text1: 文本1
            text2: 文本2
        
        Returns:
            float: 相似度分数 (0-1)
        """
        # 简单的字符级相似度计算
        if not text1 or not text2:
            return 0.0
        
        # 计算公共字符数
        set1 = set(text1)
        set2 = set(text2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0


class Retriever:
    """检索器"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        max_context_length: int = 4000,
        enable_query_expansion: bool = True,
        enable_rerank: bool = True
    ):
        """
        初始化检索器
        
        Args:
            vector_store: 向量存储器
            top_k: 检索数量
            similarity_threshold: 相似度阈值
            max_context_length: 最大上下文长度
            enable_query_expansion: 是否启用查询扩展
            enable_rerank: 是否启用重排序
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.max_context_length = max_context_length
        self.enable_query_expansion = enable_query_expansion
        self.enable_rerank = enable_rerank
        
        # 初始化组件
        self.query_expander = QueryExpander() if enable_query_expansion else None
        self.context_builder = ContextBuilder(max_context_length)
        
        logger.info(f"检索器初始化完成 - Top-K: {top_k}, 阈值: {similarity_threshold}")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> RetrievalResult:
        """
        执行检索
        
        Args:
            query: 查询文本
            top_k: 检索数量（可选）
            threshold: 相似度阈值（可选）
        
        Returns:
            RetrievalResult: 检索结果
        """
        import time
        start_time = time.time()
        
        # 使用传入参数或默认值
        actual_top_k = top_k if top_k is not None else self.top_k
        actual_threshold = threshold if threshold is not None else self.similarity_threshold
        
        logger.info(f"开始检索 - 查询: '{query[:50]}...', Top-K: {actual_top_k}")
        
        # 查询扩展
        expanded_query = query
        if self.enable_query_expansion and self.query_expander:
            expanded_query = self.query_expander.expand_query(query)
        
        # 向量检索
        chunks = self.vector_store.search(
            query=expanded_query,
            top_k=actual_top_k * 2,  # 检索更多候选，用于后续优化
            threshold=actual_threshold
        )
        
        # 重排序和多样性优化
        if self.enable_rerank and chunks:
            chunks = self._rerank_chunks(chunks, query)
        
        # 多样性优化
        chunks = self.context_builder.optimize_context_diversity(chunks)
        
        # 截断到指定数量
        chunks = chunks[:actual_top_k]
        
        # 构建上下文
        context = self.context_builder.build_context(chunks, query)
        
        # 计算耗时
        retrieval_time = time.time() - start_time
        
        result = RetrievalResult(
            query=query,
            expanded_query=expanded_query,
            chunks=chunks,
            context=context,
            total_length=len(context),
            retrieval_time=retrieval_time
        )
        
        logger.info(f"检索完成 - 耗时: {retrieval_time:.3f}s, 结果: {len(chunks)} 个分块")
        return result
    
    def _rerank_chunks(
        self,
        chunks: List[Dict[str, Any]],
        original_query: str
    ) -> List[Dict[str, Any]]:
        """
        重排序文档分块
        
        Args:
            chunks: 原始分块列表
            original_query: 原始查询
        
        Returns:
            List[Dict[str, Any]]: 重排序后的分块列表
        """
        if not chunks:
            return chunks
        
        # 简单的重排序策略：结合相似度和查询词匹配度
        for chunk in chunks:
            content = chunk['content'].lower()
            query_lower = original_query.lower()
            
            # 计算查询词在内容中的匹配度
            query_words = set(query_lower.split())
            content_words = set(content.split())
            
            word_match_score = len(query_words & content_words) / len(query_words) if query_words else 0
            
            # 结合原始相似度和词匹配度
            original_score = chunk.get('score', 0.0)
            combined_score = 0.7 * original_score + 0.3 * word_match_score
            
            chunk['rerank_score'] = combined_score
        
        # 按重排序分数排序
        reranked_chunks = sorted(chunks, key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        logger.debug(f"重排序完成 - {len(chunks)} 个分块")
        return reranked_chunks
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        获取检索统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        vector_info = self.vector_store.get_index_info()
        
        return {
            'vector_store_status': vector_info.get('status', 'unknown'),
            'total_documents': vector_info.get('chunks_count', 0),
            'index_size': vector_info.get('total_vectors', 0),
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold,
            'max_context_length': self.max_context_length,
            'query_expansion_enabled': self.enable_query_expansion,
            'rerank_enabled': self.enable_rerank
        }


if __name__ == "__main__":
    # 测试检索器
    from vector_store import VectorStore
    from document_processor import DocumentChunk
    
    # 创建测试数据
    test_chunks = [
        DocumentChunk(
            content="Python是一种高级编程语言，广泛用于数据科学和机器学习。",
            metadata={'file_name': 'python.md', 'chunk_index': 0},
            chunk_id="python_0_abc",
            file_path="./python.md",
            chunk_index=0
        ),
        DocumentChunk(
            content="机器学习是人工智能的一个分支，使用算法来分析数据。",
            metadata={'file_name': 'ml.md', 'chunk_index': 0},
            chunk_id="ml_0_def",
            file_path="./ml.md",
            chunk_index=0
        ),
        DocumentChunk(
            content="数据科学结合了统计学、编程和领域知识来提取洞察。",
            metadata={'file_name': 'ds.md', 'chunk_index': 0},
            chunk_id="ds_0_ghi",
            file_path="./ds.md",
            chunk_index=0
        )
    ]
    
    try:
        # 创建向量存储器并构建索引
        vector_store = VectorStore(index_dir="./test_retriever_index")
        vector_store.build_index(test_chunks)
        
        # 创建检索器
        retriever = Retriever(
            vector_store=vector_store,
            top_k=3,
            similarity_threshold=0.3,
            enable_query_expansion=True,
            enable_rerank=True
        )
        
        # 测试检索
        test_queries = [
            "什么是Python？",
            "机器学习的应用",
            "数据科学方法",
            "编程语言特点"
        ]
        
        for query in test_queries:
            print(f"\n查询: {query}")
            result = retriever.retrieve(query)
            
            print(f"扩展查询: {result.expanded_query}")
            print(f"检索结果: {len(result.chunks)} 个分块")
            print(f"上下文长度: {result.total_length}")
            print(f"检索耗时: {result.retrieval_time:.3f}s")
            
            for i, chunk in enumerate(result.chunks):
                score = chunk.get('score', 0)
                rerank_score = chunk.get('rerank_score', score)
                print(f"  分块 {i+1}: 相似度={score:.3f}, 重排序={rerank_score:.3f}")
                print(f"    内容: {chunk['content'][:50]}...")
        
        # 显示统计信息
        stats = retriever.get_retrieval_stats()
        print(f"\n检索统计: {stats}")
        
        print("\n检索器测试完成")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试文件
        import shutil
        from pathlib import Path
        test_dir = Path("./test_retriever_index")
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print("清理测试文件完成")