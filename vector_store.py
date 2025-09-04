#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量存储模块
负责文档向量化和 FAISS 索引管理
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from loguru import logger

from document_processor import DocumentChunk


class VectorStore:
    """向量存储器"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh",
        device: str = "cpu",
        batch_size: int = 32,
        index_dir: str = "./index"
    ):
        """
        初始化向量存储器
        
        Args:
            model_name: 向量化模型名称
            device: 计算设备
            batch_size: 批处理大小
            index_dir: 索引目录
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化向量化模型
        self.embedding_model = None
        self.dimension = None
        
        # FAISS 索引
        self.index = None
        self.chunks_metadata = []  # 存储分块元数据
        
        logger.info(f"向量存储器初始化 - 模型: {model_name}, 设备: {device}")
    
    def _load_embedding_model(self) -> None:
        """
        加载向量化模型
        """
        if self.embedding_model is None:
            try:
                logger.info(f"正在加载向量化模型: {self.model_name}")
                self.embedding_model = SentenceTransformer(
                    self.model_name,
                    device=self.device
                )
                
                # 获取向量维度
                test_embedding = self.embedding_model.encode(["测试文本"])
                self.dimension = test_embedding.shape[1]
                
                logger.info(f"向量化模型加载成功 - 维度: {self.dimension}")
                
            except Exception as e:
                logger.error(f"加载向量化模型失败: {e}")
                raise
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        将文本转换为向量
        
        Args:
            texts: 文本列表
        
        Returns:
            np.ndarray: 向量数组
        """
        if not texts:
            return np.array([])
        
        # 确保模型已加载
        self._load_embedding_model()
        
        try:
            # 批量编码
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=len(texts) > 10,
                convert_to_numpy=True,
                normalize_embeddings=True  # 归一化向量，便于计算余弦相似度
            )
            
            logger.debug(f"成功编码 {len(texts)} 个文本，向量维度: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            raise
    
    def _create_faiss_index(self, dimension: int) -> faiss.Index:
        """
        创建 FAISS 索引
        
        Args:
            dimension: 向量维度
        
        Returns:
            faiss.Index: FAISS 索引对象
        """
        # 使用 IndexFlatIP (内积索引) 进行余弦相似度搜索
        # 由于向量已归一化，内积等于余弦相似度
        index = faiss.IndexFlatIP(dimension)
        
        logger.info(f"创建 FAISS 索引 - 类型: IndexFlatIP, 维度: {dimension}")
        return index
    
    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """
        构建向量索引
        
        Args:
            chunks: 文档分块列表
        """
        if not chunks:
            logger.warning("没有文档分块，无法构建索引")
            return
        
        logger.info(f"开始构建向量索引 - {len(chunks)} 个分块")
        
        # 提取文本内容
        texts = [chunk.content for chunk in chunks]
        
        # 向量化
        embeddings = self.encode_texts(texts)
        
        if embeddings.size == 0:
            logger.error("向量化失败，无法构建索引")
            return
        
        # 创建 FAISS 索引
        self.index = self._create_faiss_index(embeddings.shape[1])
        
        # 添加向量到索引
        self.index.add(embeddings.astype(np.float32))
        
        # 保存分块元数据
        self.chunks_metadata = [
            {
                'chunk_id': chunk.chunk_id,
                'content': chunk.content,
                'metadata': chunk.metadata,
                'file_path': chunk.file_path,
                'chunk_index': chunk.chunk_index
            }
            for chunk in chunks
        ]
        
        logger.info(f"向量索引构建完成 - 索引大小: {self.index.ntotal}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        搜索相似文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            threshold: 相似度阈值
        
        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("索引为空，无法搜索")
            return []
        
        if not query.strip():
            logger.warning("查询文本为空")
            return []
        
        try:
            # 向量化查询文本
            query_embedding = self.encode_texts([query])
            
            if query_embedding.size == 0:
                logger.error("查询文本向量化失败")
                return []
            
            # 搜索
            scores, indices = self.index.search(
                query_embedding.astype(np.float32),
                min(top_k, self.index.ntotal)
            )
            
            # 处理搜索结果
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                # 检查相似度阈值
                if score < threshold:
                    continue
                
                # 获取分块元数据
                if 0 <= idx < len(self.chunks_metadata):
                    chunk_meta = self.chunks_metadata[idx]
                    result = {
                        'chunk_id': chunk_meta['chunk_id'],
                        'content': chunk_meta['content'],
                        'metadata': chunk_meta['metadata'],
                        'score': float(score),
                        'rank': i + 1
                    }
                    results.append(result)
            
            logger.debug(f"搜索完成 - 查询: '{query[:50]}...', 结果: {len(results)} 个")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def save_index(self, index_name: str = "faiss_index") -> bool:
        """
        保存索引到磁盘
        
        Args:
            index_name: 索引文件名
        
        Returns:
            bool: 保存是否成功
        """
        if self.index is None:
            logger.warning("没有索引可保存")
            return False
        
        try:
            # 保存 FAISS 索引
            index_path = self.index_dir / f"{index_name}.faiss"
            faiss.write_index(self.index, str(index_path))
            
            # 保存元数据
            metadata_path = self.index_dir / f"{index_name}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'chunks_metadata': self.chunks_metadata,
                    'model_name': self.model_name,
                    'dimension': self.dimension
                }, f)
            
            logger.info(f"索引保存成功 - 路径: {index_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            return False
    
    def load_index(self, index_name: str = "faiss_index") -> bool:
        """
        从磁盘加载索引
        
        Args:
            index_name: 索引文件名
        
        Returns:
            bool: 加载是否成功
        """
        try:
            # 检查文件是否存在
            index_path = self.index_dir / f"{index_name}.faiss"
            metadata_path = self.index_dir / f"{index_name}_metadata.pkl"
            
            if not index_path.exists() or not metadata_path.exists():
                logger.warning(f"索引文件不存在: {index_path}")
                return False
            
            # 加载 FAISS 索引
            self.index = faiss.read_index(str(index_path))
            
            # 加载元数据
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks_metadata = data['chunks_metadata']
                saved_model_name = data.get('model_name', self.model_name)
                self.dimension = data.get('dimension')
            
            # 检查模型一致性
            if saved_model_name != self.model_name:
                logger.warning(f"模型不匹配 - 保存的: {saved_model_name}, 当前: {self.model_name}")
            
            logger.info(f"索引加载成功 - 大小: {self.index.ntotal}, 维度: {self.dimension}")
            return True
            
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return False
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        获取索引信息
        
        Returns:
            Dict[str, Any]: 索引信息
        """
        if self.index is None:
            return {'status': 'empty'}
        
        return {
            'status': 'ready',
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'model_name': self.model_name,
            'chunks_count': len(self.chunks_metadata),
            'index_type': type(self.index).__name__
        }
    
    def clear_index(self) -> None:
        """
        清空索引
        """
        self.index = None
        self.chunks_metadata = []
        logger.info("索引已清空")
    
    def update_chunks(self, new_chunks: List[DocumentChunk]) -> None:
        """
        更新文档分块（重建索引）
        
        Args:
            new_chunks: 新的文档分块列表
        """
        logger.info("更新文档分块，重建索引")
        self.clear_index()
        self.build_index(new_chunks)


if __name__ == "__main__":
    # 测试向量存储器
    from document_processor import DocumentProcessor, DocumentChunk
    
    # 创建测试数据
    test_chunks = [
        DocumentChunk(
            content="这是第一个测试文档分块。它包含一些中文文本。",
            metadata={'file_name': 'test1.md', 'chunk_index': 0},
            chunk_id="test1_0_abc123",
            file_path="./test1.md",
            chunk_index=0
        ),
        DocumentChunk(
            content="这是第二个测试文档分块。它讨论了不同的主题。",
            metadata={'file_name': 'test2.md', 'chunk_index': 0},
            chunk_id="test2_0_def456",
            file_path="./test2.md",
            chunk_index=0
        ),
        DocumentChunk(
            content="第三个分块包含技术相关的内容和信息。",
            metadata={'file_name': 'test3.md', 'chunk_index': 0},
            chunk_id="test3_0_ghi789",
            file_path="./test3.md",
            chunk_index=0
        )
    ]
    
    # 创建向量存储器
    vector_store = VectorStore(
        model_name="BAAI/bge-small-zh",
        device="cpu",
        index_dir="./test_index"
    )
    
    try:
        # 构建索引
        print("构建索引...")
        vector_store.build_index(test_chunks)
        
        # 显示索引信息
        info = vector_store.get_index_info()
        print(f"索引信息: {info}")
        
        # 测试搜索
        print("\n测试搜索...")
        queries = ["测试文档", "技术内容", "不相关的查询"]
        
        for query in queries:
            results = vector_store.search(query, top_k=3, threshold=0.3)
            print(f"\n查询: '{query}'")
            print(f"结果数量: {len(results)}")
            
            for result in results:
                print(f"  - 分数: {result['score']:.3f}")
                print(f"    内容: {result['content'][:50]}...")
        
        # 测试保存和加载
        print("\n测试保存索引...")
        if vector_store.save_index("test_index"):
            print("索引保存成功")
            
            # 清空索引
            vector_store.clear_index()
            print("索引已清空")
            
            # 重新加载
            if vector_store.load_index("test_index"):
                print("索引加载成功")
                
                # 再次测试搜索
                results = vector_store.search("测试文档", top_k=2)
                print(f"重新加载后搜索结果: {len(results)} 个")
        
        print("\n测试完成")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试文件
        import shutil
        test_index_dir = Path("./test_index")
        if test_index_dir.exists():
            shutil.rmtree(test_index_dir)
            print("清理测试文件完成")