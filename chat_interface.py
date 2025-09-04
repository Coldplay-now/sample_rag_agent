#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行交互界面
提供中文交互、流式响应和用户友好的聊天体验
"""

import os
import sys
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from loguru import logger

try:
    import readline  # 提供命令行编辑功能
except ImportError:
    readline = None

from config import Config, load_config
from document_processor import DocumentProcessor
from vector_store import VectorStore
from retriever import Retriever
from llm_client import DeepSeekClient, ConversationManager, ChatMessage


@dataclass
class ChatStats:
    """聊天统计信息"""
    total_queries: int = 0
    total_response_time: float = 0.0
    documents_processed: int = 0
    index_built: bool = False
    session_start_time: float = 0.0
    
    def add_query(self, response_time: float):
        """添加查询统计"""
        self.total_queries += 1
        self.total_response_time += response_time
    
    def get_average_response_time(self) -> float:
        """获取平均响应时间"""
        return self.total_response_time / self.total_queries if self.total_queries > 0 else 0.0
    
    def get_session_duration(self) -> float:
        """获取会话持续时间"""
        return time.time() - self.session_start_time if self.session_start_time > 0 else 0.0


class ChatInterface:
    """命令行聊天界面"""
    
    def __init__(self, config: Config):
        """
        初始化聊天界面
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.stats = ChatStats()
        self.stats.session_start_time = time.time()
        
        # 初始化组件
        self.document_processor: Optional[DocumentProcessor] = None
        self.vector_store: Optional[VectorStore] = None
        self.retriever: Optional[Retriever] = None
        self.llm_client: Optional[DeepSeekClient] = None
        self.conversation: Optional[ConversationManager] = None
        
        # 界面状态
        self.is_running = False
        self.index_ready = False
        
        # 命令映射
        self.commands = {
            '/help': self._show_help,
            '/帮助': self._show_help,
            '/quit': self._quit,
            '/退出': self._quit,
            '/exit': self._quit,
            '/clear': self._clear_history,
            '/清空': self._clear_history,
            '/stats': self._show_stats,
            '/统计': self._show_stats,
            '/rebuild': self._rebuild_index,
            '/重建': self._rebuild_index,
            '/info': self._show_info,
            '/信息': self._show_info,
            '/config': self._show_config,
            '/配置': self._show_config
        }
        
        logger.info("聊天界面初始化完成")
    
    def initialize_components(self) -> bool:
        """
        初始化所有组件
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            print("🚀 正在初始化RAG Agent...")
            
            # 初始化LLM客户端
            print("📡 连接DeepSeek API...")
            self.llm_client = DeepSeekClient(
                api_key=self.config.deepseek_api_key,
                base_url=self.config.deepseek_base_url,
                model=self.config.deepseek_model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # 测试API连接
            if not self.llm_client.test_connection():
                print("❌ DeepSeek API连接失败，请检查配置")
                return False
            print("✅ DeepSeek API连接成功")
            
            # 初始化对话管理器
            self.conversation = ConversationManager(max_history=self.config.max_history)
            
            # 初始化文档处理器
            print("📚 初始化文档处理器...")
            self.document_processor = DocumentProcessor(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            
            # 初始化向量存储
            print("🔍 初始化向量存储...")
            self.vector_store = VectorStore(
                model_name=self.config.embedding_model,
                index_dir=self.config.index_dir,
                cache_dir=self.config.cache_dir
            )
            
            # 检查是否需要构建索引
            if not self.vector_store.load_index():
                print("📖 检测到新的文档目录，开始构建索引...")
                if not self._build_index():
                    print("❌ 索引构建失败")
                    return False
            else:
                print("✅ 索引加载成功")
                self.index_ready = True
            
            # 初始化检索器
            print("🎯 初始化检索器...")
            self.retriever = Retriever(
                vector_store=self.vector_store,
                top_k=self.config.top_k,
                similarity_threshold=self.config.similarity_threshold,
                enable_query_expansion=self.config.enable_query_expansion,
                enable_rerank=self.config.enable_rerank
            )
            
            print("🎉 RAG Agent初始化完成！")
            return True
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            print(f"❌ 初始化失败: {e}")
            return False
    
    def _build_index(self) -> bool:
        """
        构建文档索引
        
        Returns:
            bool: 构建是否成功
        """
        try:
            # 处理文档
            print(f"📁 扫描文档目录: {self.config.documents_dir}")
            chunks = self.document_processor.process_directory(self.config.documents_dir)
            
            if not chunks:
                print("⚠️  未找到任何文档，请检查文档目录")
                return False
            
            self.stats.documents_processed = len(chunks)
            print(f"📄 找到 {len(chunks)} 个文档分块")
            
            # 构建向量索引
            print("🔨 构建向量索引...")
            self.vector_store.build_index(chunks)
            
            print("💾 保存索引...")
            self.vector_store.save_index()
            
            self.index_ready = True
            self.stats.index_built = True
            print("✅ 索引构建完成")
            
            return True
            
        except Exception as e:
            logger.error(f"索引构建失败: {e}")
            print(f"❌ 索引构建失败: {e}")
            return False
    
    def run(self) -> None:
        """
        运行聊天界面
        """
        # 显示欢迎信息
        self._show_welcome()
        
        # 初始化组件
        if not self.initialize_components():
            print("\n❌ 初始化失败，程序退出")
            return
        
        # 显示使用提示
        self._show_usage_tips()
        
        # 开始聊天循环
        self.is_running = True
        
        try:
            while self.is_running:
                self._chat_loop()
        except KeyboardInterrupt:
            print("\n\n👋 检测到Ctrl+C，正在退出...")
        except Exception as e:
            logger.error(f"聊天循环异常: {e}")
            print(f"\n❌ 发生错误: {e}")
        finally:
            self._cleanup()
    
    def _chat_loop(self) -> None:
        """
        聊天循环
        """
        try:
            # 获取用户输入
            user_input = input("\n💬 您: ").strip()
            
            if not user_input:
                return
            
            # 检查是否是命令
            if user_input.startswith('/'):
                self._handle_command(user_input)
                return
            
            # 检查索引是否就绪
            if not self.index_ready:
                print("⚠️  索引未就绪，请先构建索引或检查文档目录")
                return
            
            # 处理用户查询
            self._handle_user_query(user_input)
            
        except EOFError:
            print("\n\n👋 检测到EOF，正在退出...")
            self.is_running = False
        except Exception as e:
            logger.error(f"聊天循环处理异常: {e}")
            print(f"❌ 处理查询时发生错误: {e}")
    
    def _handle_user_query(self, query: str) -> None:
        """
        处理用户查询
        
        Args:
            query: 用户查询
        """
        start_time = time.time()
        
        try:
            # 检索相关文档
            print("🔍 正在检索相关文档...")
            retrieval_result = self.retriever.retrieve(query)
            
            if not retrieval_result.chunks:
                print("⚠️  未找到相关文档，将基于通用知识回答")
                context = ""
            else:
                context = retrieval_result.context
                print(f"📚 找到 {len(retrieval_result.chunks)} 个相关文档片段")
            
            # 生成回答
            print("🤖 AI正在思考...\n")
            
            # 使用流式响应
            response = self.llm_client.chat_with_context(
                user_query=query,
                context=context,
                conversation_history=self.conversation.get_recent_context(),
                stream=self.config.enable_streaming
            )
            
            # 显示回答
            print("🤖 AI: ", end="", flush=True)
            
            if response.is_stream:
                # 流式显示
                full_response = ""
                for chunk in response.content:
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print()  # 换行
                
                # 记录完整回答
                response.content = full_response
            else:
                # 一次性显示
                print(response.content)
            
            # 更新对话历史
            self.conversation.add_message('user', query)
            self.conversation.add_message('assistant', response.content)
            
            # 更新统计
            response_time = time.time() - start_time
            self.stats.add_query(response_time)
            
            # 显示检索信息（调试模式）
            if self.config.debug_mode and retrieval_result.chunks:
                print(f"\n📊 检索信息: 耗时 {retrieval_result.retrieval_time:.3f}s, "
                      f"响应耗时 {response_time:.3f}s")
            
        except Exception as e:
            logger.error(f"查询处理失败: {e}")
            print(f"❌ 查询处理失败: {e}")
    
    def _handle_command(self, command: str) -> None:
        """
        处理命令
        
        Args:
            command: 命令字符串
        """
        cmd = command.lower().strip()
        
        if cmd in self.commands:
            self.commands[cmd]()
        else:
            print(f"❌ 未知命令: {command}")
            print("💡 输入 /help 或 /帮助 查看可用命令")
    
    def _show_welcome(self) -> None:
        """
        显示欢迎信息
        """
        print("\n" + "="*60)
        print("🎉 欢迎使用 RAG Agent - 智能文档问答助手")
        print("="*60)
        print("📖 基于您的文档内容提供智能问答服务")
        print("🤖 支持中文交互，流式响应，上下文理解")
        print("💡 输入 /help 或 /帮助 查看使用说明")
        print("="*60)
    
    def _show_usage_tips(self) -> None:
        """
        显示使用提示
        """
        print("\n💡 使用提示:")
        print("  • 直接输入问题开始对话")
        print("  • 输入 /help 查看所有命令")
        print("  • 输入 /quit 或 /退出 结束对话")
        print("  • 支持多轮对话和上下文理解")
        print("  • Ctrl+C 快速退出")
    
    def _show_help(self) -> None:
        """
        显示帮助信息
        """
        print("\n📖 可用命令:")
        print("  /help, /帮助     - 显示此帮助信息")
        print("  /quit, /退出     - 退出程序")
        print("  /clear, /清空    - 清空对话历史")
        print("  /stats, /统计    - 显示统计信息")
        print("  /rebuild, /重建  - 重建文档索引")
        print("  /info, /信息     - 显示系统信息")
        print("  /config, /配置   - 显示配置信息")
        print("\n💬 直接输入问题开始对话")
    
    def _quit(self) -> None:
        """
        退出程序
        """
        print("\n👋 感谢使用RAG Agent，再见！")
        self.is_running = False
    
    def _clear_history(self) -> None:
        """
        清空对话历史
        """
        if self.conversation:
            self.conversation.clear_history()
            print("✅ 对话历史已清空")
        else:
            print("⚠️  对话管理器未初始化")
    
    def _show_stats(self) -> None:
        """
        显示统计信息
        """
        print("\n📊 会话统计:")
        print(f"  查询次数: {self.stats.total_queries}")
        print(f"  平均响应时间: {self.stats.get_average_response_time():.3f}s")
        print(f"  会话时长: {self.stats.get_session_duration():.1f}s")
        print(f"  文档分块数: {self.stats.documents_processed}")
        print(f"  索引状态: {'✅ 已构建' if self.stats.index_built else '❌ 未构建'}")
        
        if self.retriever:
            retrieval_stats = self.retriever.get_retrieval_stats()
            print(f"  向量存储: {retrieval_stats.get('total_documents', 0)} 个文档")
            print(f"  检索配置: Top-{retrieval_stats.get('top_k', 0)}, "
                  f"阈值 {retrieval_stats.get('similarity_threshold', 0)}")
    
    def _rebuild_index(self) -> None:
        """
        重建文档索引
        """
        print("\n🔨 开始重建索引...")
        
        if not self.vector_store or not self.document_processor:
            print("❌ 组件未初始化")
            return
        
        # 清空现有索引
        self.vector_store.clear_index()
        self.index_ready = False
        
        # 重新构建
        if self._build_index():
            print("✅ 索引重建完成")
        else:
            print("❌ 索引重建失败")
    
    def _show_info(self) -> None:
        """
        显示系统信息
        """
        print("\n🔧 系统信息:")
        print(f"  文档目录: {self.config.documents_dir}")
        print(f"  索引目录: {self.config.index_dir}")
        print(f"  缓存目录: {self.config.cache_dir}")
        print(f"  嵌入模型: {self.config.embedding_model}")
        print(f"  LLM模型: {self.config.deepseek_model}")
        print(f"  流式响应: {'✅ 启用' if self.config.enable_streaming else '❌ 禁用'}")
        print(f"  查询扩展: {'✅ 启用' if self.config.enable_query_expansion else '❌ 禁用'}")
        print(f"  重排序: {'✅ 启用' if self.config.enable_rerank else '❌ 禁用'}")
    
    def _show_config(self) -> None:
        """
        显示配置信息
        """
        print("\n⚙️  配置信息:")
        print(f"  分块大小: {self.config.chunk_size}")
        print(f"  分块重叠: {self.config.chunk_overlap}")
        print(f"  检索数量: {self.config.top_k}")
        print(f"  相似度阈值: {self.config.similarity_threshold}")
        print(f"  最大Token: {self.config.max_tokens}")
        print(f"  温度参数: {self.config.temperature}")
        print(f"  历史记录: {self.config.max_history}")
        print(f"  调试模式: {'✅ 启用' if self.config.debug_mode else '❌ 禁用'}")
    
    def _cleanup(self) -> None:
        """
        清理资源
        """
        try:
            # 保存索引（如果有更新）
            if self.vector_store and self.index_ready:
                self.vector_store.save_index()
            
            # 显示会话统计
            if self.stats.total_queries > 0:
                print("\n📊 会话总结:")
                print(f"  总查询数: {self.stats.total_queries}")
                print(f"  平均响应时间: {self.stats.get_average_response_time():.3f}s")
                print(f"  会话时长: {self.stats.get_session_duration():.1f}s")
            
            print("\n🧹 资源清理完成")
            
        except Exception as e:
            logger.error(f"清理资源时发生错误: {e}")


def main():
    """主函数"""
    try:
        # 加载配置
        config = load_config()
        
        # 创建并运行聊天界面
        chat = ChatInterface(config)
        chat.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行异常: {e}")
        print(f"❌ 程序运行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()