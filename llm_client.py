#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM客户端模块
负责与DeepSeek API通信和流式响应处理
"""

import json
import time
from typing import Dict, Any, Optional, Iterator, List
from dataclasses import dataclass
from loguru import logger

try:
    from openai import OpenAI
except ImportError:
    logger.error("请安装openai库: pip install openai")
    raise


@dataclass
class ChatMessage:
    """聊天消息数据类"""
    role: str  # 'system', 'user', 'assistant'
    content: str
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典格式"""
        return {
            'role': self.role,
            'content': self.content
        }


@dataclass
class LLMResponse:
    """LLM响应数据类"""
    content: str
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    response_time: Optional[float] = None
    is_stream: bool = False


class DeepSeekClient:
    """DeepSeek API客户端"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        timeout: int = 30
    ):
        """
        初始化DeepSeek客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            max_tokens: 最大token数
            temperature: 温度参数
            timeout: 超时时间
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
        
        # 系统提示词
        self.system_prompt = self._get_system_prompt()
        
        logger.info(f"DeepSeek客户端初始化完成 - 模型: {model}, 温度: {temperature}")
    
    def _get_system_prompt(self) -> str:
        """
        获取系统提示词
        
        Returns:
            str: 系统提示词
        """
        return """
你是一个专业的RAG（检索增强生成）助手。你的任务是基于提供的文档内容回答用户问题。

请遵循以下原则：
1. 优先使用提供的文档内容来回答问题
2. 如果文档中没有相关信息，请明确说明
3. 保持回答的准确性和相关性
4. 使用中文进行回答
5. 回答要简洁明了，重点突出
6. 如果需要，可以结合多个文档片段的信息
7. 对于技术问题，提供具体的解决方案或建议

回答格式：
- 直接回答用户问题
- 如果引用了文档内容，可以适当说明来源
- 如果文档信息不足，诚实说明并提供可能的建议
"""
    
    def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            stream: 是否使用流式响应
            **kwargs: 其他参数
        
        Returns:
            LLMResponse: 响应结果
        """
        start_time = time.time()
        
        try:
            # 构建消息列表
            api_messages = self._build_messages(messages)
            
            # 合并参数
            params = {
                'model': self.model,
                'messages': api_messages,
                'max_tokens': self.max_tokens,
                'temperature': self.temperature,
                'stream': stream,
                **kwargs
            }
            
            logger.debug(f"发送请求到DeepSeek - 消息数: {len(api_messages)}, 流式: {stream}")
            
            if stream:
                return self._handle_stream_response(params, start_time)
            else:
                return self._handle_normal_response(params, start_time)
                
        except Exception as e:
            logger.error(f"DeepSeek API请求失败: {e}")
            response_time = time.time() - start_time
            return LLMResponse(
                content=f"抱歉，AI服务暂时不可用。错误信息: {str(e)}",
                response_time=response_time,
                is_stream=stream
            )
    
    def _build_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        """
        构建API消息格式
        
        Args:
            messages: 消息列表
        
        Returns:
            List[Dict[str, str]]: API格式的消息列表
        """
        api_messages = []
        
        # 添加系统消息
        api_messages.append({
            'role': 'system',
            'content': self.system_prompt
        })
        
        # 添加用户消息
        for msg in messages:
            api_messages.append(msg.to_dict())
        
        return api_messages
    
    def _handle_normal_response(
        self,
        params: Dict[str, Any],
        start_time: float
    ) -> LLMResponse:
        """
        处理普通响应
        
        Args:
            params: 请求参数
            start_time: 开始时间
        
        Returns:
            LLMResponse: 响应结果
        """
        response = self.client.chat.completions.create(**params)
        response_time = time.time() - start_time
        
        # 提取响应内容
        content = response.choices[0].message.content
        usage = response.usage.model_dump() if response.usage else None
        finish_reason = response.choices[0].finish_reason
        
        logger.info(f"DeepSeek响应完成 - 耗时: {response_time:.3f}s, Token使用: {usage}")
        
        return LLMResponse(
            content=content,
            usage=usage,
            model=response.model,
            finish_reason=finish_reason,
            response_time=response_time,
            is_stream=False
        )
    
    def _handle_stream_response(
        self,
        params: Dict[str, Any],
        start_time: float
    ) -> LLMResponse:
        """
        处理流式响应
        
        Args:
            params: 请求参数
            start_time: 开始时间
        
        Returns:
            LLMResponse: 响应结果（包含生成器）
        """
        try:
            stream = self.client.chat.completions.create(**params)
            
            def response_generator():
                """响应生成器"""
                full_content = ""
                
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_content += content
                        yield content
                
                # 记录完成信息
                response_time = time.time() - start_time
                logger.info(f"DeepSeek流式响应完成 - 耗时: {response_time:.3f}s, 内容长度: {len(full_content)}")
            
            return LLMResponse(
                content=response_generator(),
                response_time=None,  # 流式响应时间在生成器中计算
                is_stream=True
            )
            
        except Exception as e:
            logger.error(f"流式响应处理失败: {e}")
            response_time = time.time() - start_time
            return LLMResponse(
                content=f"抱歉，流式响应失败。错误信息: {str(e)}",
                response_time=response_time,
                is_stream=True
            )
    
    def chat_with_context(
        self,
        user_query: str,
        context: str,
        conversation_history: Optional[List[ChatMessage]] = None,
        stream: bool = False
    ) -> LLMResponse:
        """
        基于上下文进行对话
        
        Args:
            user_query: 用户查询
            context: 检索到的上下文
            conversation_history: 对话历史
            stream: 是否使用流式响应
        
        Returns:
            LLMResponse: 响应结果
        """
        # 构建带上下文的用户消息
        contextual_message = self._build_contextual_message(user_query, context)
        
        # 构建消息列表
        messages = []
        
        # 添加对话历史（如果有）
        if conversation_history:
            # 只保留最近的几轮对话，避免上下文过长
            recent_history = conversation_history[-6:]  # 最多保留3轮对话
            messages.extend(recent_history)
        
        # 添加当前查询
        messages.append(ChatMessage(role='user', content=contextual_message))
        
        logger.debug(f"基于上下文对话 - 查询: '{user_query[:50]}...', 上下文长度: {len(context)}")
        
        return self.chat(messages, stream=stream)
    
    def _build_contextual_message(self, user_query: str, context: str) -> str:
        """
        构建带上下文的用户消息
        
        Args:
            user_query: 用户查询
            context: 上下文
        
        Returns:
            str: 带上下文的消息
        """
        if not context.strip():
            return f"用户问题: {user_query}\n\n注意: 没有找到相关的文档内容，请基于你的知识回答。"
        
        return f"""{context}

基于以上文档内容，请回答用户的问题:
{user_query}

请确保回答准确、相关，并优先使用文档中的信息。如果文档信息不足，请说明并提供可能的建议。"""
    
    def test_connection(self) -> bool:
        """
        测试API连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            test_message = ChatMessage(role='user', content='你好')
            response = self.chat([test_message])
            
            if response.content and '抱歉' not in response.content:
                logger.info("DeepSeek API连接测试成功")
                return True
            else:
                logger.warning("DeepSeek API连接测试失败")
                return False
                
        except Exception as e:
            logger.error(f"DeepSeek API连接测试异常: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            'model': self.model,
            'base_url': self.base_url,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'timeout': self.timeout
        }


class ConversationManager:
    """对话管理器"""
    
    def __init__(self, max_history: int = 10):
        """
        初始化对话管理器
        
        Args:
            max_history: 最大历史记录数
        """
        self.max_history = max_history
        self.conversation_history: List[ChatMessage] = []
        
        logger.info(f"对话管理器初始化完成 - 最大历史: {max_history}")
    
    def add_message(self, role: str, content: str) -> None:
        """
        添加消息到历史记录
        
        Args:
            role: 角色
            content: 内容
        """
        message = ChatMessage(role=role, content=content)
        self.conversation_history.append(message)
        
        # 保持历史记录在限制范围内
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        logger.debug(f"添加消息到历史 - 角色: {role}, 当前历史数: {len(self.conversation_history)}")
    
    def get_history(self) -> List[ChatMessage]:
        """
        获取对话历史
        
        Returns:
            List[ChatMessage]: 对话历史
        """
        return self.conversation_history.copy()
    
    def clear_history(self) -> None:
        """
        清空对话历史
        """
        self.conversation_history.clear()
        logger.info("对话历史已清空")
    
    def get_recent_context(self, max_messages: int = 6) -> List[ChatMessage]:
        """
        获取最近的对话上下文
        
        Args:
            max_messages: 最大消息数
        
        Returns:
            List[ChatMessage]: 最近的对话上下文
        """
        return self.conversation_history[-max_messages:] if self.conversation_history else []
    
    def export_history(self) -> List[Dict[str, Any]]:
        """
        导出对话历史
        
        Returns:
            List[Dict[str, Any]]: 导出的历史数据
        """
        return [
            {
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp
            }
            for msg in self.conversation_history
        ]
    
    def import_history(self, history_data: List[Dict[str, Any]]) -> None:
        """
        导入对话历史
        
        Args:
            history_data: 历史数据
        """
        self.conversation_history.clear()
        
        for item in history_data:
            message = ChatMessage(
                role=item['role'],
                content=item['content'],
                timestamp=item.get('timestamp')
            )
            self.conversation_history.append(message)
        
        logger.info(f"导入对话历史完成 - 消息数: {len(self.conversation_history)}")


if __name__ == "__main__":
    # 测试LLM客户端
    import os
    from dotenv import load_dotenv
    
    # 加载环境变量
    load_dotenv()
    
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("请设置DEEPSEEK_API_KEY环境变量")
        exit(1)
    
    try:
        # 创建客户端
        client = DeepSeekClient(
            api_key=api_key,
            model="deepseek-chat",
            temperature=0.7
        )
        
        # 测试连接
        print("测试API连接...")
        if client.test_connection():
            print("✓ API连接成功")
        else:
            print("✗ API连接失败")
            exit(1)
        
        # 创建对话管理器
        conversation = ConversationManager()
        
        # 测试普通对话
        print("\n测试普通对话:")
        user_message = ChatMessage(role='user', content='你好，请介绍一下你自己')
        response = client.chat([user_message])
        print(f"响应: {response.content}")
        print(f"耗时: {response.response_time:.3f}s")
        
        # 添加到对话历史
        conversation.add_message('user', '你好，请介绍一下你自己')
        conversation.add_message('assistant', response.content)
        
        # 测试带上下文的对话
        print("\n测试带上下文的对话:")
        context = """
        用户查询: Python编程语言
        
        相关文档:
        
        [文档 1] 来源: python.md (相似度: 0.892)
        Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。它广泛应用于Web开发、数据科学、人工智能和自动化等领域。Python的设计哲学强调代码的可读性，使用缩进来定义代码块。
        
        [文档 2] 来源: programming.md (相似度: 0.756)
        Python支持多种编程范式，包括面向对象、函数式和过程式编程。它拥有丰富的标准库和第三方包生态系统，如NumPy、Pandas、Django等，这些工具大大扩展了Python的功能。
        """
        
        user_query = "Python有什么特点和优势？"
        response = client.chat_with_context(
            user_query=user_query,
            context=context,
            conversation_history=conversation.get_recent_context()
        )
        
        print(f"查询: {user_query}")
        print(f"响应: {response.content}")
        print(f"耗时: {response.response_time:.3f}s")
        
        # 测试流式响应
        print("\n测试流式响应:")
        stream_response = client.chat_with_context(
            user_query="请详细解释Python的应用领域",
            context=context,
            stream=True
        )
        
        if stream_response.is_stream:
            print("流式响应: ", end="", flush=True)
            for chunk in stream_response.content:
                print(chunk, end="", flush=True)
            print()  # 换行
        
        # 显示模型信息
        model_info = client.get_model_info()
        print(f"\n模型信息: {model_info}")
        
        # 显示对话历史
        history = conversation.export_history()
        print(f"\n对话历史: {len(history)} 条消息")
        
        print("\nLLM客户端测试完成")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()