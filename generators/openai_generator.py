import os
from openai import OpenAI, AzureOpenAI
from .base import BaseGenerator

class OpenAIGenerator(BaseGenerator):
    """
    使用 OpenAI 或 Azure OpenAI API 的生成器实现。
    """
    def __init__(self, model: str, api_type: str = 'openai', api_key: str = None, base_url: str = None, endpoint: str = None, api_version: str = None):
        if api_type == 'azure':
            if not all([endpoint, api_version]):
                raise ValueError("对于 Azure OpenAI，必须提供 endpoint 和 api_version。")
            self.client = AzureOpenAI(
                api_key=api_key or os.environ.get("AZURE_OPENAI_KEY"),
                azure_endpoint=endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
                api_version=api_version
            )
        elif api_type == 'custom':
            if not base_url or not api_key:
                raise ValueError("必须为自定义API类型提供 Base URL 和 API Key。")
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        else: # 默认为 'openai'
            self.client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            )
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        """
        调用 Chat Completions API 生成文本。
        """
        # 移除不支持的自定义参数
        kwargs.pop('api_type', None)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个智能的多模态助手。请根据用户提供的上下文信息来回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_message = f"调用大模型 API 时出错: {e}"
            print(error_message)
            return error_message 