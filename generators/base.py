from abc import ABC, abstractmethod

class BaseGenerator(ABC):
    """
    生成器模型（LLM）的抽象基类。
    """
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        根据给定的提示生成文本。
        :param prompt: 输入给模型的完整提示。
        :param kwargs: 其他特定于模型的参数，如 temperature, max_tokens 等。
        :return: LLM 生成的文本字符串。
        """
        pass 