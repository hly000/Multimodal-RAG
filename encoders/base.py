from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class BaseEncoder(ABC):
    """
    所有编码器实现的抽象基类。
    """
    @abstractmethod
    def encode(self, image: Optional[str] = None, text: Optional[str] = None) -> np.ndarray:
        """
        将图像或文本编码为向量。
        :param image: 图像的路径或URL。
        :param text: 要编码的文本。
        :return: 表示图像/文本的Numpy向量。
        :raises ValueError: 如果图像和文本都未提供。
        """
        pass 