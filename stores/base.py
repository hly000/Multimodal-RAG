from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

class BaseVectorStore(ABC):
    """
    所有向量存储实现的抽象基类。
    """
    @abstractmethod
    def add(self, vectors: List[np.ndarray], metadata: List[Dict[str, Any]], **kwargs):
        """
        向向量存储中添加向量和对应的元数据。
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        vector: np.ndarray, 
        top_k: int, 
        output_fields: Optional[List[str]] = None, 
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        在向量存储中执行相似度搜索。
        """
        pass
    
    @abstractmethod
    def delete_collection(self):
        """
        删除整个集合/索引，用于清理。
        """
        pass
        
    def build_index(self):
        """
        为已添加的向量构建索引。对于某些数据库可能是空操作。
        """
        pass
        
    def release(self):
        """
        将集合/索引从内存中释放。对于某些数据库可能是空操作。
        """
        pass 