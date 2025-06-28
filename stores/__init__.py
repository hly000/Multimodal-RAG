from .base import BaseVectorStore
from .faiss_store import FaissVectorStore
from .milvus_store import MilvusVectorStore
from typing import Dict, Any

def create_vector_store(config: Dict[str, Any]) -> BaseVectorStore:
    """
    根据配置创建向量存储实例的工厂函数。
    """
    store_type = config.get("type")
    
    if store_type == "faiss":
        faiss_config = config.get("faiss", {})
        if not all(k in faiss_config for k in ["index_path", "metadata_path", "dimension"]):
            raise ValueError("Faiss 配置不完整，缺少 index_path, metadata_path, 或 dimension。")
        return FaissVectorStore(**faiss_config)
        
    elif store_type == "milvus":
        milvus_config = config.get("milvus", {})
        return MilvusVectorStore(**milvus_config)
        
    else:
        raise ValueError(f"不支持的向量存储类型: '{store_type}'") 