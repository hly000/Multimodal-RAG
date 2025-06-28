import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional
from .base import BaseVectorStore

class FaissVectorStore(BaseVectorStore):
    def __init__(self, index_path: str, metadata_path: str, dimension: int, **kwargs):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = dimension
        self.index = None
        self.metadata = []
        self._load()

    def _load(self):
        """加载索引和元数据，如果不存在则创建新的。"""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            if self.index.d != self.dimension:
                print(f"警告: 索引维度 ({self.index.d}) 与配置 ({self.dimension}) 不符。将创建新索引。")
                self._create_new_index()
        else:
            self._create_new_index()
            
        if os.path.exists(self.metadata_path):
             with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

    def _create_new_index(self):
        """创建一个新的空索引。"""
        print("正在创建新的 Faiss 索引和元数据文件...")
        # 确保目录存在
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self._save()

    def _save(self):
        """保存索引和元数据到文件。"""
        print(f"正在保存 Faiss 索引到 {self.index_path}")
        faiss.write_index(self.index, self.index_path)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def add(self, vectors: List[np.ndarray], metadata: List[Dict[str, Any]], **kwargs):
        if len(vectors) == 0:
            return
        vectors_np = np.array(vectors, dtype='float32')
        self.index.add(vectors_np)
        self.metadata.extend(metadata)

    def search(
        self, 
        vector: np.ndarray, 
        top_k: int, 
        output_fields: Optional[List[str]] = None, 
        **kwargs
    ) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        query_vector_np = np.array([vector], dtype='float32')
        distances, indices = self.index.search(query_vector_np, top_k)
        
        results = []
        for i in indices[0]:
            if i != -1 and i < len(self.metadata):
                results.append(self.metadata[i])
        return results

    def build_index(self):
        # FaissIndexFlatL2 是增量添加的，所以我们只需要在这里保存最终状态。
        self._save()
        print("Faiss 索引已成功保存。")

    def delete_collection(self):
        print("正在删除旧的 Faiss 索引和元数据...")
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        self._create_new_index()
        print("旧索引已成功删除并重新创建。")
        
    def release(self):
        # 对于基于文件的 Faiss，此操作可以理解为清空内存中的对象，
        # 等待下次使用时重新从磁盘加载。
        self.index = None
        self.metadata = []
        print("Faiss 索引已从内存中释放。") 