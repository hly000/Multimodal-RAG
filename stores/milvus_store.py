from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
from .base import BaseVectorStore
from typing import List, Dict, Any, Optional

class MilvusVectorStore(BaseVectorStore):
    def __init__(self, uri, user, password, db_name, collection_name, dimension):
        self.client = MilvusClient(uri=uri, user=user, password=password, db_name=db_name)
        self.collection_name = collection_name
        self.dimension = dimension
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        if self.collection_name not in self.client.list_collections():
            pk_field = FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True)
            vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            # 允许存储动态元数据的JSON字段
            metadata_field = FieldSchema(name="metadata", dtype=DataType.JSON)
            schema = CollectionSchema(fields=[pk_field, vector_field, metadata_field], enable_dynamic_field=True)
            self.client.create_collection(self.collection_name, schema=schema)
            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="vector", index_type="AUTOINDEX")
            self.client.create_index(self.collection_name, index_params)

    def add(self, vectors: List[np.ndarray], metadata: List[Dict[str, Any]], **kwargs):
        data = [{"vector": vec, "metadata": meta} for vec, meta in zip(vectors, metadata)]
        self.client.insert(self.collection_name, data)

    def search(
        self, 
        vector: np.ndarray, 
        top_k: int, 
        output_fields: Optional[List[str]] = None, 
        **kwargs
    ) -> List[Dict[str, Any]]:
        # 默认总是返回元数据
        if output_fields is None:
            output_fields = ["metadata"]
        else:
            if "metadata" not in output_fields:
                output_fields.append("metadata")
                
        results = self.client.search(
            collection_name=self.collection_name,
            data=[vector.tolist()],
            limit=top_k,
            output_fields=output_fields,
            search_params={"metric_type": "L2"}
        )
        
        # 同时返回元数据和距离
        processed_results = []
        for res in results[0]:
            metadata = res['entity']['metadata']
            metadata['distance'] = res['distance']
            processed_results.append(metadata)
            
        return processed_results

    def delete_collection(self):
        if self.collection_name in self.client.list_collections():
            self.client.drop_collection(self.collection_name)
    
    def build_index(self):
        # Milvus的AUTOINDEX是自动构建的，此函数为空操作
        pass
        
    def release(self):
        # Milvus 客户端会自动管理连接，但可以提供一个释放加载集合的接口
        self.client.release_collection(self.collection_name)
        print(f"集合 '{self.collection_name}' 已从内存中释放。") 