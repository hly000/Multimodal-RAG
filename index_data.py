import yaml
import pandas as pd
import os
import sys
from tqdm import tqdm

from encoders import create_encoder
from stores import create_vector_store
from utils import get_config, get_image_from_url_or_path

def main(config_path="configs/config.yaml", data_path="dataset/your_data.xlsx"):
    print("1. 加载配置...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    print("2. 初始化后端组件...")
    encoder = create_encoder(config['encoder'])
    vector_store = create_vector_store(config['vector_store'])
    
    print(f"3. 加载数据源: {data_path}...")
    if not os.path.exists(data_path):
        print(f"错误：数据文件不存在于 {data_path}")
        return
        
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(data_path)
    else:
        raise ValueError("不支持的数据文件格式。请使用 CSV 或 Excel 文件。")

    print("4. 清理旧索引...")
    vector_store.delete_collection()
    
    print("5. 开始数据索引流程...")
    items_to_index = [{'path': row['url'], 'category': row.get('category', ''), 'description': row.get('desc', '')} 
                      for _, row in df.iterrows() if pd.notna(row['url'])]
    
    batch_size = 16
    for i in tqdm(range(0, len(items_to_index), batch_size), desc="索引批处理"):
        batch_items = items_to_index[i:i+batch_size]
        
        # 注意：这里的示例分别对每个项目进行编码，在实际应用中，批量编码会更高效
        vectors = [encoder.encode(image=item['path'], text=item.get('description', '')) for item in batch_items]
        metadata = [{'url': item['path'], 'category': item.get('category', ''), 'description': item.get('description', '')} for item in batch_items]
        
        vector_store.add(vectors=vectors, metadata=metadata)
    
    print("6. 构建最终索引...")
    vector_store.build_index()
    print(f"✅ 索引完成！共处理 {len(items_to_index)} 个项目。")

if __name__ == "__main__":
    # 您可以通过命令行参数覆盖默认值，或者在这里直接修改
    # 例如: python index_data.py --data_path /path/to/your/data.csv
    import argparse
    parser = argparse.ArgumentParser(description="多模态数据索引脚本")
    parser.add_argument("--config_path", type=str, default="configs/config.yaml", help="配置文件的路径")
    parser.add_argument("--data_path", type=str, default="dataset/template.xlsx", help="待索引数据文件的路径")
    args = parser.parse_args()
    
    main(config_path=args.config_path, data_path=args.data_path) 