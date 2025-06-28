import os
import requests
import torch
from tqdm import tqdm

def repair_and_download_model():
    """
    这是一个修复工具。它会检查、删除损坏的缓存文件，并从官方源重新下载。
    """
    model_url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-b-16.pt"
    cache_dir = os.path.expanduser("~/.cache/clip")
    file_path = os.path.join(cache_dir, os.path.basename(model_url))

    # 1. 检查文件是否存在
    if os.path.exists(file_path):
        print(f"检测到已存在的缓存文件: {file_path}")
        try:
            # 2. 尝试加载以验证文件是否完好
            torch.load(file_path, map_location="cpu")
            print("✅ 缓存文件验证通过，是完好的。无需重新下载。")
            return
        except Exception as e:
            # 3. 如果加载失败，说明文件已损坏，必须删除
            print(f"⚠️ 缓存文件已损坏 (错误: {e})。")
            print("正在强制删除损坏文件...")
            os.remove(file_path)
            print("损坏文件已删除。")

    # 4. 如果文件不存在或已被删除，则开始下载
    print("-" * 50)
    print(f"开始从官方源下载一个完好的模型...")
    print(f"URL: {model_url}")
    print(f"保存路径: {file_path}")

    os.makedirs(cache_dir, exist_ok=True)

    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as f, tqdm(
            desc=os.path.basename(model_url), total=total_size, unit='iB',
            unit_scale=True, unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        print("\n" + "="*50)
        print(f"✅ 模型修复/下载成功！")
        print(f"完好的模型文件已位于: {file_path}")
        print("现在请重启您的Streamlit应用。")
        print("="*50)

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    repair_and_download_model() 