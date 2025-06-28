import os
import yaml
from PIL import Image
import requests
from io import BytesIO

def get_config(config_path="configs/config.yaml"):
    """
    加载并解析YAML配置文件。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_uploaded_file(uploaded_file, save_dir="temp_uploads") -> str:
    """
    将Streamlit上传的文件保存到本地临时目录。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def get_image_from_url_or_path(source: str) -> Image.Image:
    """
    从URL或本地路径加载图片，并返回PIL Image对象。
    """
    if source.startswith(('http://', 'https://')):
        try:
            response = requests.get(source)
            response.raise_for_status()  # 如果请求失败则引发HTTPError
            image = Image.open(BytesIO(response.content))
        except requests.exceptions.RequestException as e:
            raise IOError(f"从URL加载图片失败: {source}, 错误: {e}")
    elif os.path.exists(source):
        try:
            image = Image.open(source)
        except IOError as e:
            raise IOError(f"从本地路径加载图片失败: {source}, 错误: {e}")
    else:
        raise FileNotFoundError(f"图片文件或URL不存在: {source}")
    
    return image.convert("RGB") 