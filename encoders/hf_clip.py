import torch
import cn_clip.clip as clip
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from typing import Optional
from .base import BaseEncoder

class HFClipEncoder(BaseEncoder):
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load_from_name(model_name, device=self.device)
        self.model.eval()

    def _load_image(self, image_path: str) -> Image.Image:
        if image_path.startswith("http://") or image_path.startswith("https://"):
            response = requests.get(image_path)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        return Image.open(image_path).convert("RGB")

    def encode(self, image: Optional[str] = None, text: Optional[str] = None) -> np.ndarray:
        if image is None and not (text and text.strip()):
            raise ValueError("必须提供图片或非空的文本进行编码。")

        image_features = None
        text_features = None

        with torch.no_grad():
            if image:
                pil_image = self._load_image(image)
                image_features = self.model.encode_image(self.preprocess(pil_image).unsqueeze(0).to(self.device))
                image_features /= image_features.norm(dim=-1, keepdim=True)

            if text and text.strip():
                tokens = clip.tokenize(text).to(self.device)
                text_features = self.model.encode_text(tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

            if image_features is not None and text_features is not None:
                # 通过向量加法进行特征融合, 并重新归一化
                fused_features = image_features + text_features
                fused_features /= fused_features.norm(dim=-1, keepdim=True)
                return fused_features.cpu().numpy().flatten()
            elif image_features is not None:
                return image_features.cpu().numpy().flatten()
            elif text_features is not None:
                return text_features.cpu().numpy().flatten()
        
        return np.array([]) 