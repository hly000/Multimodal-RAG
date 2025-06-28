from .base import BaseEncoder
from .hf_clip import HFClipEncoder

def create_encoder(config: dict) -> BaseEncoder:
    """
    根据配置创建编码器实例的工厂函数。
    """
    encoder_type = config.get("type")
    if encoder_type == "hf_clip":
        clip_config = config.get("hf_clip", {})
        if not clip_config.get("model_name"):
            raise ValueError("HuggingFace Clip 'hf_clip' 配置中缺少 'model_name'。")
        return HFClipEncoder(model_name=clip_config.get("model_name"))
    # 在此添加对其他编码器类型的支持
    # elif encoder_type == "some_other_encoder":
    #     return SomeOtherEncoder(...)
    else:
        raise ValueError(f"不支持的编码器类型: '{encoder_type}'") 