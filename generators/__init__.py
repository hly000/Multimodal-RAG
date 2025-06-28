from .base import BaseGenerator
from .openai_generator import OpenAIGenerator
from typing import Dict, Any

def create_generator(config: Dict[str, Any]) -> BaseGenerator:
    """
    根据配置创建生成器实例的工厂函数。
    """
    gen_type = config.get("type")
    
    if gen_type == "openai" or gen_type == "azure" or gen_type == "custom":
        # 合并 'openai', 'azure', 'custom' 的配置
        llm_config = config.get(gen_type, {})
        # 将 'type' 键作为 'api_type' 传递给构造函数
        llm_config['api_type'] = gen_type
        
        # 确保 'model' 存在
        if 'model' not in llm_config:
             raise ValueError(f"LLM 配置 '{gen_type}' 中缺少 'model'。")
             
        return OpenAIGenerator(**llm_config)
        
    # 在此可以添加对其他生成器类型的支持
    # elif gen_type == "some_other_llm":
    #     return SomeOtherLLMGenerator(...)
        
    else:
        raise ValueError(f"不支持的生成器类型: '{gen_type}'") 