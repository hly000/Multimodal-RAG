import json
import re
from generators import create_generator, BaseGenerator
from prompts import create_prompt_template, create_rerank_prompt
from typing import List, Dict, Any, Tuple, Optional

class GenerativeAssistant:
    """
    生成式助理，负责整合上下文并调用LLM生成最终答案。
    """
    def __init__(self, llm_config: Dict[str, Any]):
        """
        初始化助理。
        :param llm_config: 用于创建生成器实例的配置字典。
        """
        # 这里我们直接使用 create_generator，它会处理 OpenAI 和 Azure 的情况
        self.generator: BaseGenerator = create_generator(llm_config)
        
        # 存储原始配置，以备将来需要传递 temperature 等参数
        self.llm_config = llm_config
        self.rerank_top_k = 3 # 定义在重排后，取前K个结果用于最终生成

    def _rerank(
        self,
        instruction: str,
        candidates: List[Dict[str, Any]],
        query_image: Optional[str] = None,
        query_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """使用LLM对候选结果进行重排序。"""
        # 如果候选项包含distance字段，则我们优先根据distance进行排序
        if candidates and 'distance' in candidates[0]:
            print("检测到距离信息，将根据距离排序并跳过LLM重排。")
            # 距离越小越好，所以升序排序
            return sorted(candidates, key=lambda x: x['distance'])

        print(f"开始对 {len(candidates)} 个候选结果进行重排序...")
        rerank_prompt = create_rerank_prompt(instruction, candidates, query_image, query_text)
        
        try:
            # 强制LLM使用JSON模式
            response_json_str = self.generator.generate(rerank_prompt, response_format={"type": "json_object"})
            
            # 从LLM的返回结果中稳健地提取JSON对象
            # 某些模型可能会在JSON前后添加额外的文本(例如 '```json\n{...}\n```')
            json_match = re.search(r'\{.*\}', response_json_str, re.DOTALL)
            if not json_match:
                print(f"Rerank未能从LLM响应中找到有效的JSON块: {response_json_str}")
                return candidates
            
            clean_json_str = json_match.group(0)
            response_data = json.loads(clean_json_str)
            
            reranked_indices = response_data.get("reranked_indices", [])
            if not isinstance(reranked_indices, list):
                print("警告: Rerank返回的索引不是列表。将使用原始顺序。")
                return candidates

            # 根据LLM返回的索引重建候选列表
            reranked_candidates = [candidates[i] for i in reranked_indices if 0 <= i < len(candidates)]
            print(f"重排序完成，选出 {len(reranked_candidates)} 个相关结果。")
            return reranked_candidates

        except Exception as e:
            print(f"Rerank过程中发生错误: {e}。将使用原始顺序。")
            return candidates # 发生错误时，回退到原始列表

    def answer(
        self,
        instruction: str,
        candidates: List[Dict[str, Any]],
        query_image: Optional[str] = None,
        query_text: Optional[str] = None
    ) -> Tuple[str, Optional[int], List[Dict[str, Any]]]:
        if not candidates:
            return "抱歉，根据您的描述，我没有在知识库中找到相关的商品。", None, []

        # 1. 重排序候选项
        reranked_candidates = self._rerank(instruction, candidates, query_image, query_text)
        
        if not reranked_candidates:
             return "虽然找到了一些相似的商品，但经过进一步筛选，它们似乎不完全符合您的具体要求。", None, candidates

        # 2. 选取Top-K个结果用于最终生成
        final_candidates = reranked_candidates[:self.rerank_top_k]

        # 3. 创建最终答案的提示
        prompt = create_prompt_template(
            instruction=instruction,
            candidates=final_candidates,
            query_image=query_image,
            query_text=query_text
        )

        # 4. 调用生成器获取JSON格式的回答
        response_json_str = self.generator.generate(prompt, response_format={"type": "json_object"})
        
        # 5. 解析JSON
        try:
            json_match = re.search(r'\{.*\}', response_json_str, re.DOTALL)
            if not json_match:
                # 如果LLM完全没返回JSON，则将其整个输出作为答案
                return response_json_str, None, final_candidates

            clean_json_str = json_match.group(0)
            response_data = json.loads(clean_json_str)
            
            answer_text = response_data.get("answer_text", "我没有找到合适的答案。")
            recommended_index = response_data.get("recommended_index")

            # 验证索引是否有效
            if not isinstance(recommended_index, int) or not (0 <= recommended_index < len(final_candidates)):
                recommended_index = None

            return answer_text, recommended_index, final_candidates

        except json.JSONDecodeError:
            # 如果JSON解析失败，则将原始字符串作为答案返回
            return response_json_str, None, final_candidates
        except Exception as e:
            print(f"解析最终答案时发生错误: {e}")
            return "处理您的请求时遇到了预期之外的错误。", None, final_candidates
