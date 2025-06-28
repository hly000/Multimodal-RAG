from typing import List, Dict, Any, Optional

def create_prompt_template(
    instruction: str, 
    candidates: List[Dict[str, Any]], 
    query_image: Optional[str] = None, 
    query_text: Optional[str] = None
) -> str:
    """
    根据用户输入和检索到的候选结果，创建一个结构化的提示。

    :param instruction: 用户的核心指令或问题。
    :param candidates: 从向量存储中检索到的候选项目列表。
    :param query_image: 用户上传的查询图片URL或路径（可选）。
    :param query_text: 用户的文本查询（可选）。
    :return: 格式化后的字符串提示。
    """
    
    # 1. 构建用户查询部分
    user_query_section = "--- User Query ---\n"
    if query_text:
        user_query_section += f"Query Text: \"{query_text}\"\n"
    if query_image:
        user_query_section += f"Query Image: (User provided an image for reference)\n"
    user_query_section += f"User's specific instruction is: \"{instruction}\"\n"

    # 2. 构建检索到的上下文部分
    context_section = "--- Retrieved Relevant Products ---\n"
    if not candidates:
        context_section += "No relevant products found.\n"
    else:
        for i, item in enumerate(candidates):
            context_section += f"Product {i}:\n"
            for key, value in item.items():
                if key not in ['id', 'distance', 'is_annotated']: # 过滤掉内部元数据
                    context_section += f"  - {key.capitalize()}: {value}\n"

    # 3. 构建最终指令
    final_instruction = (
        "--- Task Instruction ---\n"
        "You are a professional shopping assistant. Your task is to answer the user's query based ONLY on the【Retrieved Relevant Products】.\n"
        "**CRITICAL RULES:**\n"
        "1. Your response MUST be in JSON format.\n"
        "2. The JSON object must have two keys: `recommended_index` and `answer_text`.\n"
        "3. `recommended_index`: The integer index of the product you are primarily recommending. If no single product is a good fit, use `null`.\n"
        "4. `answer_text`: A helpful, natural language text for the user. In this text, explain WHY the product is a good match, citing its details. If no product is a good match, honestly explain why.\n"
        "5. Your `answer_text` MUST be based strictly on the provided product information. DO NOT use external knowledge.\n"
        "Your JSON response is:"
    )
    
    return f"{user_query_section}\n{context_section}\n{final_instruction}"

def create_rerank_prompt(
    instruction: str,
    candidates: List[Dict[str, Any]],
    query_image: Optional[str] = None,
    query_text: Optional[str] = None
) -> str:
    """
    创建一个用于重排序的结构化提示，要求LLM返回JSON。
    """
    # 1. 构建用户查询部分
    user_query_section = "--- User Query ---\n"
    if query_text:
        user_query_section += f"Query Text: \"{query_text}\"\n"
    if query_image:
        user_query_section += f"Query Image: (User provided an image for reference)\n"
    user_query_section += f"User's specific instruction is: \"{instruction}\"\n"

    # 2. 构建待排序的候选商品部分
    context_section = "--- Retrieved Candidates for Reranking ---\n"
    if not candidates:
        context_section += "No candidates found.\n"
    else:
        for i, item in enumerate(candidates):
            context_section += f"Candidate {i}:\n"
            # 动态地将所有字段都包含进来，除了可能的内部元数据
            for key, value in item.items():
                # 假设 'id' 和 'distance' 是向量数据库的元数据，我们通常不需要让LLM看到
                if key not in ['id', 'distance']: 
                     context_section += f"  - {key.capitalize()}: {value}\n"

    # 3. 构建最终指令
    final_instruction = (
        "--- Task Instruction ---\n"
        "You are an expert relevance judge. Your task is to evaluate the relevance of each candidate product to the user's query.\n"
        "Based on the query and the candidate list, sort the candidates from most relevant to least relevant.\n"
        "You MUST return a JSON object with a single key 'reranked_indices', which must be a list of integers. "
        "Each integer corresponds to the original index of a candidate.\n"
        "For example, if you think Candidate 2 is most relevant, followed by Candidate 0, your response should be: {\"reranked_indices\": [2, 0, ...]}\n"
        "Your JSON response is:"
    )
    
    return f"{user_query_section}\n{context_section}\n{final_instruction}" 