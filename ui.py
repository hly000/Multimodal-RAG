import streamlit as st
from PIL import Image
import json
import os
import yaml
import sys
from typing import Dict, Any, Tuple, List
import pandas as pd
from tqdm import tqdm
import math
import io
from urllib.parse import urlparse

# 从项目模块中导入核心组件
from encoders import create_encoder, BaseEncoder
from stores import create_vector_store, BaseVectorStore
from reranker import GenerativeAssistant
from prompts import create_prompt_template
from utils import get_config, save_uploaded_file, get_image_from_url_or_path

# Streamlit页面基础设置
st.set_page_config(layout="wide", page_title="多模态 RAG 问答")

# 5. 初始化Session State
if 'app_state' not in st.session_state:
    st.session_state['app_state'] = "NEEDS_CONFIG"  # "NEEDS_CONFIG", "NEEDS_INDEX", "READY"
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = "配置"
if 'backend' not in st.session_state:
    st.session_state['backend'] = None
if 'qa_history' not in st.session_state:
    st.session_state['qa_history'] = []
if 'annotation_df' not in st.session_state:
    st.session_state['annotation_df'] = None
if 'annotation_page' not in st.session_state:
    st.session_state['annotation_page'] = 1

# 为配置表单添加持久化状态
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = "OpenAI API"
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'openai_base_url' not in st.session_state:
    st.session_state.openai_base_url = ""
if 'azure_api_key' not in st.session_state:
    st.session_state.azure_api_key = ""
if 'azure_endpoint' not in st.session_state:
    st.session_state.azure_endpoint = ""
if 'azure_api_version' not in st.session_state:
    st.session_state.azure_api_version = "2023-12-01-preview"
if 'custom_api_key' not in st.session_state:
    st.session_state.custom_api_key = ""
if 'custom_base_url' not in st.session_state:
    st.session_state.custom_base_url = ""

# 6. 定义后端服务函数
@st.cache_data
def load_base_config(config_path="configs/config.yaml") -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@st.cache_resource
def initialize_backend(_config: Dict[str, Any]) -> Tuple[BaseEncoder, BaseVectorStore, GenerativeAssistant]:
    llm_config = _config.get("llm", {})
    llm_type = llm_config.get("type", "openai")
    
    # 根据类型设置环境变量
    if llm_type == "azure":
        creds = llm_config.get("azure", {})
        os.environ["AZURE_OPENAI_KEY"] = creds.get("api_key", "")
        os.environ["AZURE_OPENAI_ENDPOINT"] = creds.get("endpoint", "")
    else: # openai or custom
        creds = llm_config.get("openai", {})
        os.environ["OPENAI_API_KEY"] = creds.get("api_key", "")
        if "base_url" in creds and creds["base_url"]:
            os.environ["OPENAI_BASE_URL"] = creds["base_url"]
            
    encoder = create_encoder(_config["encoder"])
    vector_store = create_vector_store(_config["vector_store"])
    assistant = GenerativeAssistant(llm_config)
    return encoder, vector_store, assistant

def perform_indexing(df: pd.DataFrame, vector_store: BaseVectorStore, encoder: BaseEncoder, progress_bar) -> Tuple[bool, str]:
    try:
        items_to_index = df.to_dict('records')
        vector_store.delete_collection()
        
        batch_size = 32
        num_batches = (len(items_to_index) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            batch_items = items_to_index[i * batch_size : (i + 1) * batch_size]
            
            image_urls = [item['url'] for item in batch_items]
            texts = [item.get('desc', '') for item in batch_items] # 添加文本描述
            
            # 批量编码
            vectors = [encoder.encode(image=url, text=txt) for url, txt in zip(image_urls, texts)]
            
            # 使用新的接口
            vector_store.add(vectors=vectors, metadata=batch_items)
            
            progress_bar.progress((i + 1) / num_batches)
            
        vector_store.build_index()
        return True, f"成功为 {len(items_to_index)} 条数据建立了索引。"
    except Exception as e:
        st.error(f"建立索引时发生错误: {e}")
        return False, f"建立索引时发生错误: {e}"

# 7. 构建主UI界面
st.title("🚀 多模态 RAG 问答")
tab_titles = ["⚙️ 配置", "📚 数据管理", "💬 开始问答"]
tab1, tab2, tab3 = st.tabs(tab_titles)

# 8. 实现 Tab 1: 系统配置
with tab1:
    st.subheader("系统配置")
    with st.container(border=True):
        st.info("当前版本使用内置的 Faiss 作为向量数据库，配置信息已从 `config.yaml` 加载。")
        vs_type = st.selectbox("向量数据库类型", ["faiss", "milvus"], disabled=True)
    
    with st.container(border=True):
        st.subheader("LLM 配置")
        st.selectbox(
            "选择语言模型服务商", 
            ["OpenAI API", "Azure OpenAI", "自定义 API"],
            key='llm_provider'
        )
        
        if st.session_state.llm_provider == "OpenAI API":
            st.text_input("OpenAI API Key", type="password", key="openai_api_key")
            st.text_input("OpenAI Base URL (可选)", placeholder="https://api.openai.com/v1", key="openai_base_url")
        elif st.session_state.llm_provider == "Azure OpenAI":
            st.text_input("Azure OpenAI Key", type="password", key="azure_api_key")
            st.text_input("Azure OpenAI Endpoint", key="azure_endpoint")
            st.text_input("API Version", key="azure_api_version")
        else: # 自定义 API
            st.text_input("API Key", type="password", key="custom_api_key")
            st.text_input("API Base URL", placeholder="例如: https://api.groq.com/openai/v1", key="custom_base_url")

    if st.button("应用配置", type="primary"):
        # 验证输入，直接从会话状态中读取值
        valid_input = True
        if st.session_state.llm_provider == "OpenAI API" and not st.session_state.openai_api_key:
            st.error("请输入 OpenAI API Key。")
            valid_input = False
        elif st.session_state.llm_provider == "Azure OpenAI" and (not st.session_state.azure_api_key or not st.session_state.azure_endpoint):
            st.error("请输入 Azure OpenAI Key 和 Endpoint。")
            valid_input = False
        elif st.session_state.llm_provider == "自定义 API" and not st.session_state.custom_base_url:
            st.error("请输入 API Base URL。")
            valid_input = False
        
        if valid_input:
            with st.spinner("正在初始化后端服务..."):
                dynamic_config = load_base_config()
                if st.session_state.llm_provider == "OpenAI API":
                    dynamic_config['llm']['type'] = 'openai'
                    dynamic_config['llm']['openai']['api_key'] = st.session_state.openai_api_key
                    dynamic_config['llm']['openai']['base_url'] = st.session_state.openai_base_url or None
                elif st.session_state.llm_provider == "Azure OpenAI":
                    dynamic_config['llm']['type'] = 'azure'
                    dynamic_config['llm']['azure'] = {
                        'api_key': st.session_state.azure_api_key,
                        'endpoint': st.session_state.azure_endpoint,
                        'api_version': st.session_state.azure_api_version
                    }
                else: # 自定义
                    dynamic_config['llm']['type'] = 'custom'
                    # 从已有配置中获取模型名，如果不存在则使用默认值
                    model_name = dynamic_config.get('llm', {}).get('openai', {}).get('model', 'gpt-3.5-turbo')
                    dynamic_config['llm']['custom'] = {
                        'api_key': st.session_state.custom_api_key,
                        'base_url': st.session_state.custom_base_url,
                        'model': model_name
                    }

                st.session_state.backend = initialize_backend(dynamic_config)
                st.session_state.app_state = "NEEDS_INDEX"
                st.success("✅ 配置成功！请前往“数据管理”选项卡上传数据并建立索引。")

# 9. 实现 Tab 2: 数据管理
with tab2:
    if st.session_state.app_state == "NEEDS_CONFIG":
        st.warning("⚠️ 请先在“配置”选项卡中完成系统配置。")
    else:
        st.subheader("数据上传与索引")
        left, right = st.columns([1, 1])
        with left:
            with st.container(border=True):
                st.markdown("##### 1. 上传数据文件")
                with open("dataset/template.xlsx", "rb") as f:
                    st.download_button(
                        label="下载数据模板",
                        data=f,
                        file_name="template.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                uploaded_file = st.file_uploader(
                    "支持 .csv 和 .xlsx 格式", 
                    type=["csv", "xlsx"], 
                    key="file_uploader"
                )
                if uploaded_file:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            st.session_state.annotation_df = pd.read_csv(uploaded_file)
                        else:
                            st.session_state.annotation_df = pd.read_excel(uploaded_file)
                        st.success("文件上传成功！")
                    except Exception as e:
                        st.error(f"读取文件时出错: {e}")
        
        with right:
            with st.container(border=True):
                st.markdown("##### 2. 建立索引")
                if st.session_state.annotation_df is not None:
                    st.metric("待索引数据量", f"{len(st.session_state.annotation_df)} 条")
                    if st.button("开始建立索引", type="primary"):
                        encoder, vector_store, _ = st.session_state.backend
                        progress_bar = st.progress(0, "正在建立索引...")
                        success, message = perform_indexing(st.session_state.annotation_df, vector_store, encoder, progress_bar)
                        if success:
                            st.session_state.app_state = "READY"
                            st.success(f"✅ {message} 现在可以去“开始问答”啦！")
                            progress_bar.progress(1.0, "索引完成！")
                        else:
                            st.error(f"索引失败: {message}")
                else:
                    st.info("请先上传数据文件。")
        
        st.divider()

        st.subheader("在线数据标注平台")
        if st.session_state.annotation_df is not None:
            # 创建一个工作副本
            df = st.session_state.annotation_df.copy()

            # --- 1. 标注状态分析 ---
            # 确保列存在，如果不存在则填充为空字符串
            if 'desc' not in df.columns:
                df['desc'] = ''
            if 'category' not in df.columns:
                df['category'] = ''

            # 填充NaN值以便统一处理
            df['desc'] = df['desc'].fillna('')
            df['category'] = df['category'].fillna('')

            # 判断是否已标注 (描述和类别都不为空)
            df['is_annotated'] = (df['desc'].str.strip() != '') & (df['category'].str.strip() != '')
            
            total_items = len(df)
            annotated_count = df['is_annotated'].sum()
            unannotated_count = total_items - annotated_count

            # --- 2. UI - 状态指标与筛选 ---
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("总数据量", f"{total_items} 条")
            with c2:
                st.metric("已标注", f"{annotated_count} 条")
            with c3:
                st.metric("待标注", f"{unannotated_count} 条")
            
            if unannotated_count == 0:
                st.success("🎉 恭喜！所有数据都已标注完成。")

            with st.container(border=True):
                st.markdown("##### 筛选与视图")
                filter_c1, filter_c2 = st.columns([3, 1])
                
                # 按类别筛选
                all_categories = sorted([cat for cat in df['category'].unique() if cat])
                selected_categories = filter_c1.multiselect(
                    "按类别筛选",
                    options=all_categories,
                    help="选择一个或多个类别以筛选数据。"
                )
                
                # 仅显示待标注数据的复选框
                show_only_unannotated = filter_c2.checkbox(
                    "仅显示待标注", 
                    value=(unannotated_count > 0), # 如果有未标注项，默认勾选
                    help="勾选此项以查看所有缺少描述或类别的数据。"
                )

            # --- 3. 应用筛选逻辑 ---
            df_to_display = df

            if show_only_unannotated:
                df_to_display = df_to_display[~df_to_display['is_annotated']]
            
            if selected_categories:
                df_to_display = df_to_display[df_to_display['category'].isin(selected_categories)]

            st.markdown(f"**查询结果: {len(df_to_display)} 条**")

            # --- 4. 分页与表单展示 ---
            if not df_to_display.empty:
                items_per_page = 5
                total_pages = math.ceil(len(df_to_display) / items_per_page)
                
                # 如果筛选后当前页码超出范围，重置为第一页
                if st.session_state.annotation_page > total_pages:
                    st.session_state.annotation_page = 1
                
                page_number = st.number_input('页码', min_value=1, max_value=total_pages, value=st.session_state.annotation_page)
                st.session_state.annotation_page = page_number
                
                start_idx = (page_number - 1) * items_per_page
                end_idx = start_idx + items_per_page
                
                for index, row in df_to_display.iloc[start_idx:end_idx].iterrows():
                    with st.container(border=True):
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.image(row['url'], use_container_width=True, caption=f"ID: {index}")
                        with c2:
                            with st.form(f"form_{index}"):
                                new_desc = st.text_area("描述 (desc)", value=row.get('desc', ''), height=150)
                                new_cat = st.text_input("类别 (category)", value=row.get('category', ''))
                                if st.form_submit_button("保存更改"):
                                    # 直接在原始DataFrame上修改
                                    st.session_state.annotation_df.at[index, 'desc'] = new_desc
                                    st.session_state.annotation_df.at[index, 'category'] = new_cat
                                    st.toast(f"ID {index} 已更新！")
                                    # 重新运行以刷新界面和统计数据
                                    st.rerun()
            else:
                st.info("没有找到符合筛选条件的数据。")
        else:
            st.info("请先上传数据文件以开始标注。")

# 10. 实现 Tab 3: 开始问答
with tab3:
    if st.session_state.app_state != "READY":
        st.warning("⚠️ 系统尚未就绪。请先完成配置并建立数据索引。")
    else:
        for msg in st.session_state.qa_history:
            with st.chat_message(msg["role"]):
                if msg.get("image_query"):
                    st.image(msg["image_query"], width=150)
                if msg.get("text_query"):
                    st.write(msg["text_query"])
                
                # 新的答案显示逻辑
                if "answer" in msg:
                    st.markdown(msg["answer"])
                    
                    # 如果有指定的推荐图片，则直接展示
                    if msg.get("recommended_item"):
                        st.image(
                            msg["recommended_item"]["url"], 
                            caption=f"为您推荐: {msg['recommended_item'].get('desc', '')[:50]}",
                            use_container_width=True
                        )

                    # 将所有参考图片放入折叠面板中
                    if msg.get("references"):
                        with st.expander("查看所有参考图片"):
                            cols = st.columns(len(msg["references"]))
                            for i, ref in enumerate(msg["references"]):
                                with cols[i]:
                                    st.image(ref['url'], caption=ref.get('desc', '')[:50], use_container_width=True)

        query_image_upload = st.file_uploader("上传查询图片", type=["jpg", "png", "jpeg"])
        query_text_input = st.text_area("输入你的问题")
        
        if st.button("发送问题", type="primary"):
            if not query_image_upload and not query_text_input.strip():
                st.error("请输入问题或上传图片。")
            else:
                user_message = {"role": "user"}
                query_image_bytes = None
                if query_image_upload:
                    query_image_bytes = query_image_upload.getvalue()
                    user_message["image_query"] = query_image_bytes
                if query_text_input.strip():
                    user_message["text_query"] = query_text_input
                
                st.session_state.qa_history.append(user_message)
                st.rerun()

    if st.session_state.qa_history and st.session_state.qa_history[-1]["role"] == "user":
        last_user_msg = st.session_state.qa_history[-1]
        
        query_image_path = None
        if last_user_msg.get("image_query"):
            # 将上传的图片字节保存为临时文件以传递给模型
            with open("temp_query_image.jpg", "wb") as f:
                f.write(last_user_msg["image_query"])
            query_image_path = "temp_query_image.jpg"

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                encoder, vector_store, assistant = st.session_state.backend
                
                # 编码
                query_vector = encoder.encode(image=query_image_path, text=last_user_msg.get("text_query"))
                
                # 检索
                candidates = vector_store.search(vector=query_vector, top_k=5)
                
                # 生成 (现在返回三元组)
                answer_text, recommended_idx, references = assistant.answer(
                    instruction=last_user_msg.get("text_query", "请描述这张图片，并找到相似的商品。"),
                    candidates=candidates,
                    query_image=query_image_path,
                    query_text=last_user_msg.get("text_query")
                )
                
                # 准备要存入历史记录的消息体
                assistant_message = {
                    "role": "assistant", 
                    "answer": answer_text,
                    "references": references,
                    "recommended_item": references[recommended_idx] if recommended_idx is not None else None
                }
                st.session_state.qa_history.append(assistant_message)
                st.rerun()
