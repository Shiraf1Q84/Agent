import streamlit as st
import google.generativeai as genai
import logging
import json

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# カスタムCSS
custom_css = """
<style>
    body {
        color: #888888;  /* 文字色を灰色に設定 */
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.bot {
        background-color: #475063;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
    }
    .inference-step {
        background-color: #2b313e;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .sidebar .stButton>button {
        width: 100%;
    }
</style>
"""

# Streamlitページの設定
st.set_page_config(page_title="Gemini Web Search Agent", layout="wide")
st.markdown(custom_css, unsafe_allow_html=True)

# Geminiのシステムプロンプト
CUSTOM_SYSTEM_PROMPT = """
あなたは高度なAIアシスタントです。ユーザーの質問に答えるために、以下のステップを踏んでください：
1. ユーザーの質問を理解し、適切な検索戦略を立てる。
2. Web検索を使用して情報を収集する。
3. 収集した情報を分析し、ユーザーの質問に対する回答を作成する。
4. 回答の根拠となる情報源を明記する。

各ステップごとに、以下のJSON形式で出力してください：
{"step": "ステップ番号", "action": "行動の説明", "thought": "思考過程"}

最終的な回答は、別途テキストで出力してください。
"""

# Web検索機能（モック）
def search_web(query):
    return f"Web search results for: {query}"

# ページ内容取得機能（モック）
def fetch_page(url):
    return f"Content of page: {url}"

# Geminiモデルの初期化
def initialize_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-pro')

# サイドバーの設定
with st.sidebar:
    st.title("設定")
    api_key = st.text_input("Google AI API Key", type="password")
    if st.button("APIキーを設定"):
        if api_key:
            try:
                st.session_state.model = initialize_model(api_key)
                st.success("APIキーが正常に設定されました。")
            except Exception as e:
                st.error(f"APIキーの設定中にエラーが発生しました: {str(e)}")
        else:
            st.warning("APIキーを入力してください。")

# メインコンテンツ
st.title("Gemini Web Search Agent")

# チャット履歴とUIコンポーネントの初期化
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'inference_steps' not in st.session_state:
    st.session_state.inference_steps = []

# チャットエリアとインファレンスエリアのレイアウト
chat_column, inference_column = st.columns([2, 1])

# チャットエリア
with chat_column:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ユーザー入力
    if prompt := st.chat_input("質問を入力してください"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Geminiモデルが設定されている場合、応答を生成
        if 'model' in st.session_state:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    for chunk in st.session_state.model.generate_content(
                        f"{CUSTOM_SYSTEM_PROMPT}\n\nユーザーの質問: {prompt}",
                        stream=True
                    ):
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"応答生成中にエラーが発生しました: {str(e)}")
                    logger.error(f"Error during response generation: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # 推論ステップの抽出と表示
            st.session_state.inference_steps = []
            for line in full_response.split('\n'):
                try:
                    step = json.loads(line)
                    if isinstance(step, dict) and 'step' in step:
                        st.session_state.inference_steps.append(step)
                except json.JSONDecodeError:
                    pass

# インファレンスエリア
with inference_column:
    st.subheader("推論プロセス")
    for step in st.session_state.inference_steps:
        with st.expander(f"ステップ {step['step']}"):
            st.write(f"**行動:** {step['action']}")
            st.write(f"**思考:** {step['thought']}")

# APIキーが設定されていない場合の警告
if 'model' not in st.session_state:
    st.warning("Google AI API Keyを設定してください。")