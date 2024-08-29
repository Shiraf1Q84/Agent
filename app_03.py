import streamlit as st
import google.generativeai as genai
import logging
import json
import time

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# カスタムCSS
custom_css = """
<style>
    body { background-color: #0E1117; color: #00FF00; font-family: 'Courier New', monospace; }
    .stApp { max-width: 1200px; margin: 0 auto; }
    .main-content { display: flex; }
    .chat-area { flex: 2; padding-right: 20px; }
    .inference-area { flex: 1; border-left: 1px solid #00FF00; padding-left: 20px; height: 100vh; overflow-y: auto; }
    .chat-message { border: 1px solid #00FF00; border-radius: 5px; padding: 10px; margin-bottom: 10px; }
    .user-message { background-color: #1E2A38; }
    .bot-message { background-color: #2A1E38; }
    .inference-step { border: 1px solid #00FF00; border-radius: 5px; padding: 10px; margin-bottom: 10px; }
    .blinking-cursor::after { content: '▊'; animation: blink 1s step-end infinite; }
    @keyframes blink { 50% { opacity: 0; } }
    .sidebar .stButton>button { width: 100%; background-color: #00FF00; color: #0E1117; }
</style>
"""

# Streamlitページの設定
st.set_page_config(page_title="Gemini Advanced AI Agent", layout="wide")
st.markdown(custom_css, unsafe_allow_html=True)

# Geminiのシステムプロンプト
CUSTOM_SYSTEM_PROMPT = """
あなたは高度な AI エージェントとして、以下の指示に従って動作してください：

1. ユーザーの質問を詳細に分析し、複数の視点から検討してください。
2. 必要に応じて、Web 検索や情報収集を行うことを想定してください。
3. 収集した情報を統合し、包括的な回答を作成してください。
4. 回答には、技術的な詳細、関連する理論、実践的な応用例を含めてください。
5. 可能な限り、コードスニペット、擬似コード、または技術的な図表の説明を含めてください。
6. 異なる見解や代替アプローチがある場合は、それらも提示してください。
7. 回答の信頼性を高めるため、仮想的な情報源や参考文献を引用してください。
8. 専門用語を積極的に使用し、必要に応じて簡潔な説明を付け加えてください。

各ステップを以下の JSON 形式で出力してください：
{"step": "ステップ番号", "action": "行動の説明", "thought": "詳細な思考過程", "output": "行動の結果または観察"}

最終的な回答は、JSON 形式とは別に、詳細かつ構造化されたテキストで出力してください。
"""

# モックの Web 検索と情報取得関数
def mock_web_search(query):
    time.sleep(1)  # 実際の検索を模倣するための遅延
    return f"Web search results for '{query}': [模擬データ] 関連する技術記事、学術論文、オープンソースプロジェクトの情報"

def mock_fetch_info(topic):
    time.sleep(0.5)  # 情報取得を模倣するための遅延
    return f"Detailed information about '{topic}': [模擬データ] 技術仕様、実装の詳細、パフォーマンス分析"

# Gemini モデルの初期化
def initialize_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-pro')

# サイドバー設定
with st.sidebar:
    st.markdown("## Gemini Advanced AI Agent")
    api_key = st.text_input("Google AI API Key", type="password")
    if st.button("Initialize Agent"):
        if api_key:
            try:
                st.session_state.model = initialize_model(api_key)
                st.success("Agent initialized successfully.")
            except Exception as e:
                st.error(f"Initialization error: {str(e)}")
        else:
            st.warning("Please enter an API key.")

# メインコンテンツ
st.markdown("<div class='main-content'>", unsafe_allow_html=True)

# チャットエリア
st.markdown("<div class='chat-area'>", unsafe_allow_html=True)
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"<div class='{message['role']}-message'>{message['content']}</div>", unsafe_allow_html=True)

prompt = st.chat_input("Enter your query")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)

    if 'model' in st.session_state:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                for chunk in st.session_state.model.generate_content(
                    f"{CUSTOM_SYSTEM_PROMPT}\n\nUser query: {prompt}",
                    stream=True
                ):
                    full_response += chunk.text
                    message_placeholder.markdown(f"<div class='bot-message blinking-cursor'>{full_response}</div>", unsafe_allow_html=True)
                message_placeholder.markdown(f"<div class='bot-message'>{full_response}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error during response generation: {str(e)}")
                logger.error(f"Error during response generation: {str(e)}")
        st.session_state.messages.append({"role": "assistant", "content": full_response})

st.markdown("</div>", unsafe_allow_html=True)

# 推論エリア（常に表示）
st.markdown("<div class='inference-area'>", unsafe_allow_html=True)
st.markdown("## Inference Process")
if 'inference_steps' not in st.session_state:
    st.session_state.inference_steps = []

for step in st.session_state.inference_steps:
    st.markdown(f"<div class='inference-step'>", unsafe_allow_html=True)
    st.markdown(f"**Step {step['step']}**")
    st.markdown(f"Action: {step['action']}")
    st.markdown(f"Thought: {step['thought']}")
    if 'output' in step:
        st.markdown(f"Output: {step['output']}")
    st.markdown("</div>", unsafe_allow_html=True)

# モックの推論ステップ（デモ用）
if not st.session_state.inference_steps:
    mock_steps = [
        {"step": "1", "action": "Analyzing query", "thought": "Identifying key concepts and required information"},
        {"step": "2", "action": "Planning search strategy", "thought": "Determining optimal keywords and sources"},
        {"step": "3", "action": "Executing web search", "thought": "Gathering relevant data from multiple sources",
         "output": mock_web_search("AI algorithms")},
        {"step": "4", "action": "Processing information", "thought": "Synthesizing collected data and forming conclusions"},
        {"step": "5", "action": "Generating response", "thought": "Structuring comprehensive answer with technical details"}
    ]
    for step in mock_steps:
        st.session_state.inference_steps.append(step)
        st.markdown(f"<div class='inference-step'>", unsafe_allow_html=True)
        st.markdown(f"**Step {step['step']}**")
        st.markdown(f"Action: {step['action']}")
        st.markdown(f"Thought: {step['thought']}")
        if 'output' in step:
            st.markdown(f"Output: {step['output']}")
        st.markdown("</div>", unsafe_allow_html=True)
        time.sleep(0.5)  # アニメーション効果のための遅延

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

if 'model' not in st.session_state:
    st.warning("Please initialize the agent with a valid API key.")
