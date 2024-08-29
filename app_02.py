import streamlit as st
import google.generativeai as genai
from typing import Dict, Any
import json

# Streamlit UI設定
st.set_page_config(page_title="Geminiエージェントチャットボット", layout="wide")

# カスタムCSS
st.markdown("""
<style>
    .main-content { max-width: 1200px; margin: auto; padding: 20px; }
    .chat-container { border: 1px solid #ddd; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
    .user-message { background-color: #e6f3ff; padding: 10px; border-radius: 10px; margin-bottom: 10px; }
    .assistant-message { background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin-bottom: 10px; }
    .reasoning-container { border-left: 3px solid #4CAF50; padding-left: 20px; margin-top: 20px; }
    .stTextInput>div>div>input { min-height: 50px; }
</style>
""", unsafe_allow_html=True)

# セッション状態の初期化
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'reasoning_history' not in st.session_state:
    st.session_state.reasoning_history = []

# モックツールの定義
def search_web(query: str) -> str:
    return f"Web検索結果: {query}に関する情報"

def fetch_page(url: str) -> str:
    return f"{url}の内容"

tools = [
    {
        "name": "search_web",
        "description": "Web検索を行い、結果を返します。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "検索クエリ"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_page",
        "description": "指定されたURLのWebページの内容を取得します。",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "取得するWebページのURL"}
            },
            "required": ["url"]
        }
    }
]

# Geminiモデルの設定と初期化
def initialize_model(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-pro')

# エージェントの実行とストリーミング
def run_agent(model, user_input: str) -> None:
    system_prompt = """
    あなたは高度なAIアシスタントです。ユーザーの質問に答えるために、以下のステップを踏んでください：
    1. ユーザーの質問を理解し、適切な検索戦略を立てる。
    2. 必要に応じて、提供されているツール（Web検索、ページ取得）を使用して情報を収集する。
    3. 収集した情報を分析し、ユーザーの質問に対する回答を作成する。
    4. 回答の根拠となる情報源を明記する。
    5. 推論過程を詳細に説明し、各ステップで何を考え、どのような行動をとったかを明確にする。

    回答は以下の形式で出力してください：
    ```json
    {
        "思考": "現在の思考プロセスの説明",
        "行動": {
            "tool": "使用するツール名",
            "params": {
                "param1": "値1",
                "param2": "値2"
            }
        },
        "観察": "ツールの実行結果や観察した内容",
        "結論": "最終的な回答や結論"
    }
    ```

    ユーザーの質問に答えるまで、このプロセスを繰り返してください。
    """

    prompt = f"{system_prompt}\n\nユーザーの質問: {user_input}\n\n回答:"
    
    reasoning_placeholder = st.empty()
    response_placeholder = st.empty()
    full_response = ""
    reasoning_steps = []

    for chunk in model.generate_content(prompt, tools=tools, stream=True):
        if chunk.text:
            full_response += chunk.text
            try:
                response_json = json.loads(full_response)
                display_reasoning(response_json, reasoning_placeholder)
                reasoning_steps.append(response_json)
            except json.JSONDecodeError:
                pass

    # ツールの実行（実際のAPIコールの代わりにモック）
    if reasoning_steps and "行動" in reasoning_steps[-1] and "tool" in reasoning_steps[-1]["行動"]:
        tool_name = reasoning_steps[-1]["行動"]["tool"]
        if tool_name == "search_web":
            result = search_web(reasoning_steps[-1]["行動"]["params"]["query"])
        elif tool_name == "fetch_page":
            result = fetch_page(reasoning_steps[-1]["行動"]["params"]["url"])
        else:
            result = "未知のツールが呼び出されました。"
        
        reasoning_steps[-1]["観察"] = result
        display_reasoning(reasoning_steps[-1], reasoning_placeholder)

    # 最終的な結論を表示
    final_conclusion = reasoning_steps[-1].get("結論", "結論が見つかりませんでした。")
    response_placeholder.markdown(f"**回答:** {final_conclusion}")

    # チャット履歴と推論履歴に追加
    st.session_state.chat_history.append({"role": "assistant", "content": final_conclusion})
    st.session_state.reasoning_history.append(reasoning_steps)

def display_reasoning(response: Dict[str, Any], placeholder: st.empty) -> None:
    markdown = ""
    if "思考" in response:
        markdown += f"**思考:** {response['思考']}\n\n"
    if "行動" in response:
        markdown += f"**行動:** {json.dumps(response['行動'], ensure_ascii=False, indent=2)}\n\n"
    if "観察" in response:
        markdown += f"**観察:** {response['観察']}\n\n"
    if "結論" in response:
        markdown += f"**結論:** {response['結論']}\n\n"
    placeholder.markdown(markdown)

# メイン関数
def main():
    st.title("Geminiエージェントチャットボット")

    # 2カラムレイアウト
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<div class='main-content'>", unsafe_allow_html=True)
        
        # APIキーの入力
        api_key = st.text_input("Google API Keyを入力してください", type="password")

        if api_key:
            model = initialize_model(api_key)

            # チャット履歴の表示
            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"<div class='user-message'>👤 {message['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='assistant-message'>🤖 {message['content']}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ユーザー入力
            user_input = st.text_input("質問を入力してください", key="user_input")
            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.markdown(f"<div class='user-message'>👤 {user_input}</div>", unsafe_allow_html=True)

                with st.spinner("回答を生成中..."):
                    run_agent(model, user_input)

        else:
            st.warning("Google API Keyを入力してください")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='reasoning-container'>", unsafe_allow_html=True)
        st.subheader("推論過程")
        if st.session_state.reasoning_history:
            for step in st.session_state.reasoning_history[-1]:
                display_reasoning(step, st.empty())
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
