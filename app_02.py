import streamlit as st
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Geminiのシステムプロンプト
CUSTOM_SYSTEM_PROMPT = """
あなたの役割は以下の通りです：
1. ユーザーの質問を理解し、適切な検索キーワードを計画する。
2. 計画したキーワードを使用してWeb検索を行う。
3. 検索結果から関連情報を抽出し、ユーザーの質問に答える。
4. 必要に応じて追加の情報を検索する。
5. 最終的な回答を作成し、使用した情報源を引用する。
回答する際は以下のルールに従ってください：
- 回答は簡潔にまとめ、必要に応じて箇条書きを使用する。
- 長文になる場合は適切に改行を入れて読みやすくする。
- 回答の最後に、参照したWebページのURLを記載する。
- 推論過程を詳細に説明し、各ステップで何を考え、どのような行動をとったかを明確にする。
"""

def search_ddg(query):
    url = f"https://duckduckgo.com/html/?q={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    for result in soup.find_all('div', class_='result__body'):
        title = result.find('a', class_='result__a').text
        snippet = result.find('a', class_='result__snippet').text
        link = result.find('a', class_='result__a')['href']
        results.append({'title': title, 'snippet': snippet, 'link': link})
    return results[:5]  # 上位5件の結果を返す

def fetch_page(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()[:1000]  # 最初の1000文字を返す
    except Exception as e:
        return f"Error fetching page: {str(e)}"

def create_agent(api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        return model
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise

def plan_search_keywords(question, model):
    try:
        response = model.generate_content(f"以下の質問に対する適切な検索キーワードを提案してください：\n{question}")
        return response.text
    except Exception as e:
        logger.error(f"Error planning search keywords: {str(e)}")
        raise

# Streamlit UI
st.title("Geminiを使用したWeb検索エージェント")

# セッション状態の初期化
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Google AI API Keyの入力
api_key = st.text_input("Google AI API Keyを入力してください", type="password")

if api_key:
    try:
        # APIキーが入力されたら、エージェントを作成または更新
        if 'agent' not in st.session_state or st.session_state.api_key != api_key:
            st.session_state.agent = create_agent(api_key)
            st.session_state.api_key = api_key

        # チャット履歴の表示
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # ユーザー入力
        user_input = st.chat_input("質問を入力してください")
        if user_input:
            # ユーザーの入力を表示
            with st.chat_message("user"):
                st.write(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            try:
                # 検索キーワードの計画
                with st.spinner("検索キーワードを計画中..."):
                    search_keywords = plan_search_keywords(user_input, st.session_state.agent)
                st.write("計画された検索キーワード:", search_keywords)

                # Web検索の実行
                with st.spinner("Web検索を実行中..."):
                    search_results = search_ddg(search_keywords)

                # 検索結果の表示（デバッグ用）
                st.subheader("検索結果")
                for result in search_results:
                    st.write(f"Title: {result['title']}")
                    st.write(f"Snippet: {result['snippet']}")
                    st.write(f"URL: {result['link']}")
                    st.write("---")

                # コンテンツの取得と要約
                content = ""
                for result in search_results:
                    content += fetch_page(result['link']) + "\n\n"

                # Geminiによる回答生成
                with st.spinner("回答を生成中..."):
                    prompt = f"{CUSTOM_SYSTEM_PROMPT}\n\nユーザーの質問: {user_input}\n\n検索結果:\n{content}\n\n上記の情報を基に、ユーザーの質問に答えてください。"
                    response = st.session_state.agent.generate_content(prompt)

                # エージェントの回答を表示
                with st.chat_message("assistant"):
                    st.write(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
                logger.error(f"Error during agent execution: {str(e)}")
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        logger.error(f"Error in main app flow: {str(e)}")
else:
    st.warning("Google AI API Keyを入力してください")
