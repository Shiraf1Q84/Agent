import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from tools.search_ddg import search_ddg
from tools.fetch_page import fetch_page
import os

# AgentのSystem Promptの作成
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

# 検索キーワード計画用のプロンプト
SEARCH_PLANNING_PROMPT = """
ユーザーの質問に基づいて、適切な検索キーワードを計画してください。
複数のキーワードや検索オプションを検討し、最も効果的な検索戦略を提案してください。
ユーザーの質問: {question}
検索キーワード案:
"""

def create_agent(api_key):
    tools = [search_ddg, fetch_page]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", CUSTOM_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", api_key=api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True
    )

def plan_search_keywords(question, api_key):
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", api_key=api_key)
    response = llm.invoke(SEARCH_PLANNING_PROMPT.format(question=question))
    return response.content

# Streamlit UI
st.title("インターネットで調べ物をしてくれるエージェント")

# セッション状態の初期化
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# OpenAI API Keyの入力
api_key = st.text_input("OpenAI API Keyを入力してください", type="password")

if api_key:
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

        # 検索キーワードの計画
        with st.spinner("検索キーワードを計画中..."):
            search_keywords = plan_search_keywords(user_input, api_key)
        st.write("計画された検索キーワード:", search_keywords)

        # エージェントの実行
        with st.spinner("回答を生成中..."):
            response = st.session_state.agent.invoke(
                {"input": f"検索キーワード: {search_keywords}\n質問: {user_input}"}
            )

        # エージェントの回答を表示
        with st.chat_message("assistant"):
            st.write(response["output"])
        st.session_state.chat_history.append({"role": "assistant", "content": response["output"]})

        # 推論過程の表示
        st.subheader("推論過程")
        for step in response["intermediate_steps"]:
            st.write(f"行動: {step[0]}")
            st.write(f"結果: {step[1]}")
            st.write("---")
else:
    st.warning("OpenAI API Keyを入力してください")
