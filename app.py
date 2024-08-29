import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AnyMessage
from tools.search_ddg import search_ddg
from tools.fetch_page import fetch_page

# [3-1] AgentのSystem Promptの作成
CUSTOM_SYSTEM_PROMPT = """
あなたの役割
あなたの役割はuserの入力する質問に対して、インターネットでWebページを調査をし、回答することです。
あなたが従わなければいけないルール
回答はできるだけ短く、要約して回答してください
文章が長くなる場合は改行して見やすくしてください
回答の最後に改行した後、参照したページのURLを記載してください
"""

# --- LangChain Agent ---
def create_langchain_agent():
    # [1]、[2]で定義したAgentが使用可能なToolを指定します
    tools = [search_ddg, fetch_page]
    # プロンプトを与えます。ChatPromptTemplateの詳細は書籍本体の解説をご覧ください。
    # 重要な点は、最初のrole "system"に上記で定義したCUSTOM_SYSTEM_PROMPTを与え、
    # userの入力は{input}として動的に埋め込むようにしている点です
    # agent_scratchpadはAgentの動作の途中経過を格納するためのものです
    prompt = ChatPromptTemplate.from_messages([
        ("system", CUSTOM_SYSTEM_PROMPT),
        # MessagesPlaceholder(variable_name="chat_history"),  # チャットの過去履歴はなしにしておきます
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # 使用するLLMをOpenAIのGPT-4o-miniにします（GPT-4だとfechなしに動作が完了してしまう）
    llm = ChatOpenAI(temperature=0., model_name="gpt-4o-mini")

    # Agentを作成
    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # これでAgentが途中でToolを使用する様子が可視化されます
        # memory=st.session_state['memory']  # memory≒会話履歴はなしにしておきます
    )

# --- LangGraph Agent ---
def create_langgraph_agent():
    # プロンプトを定義
    prompt = ChatPromptTemplate.from_messages([
        ("system", CUSTOM_SYSTEM_PROMPT),  # [3] で定義したのと同じシステムプロンプトです
        ("user", "{messages}")
        # MessagesPlaceholder(variable_name="agent_scratchpad") # agent_scratchpadは中間生成物の格納用でしたが、LangGraphでは不要です
    ])

    # LangGraphではGraph構造で全体を処理するので、stateを変化させノードが移るタイミングで、promptを（会話やAgentの自分メモ）を進めるように定義します
    def _modify_messages(messages: list[AnyMessage]):
        return prompt.invoke({"messages": messages}).to_messages()

    # ReactAgentExecutorの準備
    tools = [search_ddg, fetch_page]
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")  # 【2024年8月2日現在では、modelにgpt-4o-miniを使用すると日本語ではうまく動作してくれません】そのため、gpt-4oを使用
    web_browsing_agent = create_react_agent(llm, tools, state_modifier=_modify_messages)
    return web_browsing_agent


# --- Streamlit UI ---
st.title("インターネットで調べ物をしてくれるエージェント")

# API Keyの入力
api_key = st.text_input("OpenAI API Keyを入力してください", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    # Agentの種類を選択
    agent_type = st.radio("Agentの種類を選択してください", ("LangChain Agent", "LangGraph Agent"))

    # 質問の入力
    query = st.text_input("質問を入力してください")

    # 実行ボタン
    if st.button("実行"):
        if agent_type == "LangChain Agent":
            agent = create_langchain_agent()
            response = agent.invoke({'input': query})
            st.write("（Agentの回答）", response["output"])

        elif agent_type == "LangGraph Agent":
            agent = create_langgraph_agent()
            messages = agent.invoke({"messages": [("user", query)]})
            for i in range(len(messages["messages"])):
                if messages["messages"][i].type == "tool":
                    pass  # toolの出力は除外
                elif messages["messages"][i].type == "human":
                    st.write("human: ", messages["messages"][i].content)
                elif messages["messages"][i].type == "ai" and len(messages["messages"][i].content) > 0:
                    # AIがtool使用の命令ではなく、文章生成をしている場合は出力
                    st.write("AI: ", messages["messages"][i].content)
else:
    st.warning("OpenAI API Keyを入力してください")