import streamlit as st
import os
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler
from tools.search_ddg import search_ddg
from tools.fetch_page import fetch_page

CUSTOM_SYSTEM_PROMPT = """
あなたの役割
あなたの役割はuserの入力する質問に対して、インターネットでWebページを調査をし、回答することです。
あなたが従わなければいけないルール
回答はできるだけ短く、要約して回答してください
文章が長くなる場合は改行して見やすくしてください
回答の最後に改行した後、参照したページのURLを記載してください
"""

def create_agent():
    tools = [search_ddg, fetch_page]
    prompt = ChatPromptTemplate.from_messages([
        ("system", CUSTOM_SYSTEM_PROMPT),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    llm = ChatOpenAI(temperature=0., model_name="gpt-3.5-turbo")
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

st.title("インターネットで調べ物をしてくれるエージェント")

api_key = st.text_input("OpenAI API Keyを入力してください", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    query = st.text_input("質問を入力してください")

    if st.button("実行"):
        agent = create_agent()
        
        st.write("推論過程:")
        output_container = st.empty()
        
        def process_output(output):
            return output.replace("Human:", "ユーザー:").replace("AI:", "アシスタント:")

        callback = StreamlitCallbackHandler(output_container, max_thought_containers=10, expand_new_thoughts=True, collapse_completed_thoughts=False)
        
        with st.spinner("エージェントが作業中..."):
            response = agent.invoke(
                {"input": query},
                config={"callbacks": [callback]}
            )
        
        st.write("最終回答:")
        st.write(process_output(response["output"]))

else:
    st.warning("OpenAI API Keyを入力してください")