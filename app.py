import streamlit as st
import os
import networkx as nx
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from tools.search_ddg import search_ddg
from tools.fetch_page import fetch_page

# System Promptの作成
CUSTOM_SYSTEM_PROMPT = """
あなたの役割
あなたの役割はuserの入力する質問に対して、インターネットでWebページを調査をし、回答することです。
あなたが従わなければいけないルール
回答はできるだけ短く、要約して回答してください
文章が長くなる場合は改行して見やすくしてください
回答の最後に改行した後、参照したページのURLを記載してください
"""

# LangChain Agent
def create_langchain_agent():
    tools = [search_ddg, fetch_page]
    prompt = ChatPromptTemplate.from_messages([
        ("system", CUSTOM_SYSTEM_PROMPT),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    llm = ChatOpenAI(temperature=0., model_name="gpt-3.5-turbo")

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )

# NetworkX based Agent
def create_networkx_agent():
    G = nx.DiGraph()
    G.add_node("start")
    G.add_node("search")
    G.add_node("fetch")
    G.add_node("answer")
    G.add_edge("start", "search")
    G.add_edge("search", "fetch")
    G.add_edge("fetch", "answer")

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    
    def process_node(node, query):
        if node == "search":
            return search_ddg(query)
        elif node == "fetch":
            urls = [result['url'] for result in query]
            return [fetch_page(url) for url in urls[:2]]  # Fetch first 2 URLs
        elif node == "answer":
            prompt = ChatPromptTemplate.from_messages([
                ("system", CUSTOM_SYSTEM_PROMPT),
                ("user", f"Based on the following information, answer the user's question: {query}")
            ])
            return llm(prompt.format_messages(input=str(query)))
        return None

    def run_agent(query):
        current_node = "start"
        result = query
        while current_node != "answer":
            next_nodes = list(G.successors(current_node))
            if next_nodes:
                current_node = next_nodes[0]
                result = process_node(current_node, result)
        return result

    return run_agent

# Streamlit UI
st.title("インターネットで調べ物をしてくれるエージェント")

api_key = st.text_input("OpenAI API Keyを入力してください", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    agent_type = st.radio("Agentの種類を選択してください", ("LangChain Agent", "NetworkX Agent"))

    query = st.text_input("質問を入力してください")

    if st.button("実行"):
        if agent_type == "LangChain Agent":
            agent = create_langchain_agent()
            response = agent.invoke({'input': query})
            st.write("（Agentの回答）", response["output"])

        elif agent_type == "NetworkX Agent":
            agent = create_networkx_agent()
            response = agent(query)
            st.write("（Agentの回答）", response.content)

else:
    st.warning("OpenAI API Keyを入力してください")
