import streamlit as st
import google.generativeai as genai
import json
import time

# Streamlit page config
st.set_page_config(page_title="Gemini Agent", layout="wide")

# Custom CSS for a more geeky look
st.markdown("""
<style>
    body {
        color: #00FF00;
        background-color: #000000;
        font-family: 'Courier New', monospace;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #00FF00;
    }
    .chat-message.user {
        background-color: #001100;
    }
    .chat-message.bot {
        background-color: #002200;
    }
    .inference-log {
        background-color: #001100;
        border: 1px solid #00FF00;
        border-radius: 0.5rem;
        padding: 1rem;
        font-size: 0.8rem;
        height: 600px;
        overflow-y: auto;
    }
    .sidebar .stButton>button {
        width: 100%;
        background-color: #001100;
        color: #00FF00;
    }
    .stTextInput>div>div>input {
        color: #00FF00;
        background-color: #001100;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'inference_log' not in st.session_state:
    st.session_state.inference_log = []

# Gemini system prompt
SYSTEM_PROMPT = """
You are an advanced AI agent. Follow these steps to answer user queries:
1. Analyze the user's question and formulate a search strategy.
2. Use web search to gather relevant information.
3. Process and synthesize the collected data.
4. Generate a comprehensive answer with citations.
5. Output detailed reasoning steps in JSON format.

For each step, output in this JSON format:
{
    "step": "step_number",
    "process": "detailed_description_of_the_current_process",
    "analysis": "in-depth_analysis_of_the_current_step",
    "next_action": "description_of_the_next_planned_action"
}
"""

# Mock functions for web search and content fetching
def search_web(query):
    time.sleep(1)  # Simulate API call
    return f"[MOCK] Search results for: {query}"

def fetch_content(url):
    time.sleep(1)  # Simulate API call
    return f"[MOCK] Content from: {url}"

# Initialize Gemini model
@st.cache_resource
def init_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-pro')

# Main chat interface
chat_col, inference_col = st.columns([3, 2])

with chat_col:
    st.title("Gemini Agent Interface")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Enter your query"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if model is initialized
        if 'model' not in st.session_state:
            st.error("Please set up the API key first.")
        else:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in st.session_state.model.generate_content(
                    f"{SYSTEM_PROMPT}\n\nUser query: {prompt}",
                    stream=True
                ):
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # Extract and store inference steps
            for line in full_response.split('\n'):
                try:
                    step = json.loads(line)
                    if isinstance(step, dict) and 'step' in step:
                        st.session_state.inference_log.append(step)
                except json.JSONDecodeError:
                    pass

# Sidebar for API key input
with st.sidebar:
    api_key = st.text_input("Enter Google AI API Key", type="password")
    if st.button("Initialize Model"):
        if api_key:
            try:
                st.session_state.model = init_model(api_key)
                st.success("Model initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing model: {str(e)}")
        else:
            st.warning("Please enter an API key.")

# Inference log display
with inference_col:
    st.subheader("Agent Inference Log")
    inference_log = st.empty()
    
    log_content = ""
    for step in st.session_state.inference_log:
        log_content += f"Step {step['step']}:\n"
        log_content += f"Process: {step['process']}\n"
        log_content += f"Analysis: {step['analysis']}\n"
        log_content += f"Next Action: {step['next_action']}\n\n"
    
    inference_log.markdown(f'<div class="inference-log">{log_content}</div>', unsafe_allow_html=True)

# Display warning if model is not initialized
if 'model' not in st.session_state:
    st.warning("Please set up the Google AI API Key to start.")
