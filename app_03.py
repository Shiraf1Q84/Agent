import streamlit as st
import google.generativeai as genai
import time

# Streamlit page config
st.set_page_config(page_title="Gemini Agent", layout="wide")

# Custom CSS for a clean, professional look
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .chat-message.user {
        background-color: #f0f0f0;
    }
    .chat-message.bot {
        background-color: #ffffff;
    }
    .inference-log {
        background-color: #f9f9f9;
        border-radius: 0.5rem;
        padding: 1rem;
        height: 600px;
        overflow-y: auto;
    }
    .sidebar .stButton>button {
        width: 100%;
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
You are an advanced AI assistant designed to provide comprehensive and detailed responses. When answering queries, please follow these guidelines:

1. Thoroughly analyze the user's question, considering all possible aspects and interpretations.
2. Provide a detailed and extensive answer, aiming for at least 500 words.
3. Structure your response with clear sections, using headings where appropriate.
4. Include relevant examples, analogies, or case studies to illustrate your points.
5. When applicable, present multiple perspectives or approaches to the topic.
6. Cite authoritative sources or reference well-known theories and concepts.
7. Conclude with a summary of key points and, if appropriate, suggestions for further exploration of the topic.

Throughout your response, maintain a professional and informative tone. Your goal is to provide a response that is not only accurate but also educational and thought-provoking.
"""

# Mock functions for web search and content fetching
def search_web(query):
    time.sleep(1)  # Simulate API call
    return f"[MOCK] Extensive search results for: {query}"

def fetch_content(url):
    time.sleep(1)  # Simulate API call
    return f"[MOCK] Detailed content from: {url}"

# Initialize Gemini model
@st.cache_resource
def init_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash-latest')

# Main chat interface
chat_col, inference_col = st.columns([2, 1])

with chat_col:
    st.title("Gemini AI Assistant")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask your question here"):
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
                    f"{SYSTEM_PROMPT}\n\nUser question: {prompt}",
                    stream=True
                ):
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.inference_log.append(f"Processed query: {prompt}\nGenerated response length: {len(full_response)} characters")

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Google AI API Key", type="password")
    if st.button("Initialize Assistant"):
        if api_key:
            try:
                st.session_state.model = init_model(api_key)
                st.success("AI Assistant initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing assistant: {str(e)}")
        else:
            st.warning("Please enter an API key.")

# Inference log display
with inference_col:
    st.subheader("Processing Log")
    inference_log = st.empty()
    
    log_content = "\n".join(st.session_state.inference_log)
    inference_log.markdown(f'<div class="inference-log">{log_content}</div>', unsafe_allow_html=True)

# Display warning if model is not initialized
if 'model' not in st.session_state:
    st.warning("Please set up the Google AI API Key to start.")
