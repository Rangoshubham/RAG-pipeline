import streamlit as st
import time
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

import config
import vector_store_manager
import document_processor
import rag_handler

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 RAG Assistant for Your Technical Documents")

@st.cache_resource
def initialize_llm():
    """Initializes and returns the Generative AI model, cached for efficiency."""
    load_dotenv()
    
    # Fixed typo: GROQ instead of GORQ
    groq_api_key = os.getenv("GROQ_API_KEY") 
    
    if not groq_api_key:
        st.error("🔑 GROQ_API_KEY not found in environment variables.")
        return None
    
    return ChatGroq(
        model=config.LLM_MODEL_NAME,
        api_key=groq_api_key,  # Use 'api_key' here
        temperature=0,
        max_retries=2,
    )

# --- Initialization ---
llm = initialize_llm()
os.makedirs(config.UPLOAD_DIRECTORY, exist_ok=True)

# --- Main App Logic ---

# 1. Show upload screen if no file is processed yet
if "file_path" not in st.session_state:
    st.header("Step 1: Upload Your Document")
    st.info("Please upload a PDF file to begin the chat.")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file to chat with", type="pdf", label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        file_path = os.path.join(config.UPLOAD_DIRECTORY, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Store file info in session state
        st.session_state.file_path = file_path
        st.session_state.file_name = uploaded_file.name
        
        # Rerun the app to move to the chat interface
        st.rerun()

# 2. Show chat interface if a file has been uploaded and processed
else:
    file_path = st.session_state.file_path
    file_name = st.session_state.file_name

    # --- Sidebar to manage documents ---
    with st.sidebar:
        st.header("Manage Document")
        st.info(f"Currently chatting with:\n**{file_name}**")
        
        if st.button("Upload Another Document"):
            # Clear session state to go back to the upload screen
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # --- Main Chat Interface ---
    st.header(f"Step 2: Chat with {file_name}")
    
    # Create an expander for logs
    log_expander = st.expander("Processing Logs", expanded=True)
    log_container = log_expander.container()

    def ui_logger(message):
        """A logger that writes to the UI and the console."""
        log_container.write(message)
        print(message)

    # Load vector store for the current file
    with st.spinner(f"Processing '{file_name}'... This may take a moment on first upload."):
        vectordb = vector_store_manager.load_or_create_vector_store(file_path, logger=ui_logger)
        full_text = document_processor.get_full_text(file_path, logger=ui_logger)
    
    st.success(f"Ready to chat with **{file_name}**!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input and generate response
    if prompt := st.chat_input(f"Ask a question about {file_name}"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, sources = rag_handler.get_rag_response(prompt, vectordb, llm, full_text, logger=ui_logger)
                st.markdown(response)
        
        # Store the assistant's response in session state
        st.session_state.messages.append({"role": "assistant", "content": response})
