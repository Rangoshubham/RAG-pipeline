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
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Enhanced UI ---
st.markdown("""
<style>
    /* Main app styling - Soft gradient background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Chat container styling - Clean white card */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: #ffffff;
        border-radius: 20px;
        margin-top: 1rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
    }
    
    /* Title styling - Vibrant gradient */
    h1 {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
        animation: fadeIn 0.8s ease-in;
    }
    
    /* Header styling */
    h2, h3 {
        color: #2d3748;
        font-weight: 700;
    }
    
    /* All text elements - Dark gray for readability */
    p, span, div, label, li {
        color: #2d3748;
    }
    
    /* Info box styling - Purple accent */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid #667eea;
        background-color: #f0f4ff;
        animation: slideIn 0.5s ease-out;
    }
    
    /* Button styling - Purple gradient */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader styling - Soft purple background */
    .stFileUploader {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 2.5rem;
        background: linear-gradient(135deg, #f0f4ff 0%, #e9ecff 100%);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #e9ecff 0%, #dde1ff 100%);
        transform: scale(1.02);
    }
    
    .stFileUploader label {
        color: #2d3748;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Chat message styling - Distinct backgrounds */
    .stChatMessage {
        border-radius: 15px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        animation: messageSlide 0.4s ease-out;
    }
    
    /* User message - Light purple */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #f0f4ff 0%, #e9ecff 100%);
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
    }
    
    /* Assistant message - Light gray */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-left: 4px solid #48bb78;
        box-shadow: 0 2px 8px rgba(72, 187, 120, 0.1);
    }
    
    /* Chat message text */
    .stChatMessage p, .stChatMessage div, .stChatMessage span, .stChatMessage li {
        color: #2d3748;
        line-height: 1.6;
    }
    
    /* Log container styling - Clean white box */
    [data-testid="stVerticalBlock"] > [data-testid="stContainer"] {
        border-radius: 12px;
        background: #ffffff;
        border: 2px solid #e2e8f0;
        padding: 1rem;
    }
    
    /* Log text color - Red */
    .log-text {
        color: #dc2626 !important;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Sidebar styling - Soft gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #2d3748;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f0f4ff 0%, #e9ecff 100%);
        color: #2d3748;
        font-weight: 600;
        border-radius: 10px;
        border: 1px solid #667eea;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 1rem;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #667eea;
        font-weight: 800;
        font-size: 1.75rem;
    }
    
    [data-testid="stMetricLabel"] {
        color: #4a5568;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #2d3748;
    }
    
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #2d3748;
    }
    
    /* Strong/Bold text */
    strong, b {
        color: #1a202c;
        font-weight: 700;
    }
    
    /* Code blocks */
    code {
        background: #f7fafc;
        color: #2d3748;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes messageSlide {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #667eea;
    }
    
    .stSpinner p {
        color: #2d3748;
    }
    
    /* Success message styling - Green accent */
    .stSuccess {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border-left: 5px solid #48bb78;
        border-radius: 12px;
        animation: slideIn 0.5s ease-out;
    }
    
    .stSuccess p {
        color: #22543d;
        font-weight: 600;
    }
    
    /* Error message styling - Red accent */
    .stError {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border-left: 5px solid #f56565;
        border-radius: 12px;
    }
    
    .stError p {
        color: #742a2a;
        font-weight: 600;
    }
    
    /* Input box styling */
    .stChatInputContainer {
        border-radius: 15px;
        border: 2px solid #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        background: #ffffff;
    }
    
    .stChatInputContainer textarea {
        color: #2d3748;
        font-size: 1rem;
    }
    
    /* Footer styling - Purple gradient */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        text-align: center;
        padding: 15px 0;
        font-size: 14px;
        font-weight: 600;
        z-index: 999;
        box-shadow: 0 -4px 20px rgba(102, 126, 234, 0.3);
    }
    
    .footer strong {
        color: #ffffff;
        font-weight: 700;
    }
    
    /* Adjust main container to account for footer */
    .main .block-container {
        padding-bottom: 90px;
    }
    
    /* Document info card styling */
    .doc-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #e9ecff 100%);
        padding: 1.25rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
    }
    
    .doc-card p {
        color: #2d3748;
        margin: 0;
    }
    
    .doc-label {
        color: #4a5568;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .doc-name {
        color: #1a202c;
        font-weight: 700;
        font-size: 1.05rem;
        margin-top: 0.5rem;
    }
    
    /* Horizontal rule styling */
    hr {
        border: none;
        border-top: 2px solid #e2e8f0;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Developed by <strong>Shubham Kumar</strong> and <strong>Anurag Shakya</strong> | Galgotias University
</div>
""", unsafe_allow_html=True)

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
    st.title("📄 RAG Assistant for Your Technical Documents")
    
    # Create centered columns for better layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🚀 Get Started")
        st.info("💡 Upload a PDF document and start asking questions powered by AI!")
        
        # Feature highlights
        with st.expander("✨ Features", expanded=True):
            st.markdown("""
            - 🤖 **AI-Powered Answers**: Get intelligent responses from your documents
            - 🔍 **Context-Aware**: Understands the full context of your questions
            - ⚡ **Fast Processing**: Optimized vector search for quick responses
            - 💬 **Natural Conversation**: Chat naturally with your documents
            """)
        
        st.markdown("---")
        st.markdown("#### 📤 Upload Your Document")
        
        uploaded_file = st.file_uploader(
            "Drop your PDF here or click to browse",
            type="pdf",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            with st.spinner("📥 Uploading your document..."):
                # Save the uploaded file to a temporary location
                file_path = os.path.join(config.UPLOAD_DIRECTORY, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Show success message with animation
                st.success(f"✅ Successfully uploaded **{uploaded_file.name}**!")
                time.sleep(0.5)
                
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
        st.markdown("### 📋 Document Manager")
        
        # Document info card with custom styling
        st.markdown(f"""
        <div class="doc-card">
            <p class="doc-label">📄 Current Document</p>
            <p class="doc-name">{file_name}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 New Doc", use_container_width=True):
                # Clear session state to go back to the upload screen
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.logs = []
                st.success("Chat cleared! 🧹")
                time.sleep(0.5)
                st.rerun()
        
        st.markdown("---")
        
        # Statistics section
        if "messages" in st.session_state:
            total_messages = len(st.session_state.messages)
            user_messages = sum(1 for m in st.session_state.messages if m["role"] == "user")
            
            st.markdown("### 📊 Chat Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Messages", total_messages)
            with col2:
                st.metric("Questions", user_messages)
        
        st.markdown("---")
        
        # Help section
        with st.expander("💡 Tips for Better Results"):
            st.markdown("""
            - Ask specific questions
            - Reference document sections
            - Request summaries or explanations
            - Ask follow-up questions
            """)

    # --- Main Chat Interface ---
    st.title(f"💬 Chat with {file_name}")

    # Create a two-column layout with adjusted ratios
    chat_col, log_col = st.columns([2.5, 1.5])
    
    with log_col:
        st.markdown("### 📝 Processing Logs")
        log_placeholder = st.empty()

    def ui_logger(message):
        """A logger that writes to the UI and the console."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # Add to session state
        if "logs" not in st.session_state:
            st.session_state.logs = []
        st.session_state.logs.append(log_entry)
        
        # Update log display dynamically
        with log_placeholder.container(height=500, border=True):
            for log in st.session_state.logs:
                st.markdown(f'<p class="log-text">{log}</p>', unsafe_allow_html=True)
        
        # Print to console
        print(message)

    with chat_col:
        # Load vector store for the current file
        with st.spinner(f"🔄 Processing '{file_name}'... This may take a moment on first upload."):
            vectordb = vector_store_manager.load_or_create_vector_store(file_path, logger=ui_logger)
            full_text = document_processor.get_full_text(file_path, logger=ui_logger)
        
        st.success(f"✅ Ready to chat with **{file_name}**!")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Show welcome message if no messages yet
        if len(st.session_state.messages) == 0:
            with st.chat_message("assistant"):
                st.markdown(f"""
                👋 Hello! I'm your AI assistant for **{file_name}**.
                
                I can help you:
                - Answer questions about the document
                - Summarize sections or the entire document
                - Explain complex concepts
                - Find specific information
                
                What would you like to know?
                """)

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle user input and generate response
        if prompt := st.chat_input(f"Ask anything about {file_name}..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🤔 Analyzing document..."):
                    response, sources = rag_handler.get_rag_response(
                        prompt, vectordb, llm, full_text, logger=ui_logger
                    )
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to update logs display
            st.rerun()