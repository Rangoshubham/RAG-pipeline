# --- Configuration File ---

# File and Directory Paths
UPLOAD_DIRECTORY = "uploads"
VECTOR_STORE_BASE_DIR = "vector_stores"
METADATA_FILENAME = "metadata.json"

# Google Generative AI Models
LLM_MODEL_NAME = "llama-3.3-70b-versatile" # Assuming you are using Groq for generation

# Hugging Face Embedding Model (High-performance open-source option)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5" 

# Text Splitting Parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Vector Store and Retrieval Parameters
SIMILARITY_SEARCH_K = 3 

# Rate Limiting for Embeddings (No longer need the delay for local HF models!)
EMBEDDING_BATCH_SIZE = 100 # You can increase this since it's local memory now