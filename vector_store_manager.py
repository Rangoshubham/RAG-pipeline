import os
import hashlib
import json
import shutil

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # Updated Import

import config
import document_processor

def get_file_hash(filepath):
    """Calculates the MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_batch(iterable, batch_size):
    """Helper function to yield successive n-sized chunks from an iterable."""
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]

def load_or_create_vector_store(pdf_path, logger=print):
    """
    Loads a vector store for a given PDF. If the store doesn't exist or the PDF
    has changed, it creates a new one. Uses a provided logger for output.
    """
    # Updated embedding initialization
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'local_files_only': True}  # This stops it from checking the internet!
    )
    
    # Get a unique directory name from the PDF's hash
    pdf_hash = get_file_hash(pdf_path)
    persist_directory = os.path.join(config.VECTOR_STORE_BASE_DIR, pdf_hash)

    # Check if the vector store already exists
    if os.path.exists(persist_directory):
        logger(f"--- Loading existing vector store for {os.path.basename(pdf_path)} ---")
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Create a new vector store
    logger(f"--- Creating new vector store for {os.path.basename(pdf_path)} ---")
    
    # Ensure the base directory for vector stores exists
    os.makedirs(persist_directory, exist_ok=True)
    
    chunks = document_processor.load_and_split_pdf(pdf_path, logger=logger)
    
    logger("--- Creating embeddings and storing in ChromaDB... ---")
    vectordb = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    
    # Process chunks in batches (rate limit delay removed for local models)
    for i, batch in enumerate(get_batch(chunks, config.EMBEDDING_BATCH_SIZE)):
        logger(f"Processing batch {i+1}/{ (len(chunks) // config.EMBEDDING_BATCH_SIZE) + 1 }...")
        vectordb.add_documents(batch)

    # Save metadata
    metadata_path = os.path.join(persist_directory, config.METADATA_FILENAME)
    current_pdf_hash = get_file_hash(pdf_path)
    with open(metadata_path, 'w') as f:
        json.dump({'hash': current_pdf_hash}, f)
        
    logger("Successfully created and saved the vector store.\n")
    return vectordb