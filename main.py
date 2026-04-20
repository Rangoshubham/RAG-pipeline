from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

import config
import vector_store_manager
import rag_handler

def initialize_llm():
    """Initializes and returns the Generative AI model, cached for efficiency."""
    load_dotenv()
    
    # Fixed typo: GROQ instead of GORQ
    groq_api_key = os.getenv("GROQ_API_KEY") 
    
    if not groq_api_key:
        print("🔑 GROQ_API_KEY not found in environment variables.")
        return None
    
    return ChatGroq(
        model=config.LLM_MODEL_NAME,
        api_key=groq_api_key,  # Use 'api_key' here
        temperature=0,
        max_retries=2,
    )

def main():
    """Main function to run the RAG application."""
    llm = initialize_llm()
    vectordb = vector_store_manager.load_or_create_vector_store()
    
    # Start the interactive Q&A loop
    print("--- Ready to answer questions from the PDF. Type 'exit' to quit. ---")
    while True:
        user_query = input("\nPlease enter your question: ")
        if user_query.lower() == 'exit':
            print("Exiting application.")
            break
        if user_query:
            rag_handler.get_rag_response(user_query, vectordb, llm)

if __name__ == "__main__":
    main()
