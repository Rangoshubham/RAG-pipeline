import config

def get_query_intent(query: str, llm, logger=print) -> str:
    """
    Uses an LLM call to classify the user's query intent.
    """
    logger("--- Classifying user intent ---")
    
    classifier_prompt = f"""Your task is to classify the user's query into one of two categories: 'general_query' or 'specific_question'.
- 'general_query': Use for questions asking for a summary, overview, main points, or the overall purpose of the document.
- 'specific_question': Use for questions asking about a specific detail, definition, concept, or fact within the document.

Do not answer the question. Only provide the category label.

User Query: "{query}"
Category:"""
    
    response = llm.invoke(classifier_prompt)
    intent = response.content.strip().lower()

    # Clean up the response to get only the label
    if 'general_query' in intent:
        final_intent = 'general_query'
    else:
        final_intent = 'specific_question' # Default to specific if unsure
        
    logger(f"Detected intent: {final_intent}")
    return final_intent

def get_rag_response(query, vectordb, llm, full_text, logger=print):
    """
    Determines user intent via an LLM call and provides a response.
    - For specific questions, uses RAG to find relevant chunks.
    - For general requests, uses the full document text.
    """
    intent = get_query_intent(query, llm, logger)

    if intent == 'general_query':
        template = """
You are a helpful assistant. Answer the user's question based on the full content of the document provided below.

Document:
{context}

Question:
{question}

Answer:
"""
        prompt = template.format(context=full_text, question=query)
        logger("--- Generating answer from full document based on general intent ---")
        response = llm.invoke(prompt)
        return response.content, [] # No specific sources for a general query

    else:
        # Standard RAG for specific questions
        logger(f"\nSearching for relevant documents for specific question: '{query}'")
        retrieved_docs = vectordb.similarity_search(query, k=config.SIMILARITY_SEARCH_K)
        logger(f"Found {len(retrieved_docs)} relevant document chunks.\n")

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        template = """
You are a helpful assistant. Your primary goal is to answer the user's question based on the provided context.

1. First, carefully read the context below and try to answer the question using only this information.
2. If the context is sufficient, provide a direct answer based on it.
3. If the context does not contain the answer or is insufficient, use your general knowledge to answer the question.
4. **Crucially**, if you use your general knowledge, you **must** add the following disclaimer at the end of your answer: "Disclaimer: This answer was generated from my general knowledge as the provided document context was not sufficient."

Context:
{context}

Question:
{question}

Answer:
"""
        prompt = template.format(context=context, question=query)
        
        logger("--- Generating final answer from context chunks ---")
        response = llm.invoke(prompt)
        logger("--- Final answer generated ---")
            
        return response.content, retrieved_docs