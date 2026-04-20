from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

def get_full_text(pdf_path, logger=print):
    """Extracts the full text content from a PDF."""
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    return "\n".join(doc.page_content for doc in docs)

def load_and_split_pdf(pdf_path, logger=print):
    """
    Loads a PDF document and splits it into chunks.
    Uses a provided logger for output.
    """
    logger("--- Loading PDF ---")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    logger(f"PDF loaded into {len(docs)} documents (pages).\n")

    logger("--- Splitting documents into chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(docs)
    logger(f"Split the {len(docs)} pages into {len(chunks)} chunks.\n")
    return chunks