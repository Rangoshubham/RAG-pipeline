# 📄 RAG Assistant for Technical PDFs

![Python](https://img.shields.io/badge/Python-3.14%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Local_Embeddings-yellow)
![Groq](https://img.shields.io/badge/Groq-LLM-orange)

An interactive Retrieval-Augmented Generation (RAG) app for technical PDF documents.

You can upload a PDF, build a local vector index with Hugging Face embeddings, and chat with the content through a Streamlit UI powered by Groq.

## ✨ Features

- Local embedding generation with `BAAI/bge-small-en-v1.5`.
- Persistent Chroma vector stores keyed by PDF hash.
- Intent-based routing:
- `general_query` uses full-document context.
- `specific_question` uses similarity search retrieval.
- Streamlit chat interface with logs and per-document session state.

## 🛠️ Tech Stack

- UI: Streamlit
- Orchestration and prompts: LangChain
- LLM inference: Groq (`llama-3.3-70b-versatile` in [config.py](config.py))
- Embeddings: `langchain-huggingface` + `sentence-transformers`
- Vector database: ChromaDB
- PDF parsing: PyMuPDF

## 📁 Project Structure

```text
RAG-pipeline-main/
|-- app.py
|-- main.py
|-- config.py
|-- document_processor.py
|-- rag_handler.py
|-- vector_store_manager.py
|-- test.py
|-- test_ui.py
|-- requirements.txt
|-- pyproject.toml
|-- README.md
|-- uploads/
`-- vector_stores/
```

## ✅ Requirements

- Python 3.14 or newer (from [pyproject.toml](pyproject.toml)).
- A Groq API key.

## ⚙️ Setup

1. Clone the repository.

```bash
git clone https://github.com/Rangoshubham/RAG-pipeline.git
cd RAG-pipeline-main
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_actual_api_key_here
```

## 🚀 Run the App

1. (Recommended first run) Pre-download and verify embedding model cache:

```bash
python test.py
```

2. Start Streamlit:

```bash
streamlit run app.py
```

3. Open the local URL shown in the terminal, upload a PDF, and start chatting.

## 🧠 How Indexing Works

- Uploaded files are saved in `uploads/`.
- Each PDF is hashed (MD5).
- Chroma data is persisted under `vector_stores/<pdf_hash>/`.
- Re-uploading an unchanged PDF reuses the existing vector store.

## ☁️ Deployment Notes

For cloud deployment (for example, Streamlit Community Cloud):

- Set `GROQ_API_KEY` in the platform secrets.
- In [vector_store_manager.py](vector_store_manager.py), change `local_files_only` to `False` (or remove it) so the server can download embedding models at startup.

## 🩺 Troubleshooting

- Missing API key:
- Confirm `.env` exists and contains `GROQ_API_KEY`.
- Model load error:
- The app currently uses local-only model loading (`local_files_only=True`), so run `python test.py` first to populate cache.
- Permission issues on Windows:
- Ensure you have write access to `uploads/` and `vector_stores/`.

## 📝 Notes

- `requirements.txt` and `pyproject.toml` both list dependencies.
- The Streamlit entry point is `app.py`.

## 🤝 Contributing

Contributions and issues are welcome.

## 📄 License

This project is open-source and available under the MIT License.