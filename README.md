# RAG with Long-Term Memory

A Retrieval-Augmented Generation system with persistent long-term memory capabilities.

## Features

- Vector-based document retrieval using FAISS/Chroma
- LLM integration (Azure OpenAI GPT-4)
- Persistent long-term memory with decay and consolidation
- FastAPI backend
- Streamlit web interface

## Setup

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install: `pip install -r requirements.txt`
5. Configure: Ensure `.env` file has your Azure OpenAI API keys
6. Run API: `uvicorn src.api.main:app --reload`
7. Run UI: `streamlit run src.ui/streamlit_app.py`

## Project Structure

```
rag_ltm/
├── src/
│   ├── retrieval/      # Vector store and document retrieval
│   ├── generation/     # LLM integration
│   ├── memory/         # Long-term memory system
│   ├── pipeline/       # RAG pipeline orchestration
│   ├── api/           # FastAPI backend
│   └── ui/            # Streamlit frontend
├── tests/             # Test suite
├── configs/           # Configuration files
├── data/              # Data storage
└── notebooks/         # Exploration notebooks
```

## License

MIT
