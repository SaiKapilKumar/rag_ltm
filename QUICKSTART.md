# Quick Start Guide

## Setup

1. **Navigate to the project directory**:
   ```bash
   cd rag_ltm
   ```

2. **Run the setup script** (first time only):
   ```bash
   ./setup.sh
   ```
   
   Or manually:
   ```bash
   pyenv shell 3.12.0  # or your Python 3.9+ version
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   mkdir -p data/memory data/embeddings data/documents
   ```

3. **Ensure your `.env` file** in the parent directory has Azure OpenAI credentials:
   ```
   AZURE_OPENAI_EMBEDDINGS_API_KEY=your_key
   AZURE_OPENAI_EMBEDDINGS_API_VERSION=2025-01-01-preview
   AZURE_OPENAI_EMBEDDINGS_ENDPOINT=your_endpoint
   AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=embeddings

   AZURE_OPENAI_API_KEY=your_key
   AZURE_OPENAI_API_VERSION=2025-01-01-preview
   AZURE_OPENAI_API_BASE=your_endpoint
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1
   ```

## Running the System

### Option 1: Run Everything (Recommended)
```bash
./run.sh
```

This starts:
- FastAPI backend at http://localhost:8000
- Streamlit UI at http://localhost:8501

### Option 2: Run Components Separately

**Terminal 1 - API Server:**
```bash
source venv/bin/activate
python -m uvicorn src.api.main:app --reload --port 8000
```

**Terminal 2 - Streamlit UI:**
```bash
source venv/bin/activate
streamlit run src/ui/streamlit_app.py
```

## Using the System

### Web Interface (http://localhost:8501)

1. **Chat Tab**
   - Ask questions
   - View sources and memories used
   - Sessions are tracked by Session ID

2. **Documents Tab**
   - Upload text files
   - Or paste text directly
   - Documents are automatically indexed

3. **Memories Tab**
   - Search existing memories
   - Filter by type (episodic/semantic)
   - View memory details

4. **Add Fact Tab**
   - Manually add important facts
   - Set importance level
   - Add tags for organization

### API Endpoints (http://localhost:8000)

- **POST /query** - Query the RAG system
- **POST /documents/upload** - Upload a document
- **POST /documents/add** - Add text directly
- **POST /memory/fact** - Store a fact
- **GET /memory/search** - Search memories
- **GET /memory/stats** - Get memory statistics
- **GET /health** - Health check

See full API docs at: http://localhost:8000/docs

## Example Usage

### 1. Add a Document
```python
import requests

response = requests.post(
    "http://localhost:8000/documents/add",
    json={
        "text": "Python is a high-level programming language...",
        "doc_id": "python_intro",
        "metadata": {"topic": "programming"}
    }
)
```

### 2. Ask a Question
```python
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What is Python?",
        "session_id": "user123",
        "top_k": 5,
        "temperature": 0.7
    }
)
print(response.json()["answer"])
```

### 3. Add a Fact to Memory
```python
response = requests.post(
    "http://localhost:8000/memory/fact",
    json={
        "fact": "The user prefers Python for data science projects",
        "importance": 0.9,
        "tags": ["preference", "programming"]
    }
)
```

## Features

- **Document Retrieval**: FAISS-based vector search for relevant documents
- **LLM Generation**: Azure OpenAI GPT-4 for answer generation
- **Long-Term Memory**: Persistent episodic and semantic memories
- **Session Tracking**: Maintain context across conversations
- **Memory Consolidation**: Important memories are retained and strengthened
- **Source Citations**: Every answer includes source documents and memories used

## Troubleshooting

### API not starting
- Check if port 8000 is already in use
- Verify Azure OpenAI credentials in .env file
- Check logs for error messages

### Streamlit not connecting
- Ensure API server is running first
- Check if port 8501 is available
- Try refreshing the browser

### Memory not persisting
- Check data/memory directory exists
- Verify write permissions
- Check ltm.db file is being created

## Project Structure

```
rag_ltm/
├── src/
│   ├── retrieval/      # Vector store and retrieval
│   ├── generation/     # LLM integration
│   ├── memory/         # Long-term memory
│   ├── pipeline/       # RAG orchestration
│   ├── api/           # FastAPI backend
│   └── ui/            # Streamlit interface
├── data/
│   ├── memory/        # SQLite database
│   ├── embeddings/    # Vector indices
│   └── documents/     # Uploaded files
├── configs/           # Configuration files
└── tests/            # Test suite
```

## Next Steps

- Explore the Streamlit UI
- Add your own documents
- Test different queries
- Monitor memory statistics
- Customize prompts in src/generation/prompts.py
- Adjust configuration in configs/config.yaml
