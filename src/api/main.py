from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import after path modification
from src.pipeline.pipeline_factory import PipelineFactory
from src.pipeline.rag_with_memory import RAGWithMemory
from src.pipeline.config import PipelineConfig
from src.memory.memory_store import MemoryStore
from src.memory.long_term_memory import LongTermMemory
from src.memory.memory_types import MemoryType
from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.document_processor import DocumentProcessor

app = FastAPI(
    title="RAG with Long-Term Memory API",
    description="Retrieval-Augmented Generation with Persistent Memory",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
config = PipelineConfig.from_yaml(str(config_path))
pipeline: Optional[RAGWithMemory] = None

@app.on_event("startup")
async def startup():
    global pipeline
    
    # Create base pipeline
    base_pipeline = PipelineFactory.create_pipeline(config)
    
    # Add memory
    db_path = Path(__file__).parent.parent.parent / "data" / "memory" / "ltm.db"
    memory_store = MemoryStore(str(db_path))
    embedder = EmbeddingManager(model_name=config.embedding_model)
    ltm = LongTermMemory(memory_store, embedder)
    
    # Create memory-enhanced pipeline
    pipeline = RAGWithMemory(
        retriever=base_pipeline.retriever,
        llm_manager=base_pipeline.llm_manager,
        long_term_memory=ltm,
        top_k=config.document_retrieval_k,
        memory_k=config.memory_retrieval_k,
        max_context_length=config.max_context_length
    )
    
    print("RAG with Memory pipeline initialized successfully")
    print("="*60)
    print("📊 INDEXING PERFORMANCE MONITORING ENABLED")
    print("   - File processing time will be displayed")
    print("   - Chunking and embedding timing tracked")
    print("   - Vector indexing performance logged")
    print("="*60)

@app.get("/")
def root():
    return {
        "name": "RAG with Long-Term Memory API",
        "version": "1.0.0",
        "status": "running",
        "supported_file_types": DocumentProcessor.get_supported_extensions()
    }

@app.get("/health")
def health_check():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    stats = pipeline.get_memory_stats()
    return {
        "status": "healthy",
        "memory_count": stats["total_memories"],
        "pipeline_ready": True
    }

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: Optional[int] = None
    temperature: float = 0.7
    include_sources: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    memories_used: List[Dict]
    query: str
    tokens_used: int
    retrieval_time: float
    generation_time: float
    total_time: float
    metadata: dict

class AddDocumentRequest(BaseModel):
    text: str
    doc_id: str
    metadata: Optional[Dict] = None

class AddFactRequest(BaseModel):
    fact: str
    importance: float = 0.8
    tags: Optional[List[str]] = None

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        response = pipeline.query(
            query_text=request.query,
            session_id=request.session_id,
            top_k=request.top_k,
            temperature=request.temperature,
            include_sources=request.include_sources
        )
        
        return QueryResponse(
            answer=response.answer,
            sources=response.sources,
            memories_used=response.memories_used,
            query=response.query,
            tokens_used=response.tokens_used,
            retrieval_time=response.retrieval_time,
            generation_time=response.generation_time,
            total_time=response.total_time,
            metadata=response.metadata
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document (supports PDF, TXT, MD)"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    start_time = time.time()
    print(f"📄 Starting document upload: {file.filename}")
    
    try:
        # File processing timing
        process_start = time.time()
        content = await file.read()
        
        # Process the file based on its type
        try:
            text = DocumentProcessor.process_file(content, file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file")
        
        process_time = time.time() - process_start
        print(f"⚡ File processing completed in {process_time:.3f}s")
        
        # Indexing timing
        index_start = time.time()
        pipeline.add_documents(
            texts=[text],
            doc_ids=[file.filename],
            metadatas=[{
                "filename": file.filename,
                "source": "upload",
                "file_type": Path(file.filename).suffix.lower()
            }]
        )
        index_time = time.time() - index_start
        total_time = time.time() - start_time
        
        print(f"🚀 Indexing completed in {index_time:.3f}s")
        print(f"✅ Total upload time: {total_time:.3f}s for {file.filename} ({len(text)} chars)")
        
        return {
            "status": "success",
            "filename": file.filename,
            "size": len(text),
            "file_type": Path(file.filename).suffix.lower(),
            "processing_time": round(process_time, 3),
            "indexing_time": round(index_time, 3),
            "total_time": round(total_time, 3)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/add")
async def add_document(request: AddDocumentRequest):
    """Add document from text"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    start_time = time.time()
    print(f"📝 Starting text indexing: {request.doc_id} ({len(request.text)} chars)")
    
    try:
        pipeline.add_documents(
            texts=[request.text],
            doc_ids=[request.doc_id],
            metadatas=[request.metadata or {}]
        )
        
        total_time = time.time() - start_time
        print(f"✅ Text indexing completed in {total_time:.3f}s for {request.doc_id}")
        
        return {
            "status": "success",
            "doc_id": request.doc_id,
            "size": len(request.text),
            "indexing_time": round(total_time, 3)
        }
    except Exception as e:
        print(f"❌ Indexing failed for {request.doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/fact", response_model=dict)
async def add_fact(request: AddFactRequest):
    """Store a fact in long-term memory"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        pipeline.add_fact(
            fact=request.fact,
            importance=request.importance,
            tags=request.tags
        )
        
        return {
            "status": "success",
            "fact": request.fact,
            "importance": request.importance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/stats")
async def get_memory_stats():
    """Get memory system statistics"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return pipeline.get_memory_stats()

@app.get("/memory/search")
async def search_memories(query: str, k: int = 5, memory_type: Optional[str] = None):
    """Search memories"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        mem_type = MemoryType(memory_type) if memory_type else None
        memories = pipeline.ltm.retrieve_memories(query, k=k, memory_type=mem_type)
        
        return {
            "query": query,
            "count": len(memories),
            "memories": [
                {
                    "content": m.content,
                    "type": m.memory_type.value,
                    "importance": m.importance,
                    "strength": m.strength,
                    "created_at": m.created_at.isoformat(),
                    "access_count": m.access_count
                }
                for m in memories
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
