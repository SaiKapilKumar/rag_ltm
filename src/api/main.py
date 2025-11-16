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
from src.memory.frequency_analyzer import FrequencyAnalyzer
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
frequency_analyzer: Optional[FrequencyAnalyzer] = None

@app.on_event("startup")
async def startup():
    global pipeline, frequency_analyzer
    
    # Create base pipeline
    base_pipeline = PipelineFactory.create_pipeline(config)
    
    # Add memory
    db_path = Path(__file__).parent.parent.parent / "data" / "memory" / "ltm.db"
    memory_store = MemoryStore(str(db_path))
    embedder = EmbeddingManager(
        model_name=config.embedding_model,
        provider=config.embedding_provider
    )
    ltm = LongTermMemory(
        memory_store=memory_store,
        embedding_manager=embedder,
        importance_threshold=config.importance_threshold,
        decay_enabled=config.decay_enabled,
        decay_rate=config.decay_rate
    )
    
    # Create frequency analyzer
    frequency_analyzer = FrequencyAnalyzer(memory_store)
    
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
    user_id: Optional[str] = None
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
            user_id=request.user_id,
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
        
        # Save the file to data/documents/ folder
        documents_dir = Path(__file__).parent.parent.parent / "data" / "documents"
        documents_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = documents_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(content)
        print(f" File saved to: {file_path}")
        
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
        
        print(f" Indexing completed in {index_time:.3f}s")
        print(f" Total upload time: {total_time:.3f}s for {file.filename} ({len(text)} chars)")
        
        return {
            "status": "success",
            "filename": file.filename,
            "file_path": str(file_path),
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
    print(f" Starting text indexing: {request.doc_id} ({len(request.text)} chars)")
    
    try:
        pipeline.add_documents(
            texts=[request.text],
            doc_ids=[request.doc_id],
            metadatas=[request.metadata or {}]
        )
        
        total_time = time.time() - start_time
        print(f" Text indexing completed in {total_time:.3f}s for {request.doc_id}")
        
        return {
            "status": "success",
            "doc_id": request.doc_id,
            "size": len(request.text),
            "indexing_time": round(total_time, 3)
        }
    except Exception as e:
        print(f" Indexing failed for {request.doc_id}: {str(e)}")
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

@app.post("/memory/decay")
async def apply_decay():
    """Manually apply decay to all memories (batch consolidation)"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        result = pipeline.ltm.apply_decay_to_all_memories()
        return {
            "status": "success",
            "message": "Decay applied to all memories",
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/documents/list")
async def list_documents():
    """List all documents in the vector store"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        documents = pipeline.retriever.vector_store.list_documents()
        return {
            "count": len(documents),
            "documents": documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/stats")
async def get_document_stats():
    """Get vector store statistics"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        stats = pipeline.retriever.vector_store.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/clear")
async def clear_documents():
    """Clear all documents from the vector store"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        pipeline.retriever.vector_store.clear()
        return {
            "status": "success",
            "message": "All documents cleared from vector store"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SESSION MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/sessions")
async def list_sessions(user_id: Optional[str] = None):
    """List all sessions, optionally filtered by user"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        sessions = pipeline.ltm.memory_store.get_all_sessions(user_id)
        return {
            "count": len(sessions),
            "sessions": sessions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/memories")
async def get_session_memories(
    session_id: str,
    memory_type: Optional[str] = None,
    limit: Optional[int] = None
):
    """Get all memories for a specific session"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        mem_type = MemoryType(memory_type) if memory_type else None
        memories = pipeline.ltm.memory_store.get_memories_by_session(
            session_id,
            memory_type=mem_type,
            limit=limit
        )
        
        return {
            "session_id": session_id,
            "count": len(memories),
            "memories": [
                {
                    "memory_id": m.memory_id,
                    "content": m.content,
                    "type": m.memory_type.value,
                    "importance": m.importance,
                    "strength": m.strength,
                    "created_at": m.created_at.isoformat(),
                    "access_count": m.access_count,
                    "tags": m.tags
                }
                for m in memories
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/stats")
async def get_session_stats(session_id: str):
    """Get statistics for a specific session"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        stats = pipeline.ltm.memory_store.get_session_stats(session_id)
        return {
            "session_id": session_id,
            **stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/timeline")
async def get_session_timeline(session_id: str, limit: Optional[int] = None):
    """Get chronological conversation timeline for a session"""
    if frequency_analyzer is None:
        raise HTTPException(status_code=503, detail="Frequency analyzer not initialized")
    
    try:
        timeline = frequency_analyzer.get_conversation_timeline(session_id, limit)
        return {
            "session_id": session_id,
            "count": len(timeline),
            "timeline": timeline
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/frequent-topics")
async def get_session_frequent_topics(session_id: str, top_n: int = 10):
    """Get most frequently discussed topics in a session"""
    if frequency_analyzer is None:
        raise HTTPException(status_code=503, detail="Frequency analyzer not initialized")
    
    try:
        topics = frequency_analyzer.get_frequent_topics(session_id=session_id, top_n=top_n)
        return {
            "session_id": session_id,
            "count": len(topics),
            "topics": topics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/patterns")
async def get_session_patterns(session_id: str):
    """Get query patterns for a session"""
    if frequency_analyzer is None:
        raise HTTPException(status_code=503, detail="Frequency analyzer not initialized")
    
    try:
        patterns = frequency_analyzer.get_query_patterns(session_id=session_id)
        return {
            "session_id": session_id,
            **patterns
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}/memories")
async def clear_session_memories(session_id: str, memory_type: Optional[str] = None):
    """Clear memories for a specific session"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        mem_type = MemoryType(memory_type) if memory_type else None
        deleted_count = pipeline.ltm.memory_store.delete_session_memories(
            session_id,
            memory_type=mem_type
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "deleted_count": deleted_count,
            "memory_type": memory_type or "all"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# USER MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/users")
async def list_users():
    """Get list of unique users"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        users = pipeline.ltm.memory_store.get_unique_users()
        return {
            "count": len(users),
            "users": users
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/memories")
async def get_user_memories(
    user_id: str,
    memory_type: Optional[str] = None
):
    """Get all memories for a specific user across all sessions"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        mem_type = MemoryType(memory_type) if memory_type else None
        memories = pipeline.ltm.memory_store.get_memories_by_user(
            user_id,
            memory_type=mem_type
        )
        
        return {
            "user_id": user_id,
            "count": len(memories),
            "memories": [
                {
                    "memory_id": m.memory_id,
                    "content": m.content,
                    "type": m.memory_type.value,
                    "importance": m.importance,
                    "strength": m.strength,
                    "created_at": m.created_at.isoformat(),
                    "session_id": m.metadata.get("session_id"),
                    "access_count": m.access_count
                }
                for m in memories
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/stats")
async def get_user_stats(user_id: str):
    """Get detailed statistics for a specific user"""
    if frequency_analyzer is None:
        raise HTTPException(status_code=503, detail="Frequency analyzer not initialized")
    
    try:
        stats = frequency_analyzer.get_user_memory_usage(user_id)
        return {
            "user_id": user_id,
            **stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/frequent-topics")
async def get_user_frequent_topics(user_id: str, top_n: int = 10):
    """Get most frequently discussed topics for a user"""
    if frequency_analyzer is None:
        raise HTTPException(status_code=503, detail="Frequency analyzer not initialized")
    
    try:
        topics = frequency_analyzer.get_frequent_topics(user_id=user_id, top_n=top_n)
        return {
            "user_id": user_id,
            "count": len(topics),
            "topics": topics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/patterns")
async def get_user_patterns(user_id: str):
    """Get query patterns for a user"""
    if frequency_analyzer is None:
        raise HTTPException(status_code=503, detail="Frequency analyzer not initialized")
    
    try:
        patterns = frequency_analyzer.get_query_patterns(user_id=user_id)
        return {
            "user_id": user_id,
            **patterns
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/users/{user_id}/memories")
async def clear_user_memories(user_id: str):
    """Clear all memories for a user across all sessions"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        deleted_count = pipeline.ltm.memory_store.delete_user_memories(user_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "deleted_count": deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/migrate")
async def migrate_memory_data():
    """Migrate existing memory data to extract session_id and user_id from metadata"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        updated = pipeline.ltm.memory_store.migrate_existing_data()
        return {
            "status": "success",
            "message": "Data migration completed",
            "updated_count": updated
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/clear")
async def clear_all_memories():
    """Clear all memories from the system"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        conn = pipeline.ltm.memory_store._get_connection()
        cursor = conn.cursor()
        
        # Get count before deletion
        cursor.execute('SELECT COUNT(*) FROM memories')
        count = cursor.fetchone()[0]
        
        # Clear all memories and sessions
        cursor.execute('DELETE FROM memories')
        cursor.execute('DELETE FROM sessions')
        
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "message": "All memories cleared",
            "deleted_count": count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
