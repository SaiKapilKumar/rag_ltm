from .rag_pipeline import RAGPipeline
from .config import PipelineConfig
from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.faiss_store import FAISSVectorStore
from src.retrieval.chroma_store import ChromaVectorStore
from src.retrieval.retriever import DocumentRetriever
from src.retrieval.chunking import TextChunker
from src.generation.llm_manager import LLMManager
from pathlib import Path

class PipelineFactory:
    """Factory for creating RAG pipelines"""
    
    @staticmethod
    def create_pipeline(config: PipelineConfig) -> RAGPipeline:
        """Create a RAG pipeline from configuration"""
        
        # Create embedding manager
        embedder = EmbeddingManager(
            model_name=config.embedding_model,
            provider=config.embedding_provider
        )
        
        # Create vector store
        if config.vector_db_type == "faiss":
            persist_dir = str(Path(__file__).parent.parent.parent / "data" / "embeddings" / "faiss")
            vector_store = FAISSVectorStore(
                dimension=embedder.get_dimension(),
                persist_directory=persist_dir
            )
            # Try to load existing index
            persist_path = str(Path(persist_dir) / "faiss_index")
            loaded = vector_store.load(persist_path)
            if loaded:
                print(f"✅ Loaded existing FAISS index with {len(vector_store.documents)} documents")
            else:
                print("📝 Starting with empty FAISS index")
        elif config.vector_db_type == "chroma":
            persist_dir = str(Path(__file__).parent.parent.parent / "data" / "embeddings" / "chroma")
            vector_store = ChromaVectorStore(persist_directory=persist_dir)
            stats = vector_store.get_stats()
            print(f"✅ Loaded Chroma collection with {stats['total_documents']} documents")
        else:
            raise ValueError(f"Unsupported vector_db_type: {config.vector_db_type}")
        
        # Create text chunker
        chunker = TextChunker(
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap
        )
        
        # Create retriever
        retriever = DocumentRetriever(
            vector_store=vector_store,
            embedding_manager=embedder,
            chunker=chunker
        )
        
        # Create LLM manager
        llm_manager = LLMManager(
            provider=config.llm_provider,
            model=config.llm_model
        )
        
        # Create pipeline
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_manager=llm_manager,
            top_k=config.document_retrieval_k,
            max_context_length=config.max_context_length
        )
        
        return pipeline
