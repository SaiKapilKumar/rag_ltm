from .rag_pipeline import RAGPipeline
from .config import PipelineConfig
from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.faiss_store import FAISSVectorStore
from src.retrieval.chroma_store import ChromaVectorStore
from src.retrieval.retriever import DocumentRetriever
from src.retrieval.chunking import TextChunker
from src.generation.llm_manager import LLMManager

class PipelineFactory:
    """Factory for creating RAG pipelines"""
    
    @staticmethod
    def create_pipeline(config: PipelineConfig) -> RAGPipeline:
        """Create a RAG pipeline from configuration"""
        
        # Create embedding manager
        embedder = EmbeddingManager(
            model_name=config.embedding_model,
            provider="sentence-transformers"
        )
        
        # Create vector store
        if config.vector_db_type == "faiss":
            vector_store = FAISSVectorStore(dimension=embedder.get_dimension())
        elif config.vector_db_type == "chroma":
            vector_store = ChromaVectorStore()
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
