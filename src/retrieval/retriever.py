from typing import List, Optional, Dict
from .vector_store import VectorStoreBase, SearchResult, Document
from .embeddings import EmbeddingManager
from .chunking import TextChunker

class DocumentRetriever:
    """High-level retrieval interface"""
    
    def __init__(
        self,
        vector_store: VectorStoreBase,
        embedding_manager: EmbeddingManager,
        chunker: Optional[TextChunker] = None
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.chunker = chunker or TextChunker()
    
    def add_documents(
        self,
        texts: List[str],
        doc_ids: List[str],
        metadatas: Optional[List[Dict]] = None
    ):
        """Add documents to the retriever"""
        import time
        
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        all_documents = []
        total_chunks = 0
        
        # Process each document
        for text, doc_id, metadata in zip(texts, doc_ids, metadatas):
            # Chunking timing
            chunk_start = time.time()
            chunks = self.chunker.chunk_text(text, doc_id, metadata)
            chunk_time = time.time() - chunk_start
            total_chunks += len(chunks)
            
            print(f"  📄 Chunking: {len(chunks)} chunks created in {chunk_time:.3f}s")
            
            # Embedding timing - process in batches for efficiency
            embed_start = time.time()
            chunk_texts = [chunk.content for chunk in chunks]
            
            if len(chunks) > 1:
                print(f"  🧮 Creating embeddings for {len(chunks)} chunks...")
                # Batch embedding for efficiency
                embeddings = self.embedding_manager.embed_texts(chunk_texts)
            else:
                embeddings = [self.embedding_manager.embed_text(chunk_texts[0])]
            
            # Create document objects
            for chunk, embedding in zip(chunks, embeddings):
                doc = Document(
                    doc_id=chunk.chunk_id,
                    content=chunk.content,
                    embedding=embedding,
                    metadata={**metadata, "original_doc_id": doc_id}
                )
                all_documents.append(doc)
            
            embed_time = time.time() - embed_start
            print(f"  🧮 Embeddings: {len(chunks)} embeddings created in {embed_time:.3f}s")
        
        # Vector store indexing
        if all_documents:
            index_start = time.time()
            self.vector_store.add_documents(all_documents)
            index_time = time.time() - index_start
            
            print(f"  🔍 Vector indexing: {len(all_documents)} documents indexed in {index_time:.3f}s")
    
    def retrieve(self, query: str, k: int = 5) -> List[SearchResult]:
        """Retrieve relevant documents for a query"""
        query_embedding = self.embedding_manager.embed_text(query)
        return self.vector_store.search(query_embedding, k)
