import chromadb
from typing import List
from .vector_store import VectorStoreBase, Document, SearchResult

class ChromaVectorStore(VectorStoreBase):
    """ChromaDB-based vector store"""
    
    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = "../data/embeddings/chroma"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def add_documents(self, documents: List[Document]):
        """Add documents to Chroma"""
        if not documents:
            return
        
        self.collection.add(
            ids=[doc.doc_id for doc in documents],
            embeddings=[doc.embedding for doc in documents],
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata for doc in documents]
        )
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[SearchResult]:
        """Search for similar documents"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        search_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                doc = Document(
                    doc_id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    embedding=results['embeddings'][0][i] if results['embeddings'] else [],
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                )
                score = 1.0 - results['distances'][0][i] if results['distances'] else 0.5
                search_results.append(SearchResult(document=doc, score=float(score)))
        
        return search_results
    
    def delete_document(self, doc_id: str):
        """Delete a document"""
        self.collection.delete(ids=[doc_id])
    
    def save(self, path: str):
        """Save (Chroma auto-persists)"""
        pass
    
    def load(self, path: str):
        """Load (Chroma auto-loads)"""
        pass
