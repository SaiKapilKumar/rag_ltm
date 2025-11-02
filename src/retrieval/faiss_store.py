import faiss
import numpy as np
import pickle
from typing import List
from .vector_store import VectorStoreBase, Document, SearchResult

class FAISSVectorStore(VectorStoreBase):
    """FAISS-based vector store"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.id_to_index = {}
    
    def add_documents(self, documents: List[Document]):
        """Add documents to FAISS index"""
        if not documents:
            return
        
        print(f"    🔢 FAISS: Adding {len(documents)} document chunks to vector index...")
        
        embeddings = np.array([doc.embedding for doc in documents], dtype=np.float32)
        start_idx = len(self.documents)
        
        # Add embeddings to FAISS index
        self.index.add(embeddings)
        print(f"    📊 FAISS: Index now contains {self.index.ntotal} total vectors")
        
        # Update document storage and ID mapping
        for i, doc in enumerate(documents):
            self.documents.append(doc)
            self.id_to_index[doc.doc_id] = start_idx + i
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[SearchResult]:
        """Search for similar documents"""
        if len(self.documents) == 0:
            return []
        
        k = min(k, len(self.documents))
        query_vec = np.array([query_embedding], dtype=np.float32)
        
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                similarity = 1.0 / (1.0 + dist)
                results.append(SearchResult(
                    document=self.documents[idx],
                    score=float(similarity)
                ))
        
        return results
    
    def delete_document(self, doc_id: str):
        """Delete a document (rebuild index)"""
        if doc_id not in self.id_to_index:
            return
        
        idx_to_remove = self.id_to_index[doc_id]
        self.documents.pop(idx_to_remove)
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.id_to_index = {}
        
        if self.documents:
            self.add_documents(self.documents)
    
    def save(self, path: str):
        """Save index and documents"""
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.docs", "wb") as f:
            pickle.dump((self.documents, self.id_to_index), f)
    
    def load(self, path: str):
        """Load index and documents"""
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.docs", "rb") as f:
            self.documents, self.id_to_index = pickle.load(f)
