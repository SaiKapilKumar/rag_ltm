import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Set
from pathlib import Path
from .vector_store import VectorStoreBase, Document, SearchResult

class FAISSVectorStore(VectorStoreBase):
    """FAISS-based vector store"""
    
    def __init__(self, dimension: int, persist_directory: str = "data/embeddings/faiss"):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.id_to_index = {}
        self.persist_directory = persist_directory
        self.persist_path = os.path.join(persist_directory, "faiss_index")
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
    
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
        
        # Auto-save after adding documents
        self.save(self.persist_path)
        print(f"    💾 FAISS: Index saved to {self.persist_path}")
    
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
            # Rebuild without auto-save to avoid double save
            embeddings = np.array([doc.embedding for doc in self.documents], dtype=np.float32)
            self.index.add(embeddings)
            for i, doc in enumerate(self.documents):
                self.id_to_index[doc.doc_id] = i
        
        # Save after deletion
        self.save(self.persist_path)
    
    def save(self, path: str):
        """Save index and documents"""
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.docs", "wb") as f:
            pickle.dump((self.documents, self.id_to_index), f)
    
    def load(self, path: str):
        """Load index and documents"""
        try:
            if os.path.exists(f"{path}.index") and os.path.exists(f"{path}.docs"):
                self.index = faiss.read_index(f"{path}.index")
                with open(f"{path}.docs", "rb") as f:
                    self.documents, self.id_to_index = pickle.load(f)
                print(f"    📂 FAISS: Loaded {len(self.documents)} documents from {path}")
                return True
        except Exception as e:
            print(f"    ⚠️ FAISS: Failed to load index from {path}: {str(e)}")
        return False
    
    def list_documents(self) -> List[Dict]:
        """List all documents in the store"""
        unique_docs = {}
        for doc in self.documents:
            original_id = doc.metadata.get('original_doc_id', doc.doc_id)
            if original_id not in unique_docs:
                unique_docs[original_id] = {
                    'doc_id': original_id,
                    'filename': doc.metadata.get('filename', original_id),
                    'source': doc.metadata.get('source', 'unknown'),
                    'file_type': doc.metadata.get('file_type', ''),
                    'chunk_count': 0
                }
            unique_docs[original_id]['chunk_count'] += 1
        return list(unique_docs.values())
    
    def clear(self):
        """Clear all documents from the store"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.id_to_index = {}
        # Save empty state
        self.save(self.persist_path)
        print("    🗑️ FAISS: Vector store cleared")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        unique_docs = set()
        for doc in self.documents:
            original_id = doc.metadata.get('original_doc_id', doc.doc_id)
            unique_docs.add(original_id)
        
        return {
            'total_chunks': len(self.documents),
            'total_documents': len(unique_docs),
            'index_size': self.index.ntotal,
            'dimension': self.dimension
        }
