import chromadb
from typing import List, Dict
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
    
    def list_documents(self) -> List[Dict]:
        """List all documents in the store"""
        try:
            # Get all documents from collection
            results = self.collection.get()
            
            if not results['ids']:
                return []
            
            unique_docs = {}
            for i, doc_id in enumerate(results['ids']):
                metadata = results['metadatas'][i] if results['metadatas'] else {}
                original_id = metadata.get('original_doc_id', doc_id)
                
                if original_id not in unique_docs:
                    unique_docs[original_id] = {
                        'doc_id': original_id,
                        'filename': metadata.get('filename', original_id),
                        'source': metadata.get('source', 'unknown'),
                        'file_type': metadata.get('file_type', ''),
                        'chunk_count': 0
                    }
                unique_docs[original_id]['chunk_count'] += 1
            
            return list(unique_docs.values())
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []
    
    def clear(self):
        """Clear all documents from the store"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.get_or_create_collection(name=self.collection.name)
            print("    🗑️ Chroma: Vector store cleared")
        except Exception as e:
            print(f"Error clearing store: {e}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        try:
            results = self.collection.get()
            total_chunks = len(results['ids']) if results['ids'] else 0
            
            unique_docs = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    original_id = metadata.get('original_doc_id', '')
                    if original_id:
                        unique_docs.add(original_id)
            
            return {
                'total_chunks': total_chunks,
                'total_documents': len(unique_docs),
                'collection_name': self.collection.name
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {'total_chunks': 0, 'total_documents': 0, 'collection_name': self.collection.name}
