from abc import ABC, abstractmethod
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Document:
    """Document with embedding"""
    doc_id: str
    content: str
    embedding: List[float]
    metadata: Dict

@dataclass
class SearchResult:
    """Search result with score"""
    document: Document
    score: float

class VectorStoreBase(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]):
        """Add documents to the store"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], k: int = 5) -> List[SearchResult]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def delete_document(self, doc_id: str):
        """Delete a document"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save the index"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load the index"""
        pass
