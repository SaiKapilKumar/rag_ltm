import uuid
from datetime import datetime
from typing import List, Optional, Dict
from .memory_types import Memory, MemoryType
from .memory_store import MemoryStore
from .importance_calculator import ImportanceCalculator
from src.retrieval.embeddings import EmbeddingManager

class LongTermMemory:
    """Long-term memory system with vector search"""
    
    def __init__(
        self,
        memory_store: MemoryStore,
        embedding_manager: EmbeddingManager,
        importance_threshold: float = 0.3
    ):
        self.memory_store = memory_store
        self.embedding_manager = embedding_manager
        self.importance_threshold = importance_threshold
        self.importance_calc = ImportanceCalculator()
    
    def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        source: str = "system",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        importance: Optional[float] = None
    ) -> Memory:
        """Store a new memory"""
        if tags is None:
            tags = []
        if metadata is None:
            metadata = {}
        
        # Calculate importance if not provided
        if importance is None:
            importance = self.importance_calc.calculate_importance(content, metadata)
        
        # Generate embedding
        embedding = self.embedding_manager.embed_text(content)
        
        # Create memory
        memory = Memory(
            memory_id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            embedding=embedding,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            importance=importance,
            strength=1.0,
            source=source,
            tags=tags,
            metadata=metadata,
            related_memory_ids=[]
        )
        
        # Store
        self.memory_store.store_memory(memory)
        return memory
    
    def retrieve_memories(
        self,
        query: str,
        k: int = 5,
        memory_type: Optional[MemoryType] = None,
        min_strength: float = 0.1
    ) -> List[Memory]:
        """Retrieve relevant memories for a query"""
        # Get all memories matching filters
        all_memories = self.memory_store.get_all_memories(
            memory_type=memory_type,
            min_strength=min_strength
        )
        
        if not all_memories:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_text(query)
        
        # Calculate similarities
        similarities = []
        for memory in all_memories:
            similarity = self._cosine_similarity(query_embedding, memory.embedding)
            similarities.append((memory, similarity))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_memories = [mem for mem, _ in similarities[:k]]
        
        # Update access for retrieved memories
        for memory in top_memories:
            self.memory_store.update_memory_access(memory.memory_id)
        
        return top_memories
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about stored memories"""
        all_memories = self.memory_store.get_all_memories()
        
        episodic = sum(1 for m in all_memories if m.memory_type == MemoryType.EPISODIC)
        semantic = sum(1 for m in all_memories if m.memory_type == MemoryType.SEMANTIC)
        
        avg_strength = sum(m.strength for m in all_memories) / len(all_memories) if all_memories else 0
        avg_importance = sum(m.importance for m in all_memories) / len(all_memories) if all_memories else 0
        
        return {
            "total_memories": len(all_memories),
            "episodic_memories": episodic,
            "semantic_memories": semantic,
            "average_strength": avg_strength,
            "average_importance": avg_importance
        }
