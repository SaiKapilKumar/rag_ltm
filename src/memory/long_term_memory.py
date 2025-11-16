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
        importance_threshold: float = 0.3,
        decay_enabled: bool = True,
        decay_rate: float = 0.01
    ):
        self.memory_store = memory_store
        self.embedding_manager = embedding_manager
        self.importance_threshold = importance_threshold
        self.importance_calc = ImportanceCalculator()
        self.decay_enabled = decay_enabled
        self.decay_rate = decay_rate
    
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
        min_strength: float = 0.1,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Memory]:
        """Retrieve relevant memories for a query
        
        Args:
            query: Query text to search for
            k: Number of memories to retrieve
            memory_type: Filter by memory type
            min_strength: Minimum strength threshold
            session_id: Filter by specific session
            user_id: Filter by specific user
        """
        # Get memories based on filtering criteria
        if session_id:
            all_memories = self.memory_store.get_memories_by_session(
                session_id,
                memory_type=memory_type,
                min_strength=min_strength
            )
        elif user_id:
            all_memories = self.memory_store.get_memories_by_user(
                user_id,
                memory_type=memory_type,
                min_strength=min_strength
            )
        else:
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
        
        # Apply decay and update access for retrieved memories (batch update)
        updates = []
        for memory in top_memories:
            new_strength = self.calculate_decay(memory)
            updates.append((memory.memory_id, new_strength))
        
        if updates:
            self.memory_store.batch_update_memory_access(updates)
        
        return top_memories
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    def calculate_decay(self, memory: Memory) -> float:
        """Calculate new strength after time-based decay
        
        Args:
            memory: Memory object to calculate decay for
            
        Returns:
            New strength value after decay (clamped to 0.1-1.0)
        """
        if not self.decay_enabled:
            return memory.strength
        
        # Calculate time elapsed since last access
        time_delta = datetime.now() - memory.last_accessed
        days_elapsed = time_delta.total_seconds() / 86400  # Convert to days
        
        if days_elapsed <= 0:
            return memory.strength
        
        # Importance acts as a protection factor (0-1)
        # High importance memories decay slower
        importance_protection = memory.importance * 0.5
        
        # Calculate decay amount
        decay_amount = self.decay_rate * days_elapsed * (1 - importance_protection)
        
        # Apply decay
        new_strength = memory.strength - decay_amount
        
        # Clamp between 0.1 (minimum threshold) and 1.0 (maximum)
        return max(0.1, min(1.0, new_strength))
    
    def apply_decay_to_all_memories(self) -> Dict:
        """Apply decay to all memories in the store
        
        Useful for batch consolidation or maintenance tasks.
        
        Returns:
            Statistics about the decay operation
        """
        all_memories = self.memory_store.get_all_memories()
        
        updated_count = 0
        weakened_count = 0
        
        for memory in all_memories:
            old_strength = memory.strength
            new_strength = self.calculate_decay(memory)
            
            if new_strength != old_strength:
                self.memory_store.update_memory_strength(memory.memory_id, new_strength)
                updated_count += 1
                
                if new_strength < 0.3:
                    weakened_count += 1
        
        return {
            "total_memories": len(all_memories),
            "updated_count": updated_count,
            "weakened_count": weakened_count,
            "decay_enabled": self.decay_enabled,
            "decay_rate": self.decay_rate
        }
    
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
