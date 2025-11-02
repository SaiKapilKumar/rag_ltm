from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
from enum import Enum

class MemoryType(Enum):
    """Types of memories"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

@dataclass
class Memory:
    """Base memory structure"""
    memory_id: str
    content: str
    memory_type: MemoryType
    embedding: List[float]
    
    # Metadata
    created_at: datetime
    last_accessed: datetime
    access_count: int
    
    # Memory strength
    importance: float
    strength: float
    
    # Context
    source: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    # Relations
    related_memory_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
            "strength": self.strength,
            "source": self.source,
            "tags": self.tags,
            "metadata": self.metadata,
            "related_memory_ids": self.related_memory_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Memory":
        """Create from dictionary"""
        return cls(
            memory_id=data["memory_id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            embedding=data["embedding"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data["access_count"],
            importance=data["importance"],
            strength=data["strength"],
            source=data["source"],
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            related_memory_ids=data.get("related_memory_ids", [])
        )
