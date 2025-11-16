# Long-Term Memory in RAG Systems: Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [What is Long-Term Memory?](#what-is-long-term-memory)
3. [Memory Types](#memory-types)
4. [Architecture Overview](#architecture-overview)
5. [Core Components](#core-components)
6. [How to Build a Long-Term Memory System](#how-to-build-a-long-term-memory-system)
7. [Implementation Details](#implementation-details)
8. [Memory Retrieval Process](#memory-retrieval-process)
9. [Integration with RAG Pipeline](#integration-with-rag-pipeline)
10. [Best Practices](#best-practices)
11. [Advanced Concepts](#advanced-concepts)
12. [Performance Considerations](#performance-considerations)

---

## Introduction

Long-term memory (LTM) in AI systems enables models to remember and utilize information from past interactions, creating a more contextual and personalized experience. Unlike short-term context windows, LTM persists across sessions and can recall relevant information from historical interactions.

### Why Long-Term Memory Matters

- **Contextual Continuity**: Remember user preferences, past conversations, and established facts
- **Knowledge Accumulation**: Build up a knowledge base over time from interactions
- **Personalization**: Adapt responses based on user history
- **Efficiency**: Avoid re-asking for information already provided
- **Enhanced RAG**: Combine document retrieval with experiential knowledge

---

## What is Long-Term Memory?

Long-term memory in AI systems mimics human memory by storing information persistently and retrieving it when relevant. It consists of:

### Key Characteristics

1. **Persistence**: Survives beyond individual sessions
2. **Selectivity**: Not all information is stored; importance determines retention
3. **Decay**: Memory strength decreases over time without reinforcement
4. **Consolidation**: Important memories are strengthened and preserved
5. **Association**: Memories link to related concepts for better retrieval

### Differences from Context Windows

| Aspect | Context Window | Long-Term Memory |
|--------|---------------|------------------|
| Duration | Single conversation | Persistent across sessions |
| Size | Limited (tokens) | Virtually unlimited |
| Storage | In-memory | Database-backed |
| Retrieval | Sequential | Semantic search |
| Selection | Recent messages | Relevance-based |

---

## Memory Types

Based on cognitive science, we implement three types of memory:

### 1. Episodic Memory

**Definition**: Memories of specific events or interactions

**Characteristics**:
- Time-stamped experiences
- Conversational history
- Context-rich interactions

**Example Use Cases**:
```
"Remember when the user asked about Python decorators yesterday?"
"In our last conversation, they mentioned preferring functional programming"
```

**Storage Format**:
```python
{
    "content": "Q: How do I implement caching?\nA: You can use Python's @lru_cache decorator...",
    "type": "episodic",
    "created_at": "2025-11-03T10:30:00",
    "metadata": {
        "session_id": "user_123",
        "query": "How do I implement caching?"
    }
}
```

### 2. Semantic Memory

**Definition**: Facts, concepts, and general knowledge

**Characteristics**:
- Context-independent truths
- Domain-specific knowledge
- User preferences and rules

**Example Use Cases**:
```
"The user prefers tabs over spaces"
"The project uses TypeScript with strict mode enabled"
"API endpoint: https://api.example.com/v1"
```

**Storage Format**:
```python
{
    "content": "User prefers async/await over Promise.then() chains",
    "type": "semantic",
    "importance": 0.85,
    "tags": ["coding_style", "javascript", "preferences"]
}
```

### 3. Procedural Memory

**Definition**: How-to knowledge and procedures (planned for future implementation)

**Characteristics**:
- Step-by-step processes
- Workflows and procedures
- Learned skills

**Example Use Cases**:
```
"User's typical deployment workflow"
"Steps for debugging production issues"
"Code review checklist"
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                RAG Pipeline with Memory                      │
├─────────────────────┬───────────────────┬───────────────────┤
│                     │                   │                   │
│   Document         │    Memory         │    Context        │
│   Retrieval        │    Retrieval      │    Assembly       │
│                    │                   │                   │
│  ┌──────────┐     │  ┌─────────┐     │  ┌──────────┐    │
│  │  Vector  │     │  │ Memory  │     │  │ Combine  │    │
│  │  Store   │     │  │  Store  │     │  │ Sources  │    │
│  │ (FAISS)  │     │  │(SQLite) │     │  │          │    │
│  └──────────┘     │  └─────────┘     │  └──────────┘    │
│       │           │       │          │       │           │
│       ▼           │       ▼          │       ▼           │
│  Top-K Docs       │  Top-K Memories  │  Context String   │
└───────────────────┴──────────────────┴───────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   LLM Generation                             │
│            (Azure OpenAI GPT-4)                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Response                                  │
│           + Memory Storage (Episodic)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Memory Data Model

**Location**: `src/memory/memory_types.py`

The `Memory` class represents a single memory unit:

```python
@dataclass
class Memory:
    # Identification
    memory_id: str                    # Unique identifier
    content: str                      # Memory content/text
    memory_type: MemoryType           # episodic/semantic/procedural
    
    # Vector representation
    embedding: List[float]            # For semantic search
    
    # Temporal metadata
    created_at: datetime              # When memory was created
    last_accessed: datetime           # Last retrieval time
    access_count: int                 # Number of retrievals
    
    # Memory strength metrics
    importance: float                 # 0-1: How important is this?
    strength: float                   # 0-1: Current retention strength
    
    # Contextual information
    source: str                       # Origin (user, system, document)
    tags: List[str]                   # Categorization tags
    metadata: Dict                    # Additional context
    
    # Relationships
    related_memory_ids: List[str]     # Links to related memories
```

**Key Design Decisions**:
- **Embeddings**: Enable semantic similarity search
- **Temporal tracking**: Support decay and consolidation
- **Importance scoring**: Determines retention priority
- **Metadata flexibility**: Extensible for future features

### 2. Memory Storage

**Location**: `src/memory/memory_store.py`

SQLite-based persistent storage with CRUD operations:

```python
class MemoryStore:
    def store_memory(self, memory: Memory)
    def get_memory(self, memory_id: str) -> Optional[Memory]
    def update_memory_access(self, memory_id: str)
    def update_memory_strength(self, memory_id: str, new_strength: float)
    def get_all_memories(
        memory_type: Optional[MemoryType] = None,
        min_strength: float = 0.0
    ) -> List[Memory]
    def delete_memory(self, memory_id: str)
```

**Database Schema**:
```sql
CREATE TABLE memories (
    memory_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    embedding TEXT NOT NULL,           -- JSON serialized vector
    created_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER NOT NULL,
    importance REAL NOT NULL,
    strength REAL NOT NULL,
    source TEXT NOT NULL,
    tags TEXT,                         -- JSON array
    metadata TEXT,                     -- JSON object
    related_memory_ids TEXT            -- JSON array
)
```

**Why SQLite?**
- Serverless: No separate database process
- File-based: Easy backup and portability
- Fast: Suitable for millions of records
- SQL queries: Powerful filtering and sorting
- Zero configuration: Works out of the box

### 3. Importance Calculator

**Location**: `src/memory/importance_calculator.py`

Determines how important a memory is for retention:

```python
class ImportanceCalculator:
    @staticmethod
    def calculate_importance(
        content: str,
        metadata: dict = None,
        context: str = ""
    ) -> float:
        score = 0.5  # Base score
        
        # Length-based adjustments
        if len(content) > 200:
            score += 0.1  # Detailed content
        elif len(content) < 50:
            score -= 0.1  # Too brief
        
        # Keyword detection
        important_keywords = [
            'important', 'critical', 'key', 'essential', 
            'remember', 'prefer', 'always', 'never', 
            'must', 'should', 'fact'
        ]
        keyword_count = sum(1 for kw in important_keywords 
                           if kw in content.lower())
        score += min(keyword_count * 0.05, 0.2)
        
        # User-marked importance
        if metadata and metadata.get('user_marked_important'):
            score += 0.3
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
```

**Scoring Factors**:
1. **Content length**: More detailed = potentially more important
2. **Keywords**: Explicit importance markers
3. **User signals**: Explicit importance marking
4. **Source type**: User facts vs. casual chat
5. **Punctuation**: Emphasis indicators

### 4. Long-Term Memory Manager

**Location**: `src/memory/long_term_memory.py`

The main interface for memory operations:

```python
class LongTermMemory:
    def __init__(
        self,
        memory_store: MemoryStore,
        embedding_manager: EmbeddingManager,
        importance_threshold: float = 0.3
    ):
        self.memory_store = memory_store
        self.embedding_manager = embedding_manager
        self.importance_calc = ImportanceCalculator()
    
    def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        source: str = "system",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        importance: Optional[float] = None
    ) -> Memory
    
    def retrieve_memories(
        self,
        query: str,
        k: int = 5,
        memory_type: Optional[MemoryType] = None,
        min_strength: float = 0.1
    ) -> List[Memory]
    
    def get_memory_stats(self) -> Dict
```

**Key Responsibilities**:
- Generate embeddings for new memories
- Calculate importance scores
- Perform semantic similarity search
- Track memory access patterns
- Provide memory statistics

---

## How to Build a Long-Term Memory System

### Step 1: Define Memory Structure

Create data models for your memories:

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class MemoryType(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

@dataclass
class Memory:
    memory_id: str
    content: str
    memory_type: MemoryType
    embedding: List[float]
    created_at: datetime
    importance: float
    strength: float
    # ... additional fields
```

### Step 2: Set Up Persistent Storage

Choose a storage backend (SQLite, PostgreSQL, MongoDB):

```python
import sqlite3

class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                content TEXT,
                memory_type TEXT,
                embedding TEXT,
                created_at TEXT,
                importance REAL,
                strength REAL
            )
        ''')
        conn.commit()
        conn.close()
```

### Step 3: Implement Embedding Generation

Use a model to generate vector embeddings:

```python
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
```

### Step 4: Implement Importance Scoring

Create a system to evaluate memory importance:

```python
class ImportanceCalculator:
    @staticmethod
    def calculate_importance(content: str, metadata: dict) -> float:
        score = 0.5
        
        # Add your scoring logic
        if 'important' in content.lower():
            score += 0.2
        
        if metadata.get('user_marked_important'):
            score += 0.3
        
        return min(1.0, score)
```

### Step 5: Implement Semantic Search

Retrieve relevant memories using similarity:

```python
import numpy as np

class LongTermMemory:
    def retrieve_memories(
        self, 
        query: str, 
        k: int = 5
    ) -> List[Memory]:
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_text(query)
        
        # Get all memories
        all_memories = self.memory_store.get_all_memories()
        
        # Calculate similarities
        similarities = []
        for memory in all_memories:
            sim = self._cosine_similarity(
                query_embedding, 
                memory.embedding
            )
            similarities.append((memory, sim))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in similarities[:k]]
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / 
                    (np.linalg.norm(v1) * np.linalg.norm(v2)))
```

### Step 6: Integrate with RAG Pipeline

Combine document retrieval with memory retrieval:

```python
class RAGWithMemory:
    def __init__(self, retriever, llm, memory_system):
        self.retriever = retriever
        self.llm = llm
        self.memory = memory_system
    
    def query(self, question: str, session_id: str):
        # Retrieve documents
        documents = self.retriever.retrieve(question, k=5)
        
        # Retrieve memories
        memories = self.memory.retrieve_memories(question, k=3)
        
        # Assemble context
        context = self._build_context(documents, memories)
        
        # Generate response
        response = self.llm.generate(
            prompt=self._format_prompt(question, context)
        )
        
        # Store interaction as memory
        self.memory.store_memory(
            content=f"Q: {question}\nA: {response}",
            memory_type=MemoryType.EPISODIC,
            source=f"session_{session_id}"
        )
        
        return response
```

### Step 7: Implement Memory Consolidation

Strengthen important memories and decay unused ones:

```python
class MemoryConsolidation:
    def consolidate(self, decay_rate: float = 0.01):
        memories = self.memory_store.get_all_memories()
        
        for memory in memories:
            # Calculate time since last access
            days_since_access = (
                datetime.now() - memory.last_accessed
            ).days
            
            # Apply decay
            decay = decay_rate * days_since_access
            new_strength = max(0, memory.strength - decay)
            
            # Boost important memories
            if memory.importance > 0.7:
                new_strength = min(1.0, new_strength + 0.1)
            
            # Update or delete
            if new_strength < 0.1:
                self.memory_store.delete_memory(memory.memory_id)
            else:
                self.memory_store.update_memory_strength(
                    memory.memory_id, 
                    new_strength
                )
```

### Step 8: Add Access Tracking

Track when memories are used:

```python
def retrieve_memories(self, query: str, k: int = 5):
    # ... retrieval logic ...
    
    # Update access tracking for retrieved memories
    for memory in top_memories:
        self.memory_store.update_memory_access(memory.memory_id)
    
    return top_memories
```

---

## Implementation Details

### Configuration

**File**: `configs/config.yaml`

```yaml
memory:
  storage_backend: sqlite
  db_path: ../data/memory/ltm.db
  decay_enabled: true
  decay_rate: 0.01                 # 1% per day
  consolidation_interval: 86400     # 24 hours
  importance_threshold: 0.3         # Minimum to store
  max_memories: 10000               # Storage limit
```

### Memory Storage Flow

```
User Input
    │
    ▼
┌─────────────────────┐
│ Calculate           │
│ Importance Score    │──→ < threshold? → Discard
└─────────┬───────────┘
          │ ≥ threshold
          ▼
┌─────────────────────┐
│ Generate            │
│ Embedding           │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Create Memory       │
│ Object              │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Store in Database   │
└─────────────────────┘
```

### Memory Retrieval Flow

```
Query
  │
  ▼
┌─────────────────────┐
│ Generate Query      │
│ Embedding           │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Load All Memories   │
│ (filtered by type,  │
│  min_strength)      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Calculate           │
│ Cosine Similarities │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Sort by Similarity  │
│ Return Top-K        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Update Access       │
│ Tracking            │
└─────────────────────┘
```

### Prompt Engineering with Memory

**System Prompt**:
```
You are a helpful assistant with access to:
1. Relevant documents from the knowledge base
2. Memories from past conversations
3. Important facts the user has shared

Use all three sources to provide accurate, contextual responses.
When citing information, indicate whether it comes from 
documents or memories.
```

**User Prompt Template**:
```
Question: {query}

Relevant Documents:
{document_contexts}

Relevant Memories:
{memory_contexts}

Please answer the question using the provided information.
```

---

## Memory Retrieval Process

### Detailed Algorithm

```python
def retrieve_memories(
    self,
    query: str,
    k: int = 5,
    memory_type: Optional[MemoryType] = None,
    min_strength: float = 0.1
) -> List[Memory]:
    """
    Retrieve the most relevant memories for a query
    
    Steps:
    1. Filter memories by type and strength
    2. Generate query embedding
    3. Calculate similarity scores
    4. Sort and select top-k
    5. Update access tracking
    """
    
    # Step 1: Pre-filter memories
    candidate_memories = self.memory_store.get_all_memories(
        memory_type=memory_type,
        min_strength=min_strength
    )
    
    if not candidate_memories:
        return []
    
    # Step 2: Embed query
    query_embedding = self.embedding_manager.embed_text(query)
    
    # Step 3: Score all candidates
    scored_memories = []
    for memory in candidate_memories:
        # Cosine similarity
        similarity = self._cosine_similarity(
            query_embedding, 
            memory.embedding
        )
        
        # Boost by importance and recency
        boost = (
            memory.importance * 0.3 + 
            memory.strength * 0.2
        )
        
        final_score = similarity + boost
        scored_memories.append((memory, final_score))
    
    # Step 4: Sort and select
    scored_memories.sort(key=lambda x: x[1], reverse=True)
    top_memories = [mem for mem, _ in scored_memories[:k]]
    
    # Step 5: Track access
    for memory in top_memories:
        self.memory_store.update_memory_access(memory.memory_id)
    
    return top_memories
```

### Similarity Metrics

**Cosine Similarity**:
```python
def _cosine_similarity(self, vec1, vec2) -> float:
    """
    Measure angle between two vectors
    Range: [-1, 1], typically [0, 1] for embeddings
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    return float(dot_product / norm_product)
```

**Euclidean Distance** (alternative):
```python
def _euclidean_distance(self, vec1, vec2) -> float:
    """
    Direct distance in embedding space
    Lower = more similar
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.linalg.norm(v1 - v2))
```

### Ranking Strategy

Combine multiple signals for better ranking:

```python
def _calculate_relevance_score(
    self, 
    memory: Memory,
    similarity: float
) -> float:
    """
    Combined relevance score
    """
    # Base similarity (0-1)
    score = similarity * 0.5
    
    # Importance boost (0-0.3)
    score += memory.importance * 0.3
    
    # Strength/recency (0-0.2)
    score += memory.strength * 0.2
    
    # Access frequency bonus
    if memory.access_count > 5:
        score += 0.1
    
    return score
```

---

## Integration with RAG Pipeline

### Complete Pipeline Flow

```python
class RAGWithMemory(RAGPipeline):
    """
    Enhanced RAG with long-term memory
    """
    
    def query(
        self,
        query_text: str,
        session_id: Optional[str] = None,
        store_interaction: bool = True
    ) -> MemoryRAGResponse:
        
        # 1. Document Retrieval
        documents = self.retriever.retrieve(query_text, k=5)
        
        # 2. Memory Retrieval
        memories = self.ltm.retrieve_memories(query_text, k=3)
        
        # 3. Context Assembly
        doc_context = self._format_documents(documents)
        mem_context = self._format_memories(memories)
        
        # 4. Prompt Construction
        prompt = f"""
Question: {query_text}

Relevant Documents:
{doc_context}

Relevant Past Interactions:
{mem_context}

Please provide an answer using the above information.
"""
        
        # 5. LLM Generation
        response = self.llm_manager.generate(
            prompt=prompt,
            temperature=0.7
        )
        
        # 6. Store Interaction
        if store_interaction:
            interaction = f"Q: {query_text}\nA: {response.content}"
            self.ltm.store_memory(
                content=interaction,
                memory_type=MemoryType.EPISODIC,
                source=f"conversation_{session_id}",
                metadata={
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        return MemoryRAGResponse(
            answer=response.content,
            documents=documents,
            memories=memories,
            tokens_used=response.tokens_used
        )
```

### Context Assembly Strategies

**Strategy 1: Interleaved Context**
```python
def _assemble_interleaved_context(
    self, 
    documents: List[str],
    memories: List[Memory]
) -> str:
    """
    Alternate between documents and memories
    """
    context = "## Information Sources\n\n"
    
    for i, (doc, mem) in enumerate(zip(documents, memories)):
        context += f"Document {i+1}: {doc}\n\n"
        context += f"Related Memory: {mem.content}\n\n"
    
    return context
```

**Strategy 2: Separated Context**
```python
def _assemble_separated_context(
    self,
    documents: List[str],
    memories: List[Memory]
) -> str:
    """
    Group documents and memories separately
    """
    context = "## Documents\n"
    for i, doc in enumerate(documents):
        context += f"{i+1}. {doc}\n\n"
    
    context += "\n## Past Interactions\n"
    for i, mem in enumerate(memories):
        context += f"{i+1}. {mem.content}\n"
        context += f"   (From: {mem.created_at.strftime('%Y-%m-%d')})\n\n"
    
    return context
```

**Strategy 3: Prioritized Context** (Recommended)
```python
def _assemble_prioritized_context(
    self,
    documents: List[str],
    memories: List[Memory],
    max_tokens: int = 3000
) -> str:
    """
    Prioritize by relevance, respect token limit
    """
    items = []
    
    # Add documents with source marker
    for doc in documents:
        items.append(("document", doc, len(doc.split())))
    
    # Add memories with metadata
    for mem in memories:
        items.append(("memory", mem.content, len(mem.content.split())))
    
    # Build context respecting token limit
    context = ""
    token_count = 0
    
    for source_type, content, tokens in items:
        if token_count + tokens > max_tokens:
            break
        
        if source_type == "document":
            context += f"[DOC] {content}\n\n"
        else:
            context += f"[MEMORY] {content}\n\n"
        
        token_count += tokens
    
    return context
```

---

## Best Practices

### 1. Storage Management

**Set Importance Thresholds**:
```python
# Don't store everything
if importance < 0.3:
    return  # Skip storage

# Prioritize user-marked facts
if metadata.get('user_marked_important'):
    importance = max(importance, 0.8)
```

**Implement Storage Limits**:
```python
def store_memory(self, memory: Memory):
    # Check storage limit
    if self.get_memory_count() >= self.max_memories:
        # Remove weakest memories
        self._cleanup_weak_memories()
    
    self.memory_store.store_memory(memory)
```

### 2. Retrieval Optimization

**Use Appropriate K Values**:
```python
# Episodic memories: recent interactions
episodic_memories = ltm.retrieve_memories(
    query, 
    k=3,  # Fewer, more recent
    memory_type=MemoryType.EPISODIC
)

# Semantic memories: facts and knowledge
semantic_memories = ltm.retrieve_memories(
    query,
    k=5,  # More comprehensive
    memory_type=MemoryType.SEMANTIC
)
```

**Filter by Strength**:
```python
# Only retrieve strong memories
memories = ltm.retrieve_memories(
    query,
    min_strength=0.3  # Ignore weak/decayed memories
)
```

### 3. Memory Consolidation

**Run Periodic Consolidation**:
```python
import schedule

def consolidate_memories():
    """Run daily memory consolidation"""
    consolidator = MemoryConsolidation(memory_store)
    consolidator.consolidate(decay_rate=0.01)

# Schedule daily at 2 AM
schedule.every().day.at("02:00").do(consolidate_memories)
```

**Decay Strategy**:
```python
def apply_decay(memory: Memory, days_elapsed: int) -> float:
    """
    Exponential decay with importance protection
    """
    base_decay = 0.01 * days_elapsed
    
    # Protect important memories
    protection = memory.importance * 0.5
    actual_decay = base_decay * (1 - protection)
    
    new_strength = memory.strength - actual_decay
    return max(0.0, new_strength)
```

### 4. Session Management

**Track Conversations**:
```python
def query_with_session(
    self,
    query: str,
    session_id: str
):
    # Retrieve session-specific memories
    session_memories = self.ltm.retrieve_memories(
        query,
        metadata_filter={"session_id": session_id}
    )
    
    # Also get general memories
    general_memories = self.ltm.retrieve_memories(query, k=2)
    
    # Combine both
    all_memories = session_memories + general_memories
    
    # Generate response
    response = self._generate_with_memories(query, all_memories)
    
    return response
```

### 5. Error Handling

```python
def retrieve_memories(self, query: str) -> List[Memory]:
    try:
        # Attempt retrieval
        memories = self._retrieve_from_db(query)
        return memories
    
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        return []  # Graceful degradation
    
    except EmbeddingError as e:
        logger.error(f"Embedding error: {e}")
        # Try without semantic search
        return self._retrieve_recent_memories(k=5)
```

### 6. Privacy Considerations

**Implement Data Retention Policies**:
```python
def cleanup_old_memories(self, days: int = 90):
    """
    Remove memories older than specified days
    (except explicitly marked important)
    """
    cutoff_date = datetime.now() - timedelta(days=days)
    
    memories = self.memory_store.get_all_memories()
    for memory in memories:
        if (memory.created_at < cutoff_date and 
            memory.importance < 0.7):
            self.memory_store.delete_memory(memory.memory_id)
```

**User Control**:
```python
def forget_session(self, session_id: str):
    """Allow users to delete conversation history"""
    memories = self.memory_store.get_all_memories()
    for memory in memories:
        if memory.metadata.get('session_id') == session_id:
            self.memory_store.delete_memory(memory.memory_id)
```

---

## Advanced Concepts

### 1. Memory Consolidation Algorithm

Inspired by human memory consolidation during sleep:

```python
class MemoryConsolidation:
    """
    Consolidate memories: strengthen important ones,
    decay unused ones
    """
    
    def consolidate(
        self,
        decay_rate: float = 0.01,
        boost_threshold: float = 0.7
    ):
        memories = self.memory_store.get_all_memories()
        
        for memory in memories:
            # Calculate time-based decay
            days_since_access = (
                datetime.now() - memory.last_accessed
            ).days
            
            decay = decay_rate * days_since_access
            
            # Apply decay
            new_strength = memory.strength - decay
            
            # Boost important, frequently accessed memories
            if (memory.importance > boost_threshold and 
                memory.access_count > 5):
                new_strength += 0.1  # Consolidation boost
            
            # Prune weak memories
            if new_strength < 0.1:
                self.memory_store.delete_memory(memory.memory_id)
            else:
                new_strength = min(1.0, new_strength)
                self.memory_store.update_memory_strength(
                    memory.memory_id,
                    new_strength
                )
```

### 2. Memory Linking

Create associations between related memories:

```python
def link_related_memories(
    self,
    memory_id: str,
    similarity_threshold: float = 0.8
):
    """
    Find and link semantically similar memories
    """
    memory = self.memory_store.get_memory(memory_id)
    if not memory:
        return
    
    # Find similar memories
    all_memories = self.memory_store.get_all_memories()
    related = []
    
    for other in all_memories:
        if other.memory_id == memory_id:
            continue
        
        similarity = self._cosine_similarity(
            memory.embedding,
            other.embedding
        )
        
        if similarity >= similarity_threshold:
            related.append(other.memory_id)
    
    # Update relationships
    memory.related_memory_ids = related
    self.memory_store.store_memory(memory)
```

### 3. Hierarchical Memory

Organize memories in a hierarchy:

```python
class HierarchicalMemory:
    """
    Organize memories at different abstraction levels
    """
    
    def create_summary_memory(
        self,
        memories: List[Memory],
        level: str = "daily"
    ) -> Memory:
        """
        Create a summary memory from multiple memories
        """
        # Concatenate content
        combined_content = "\n".join([m.content for m in memories])
        
        # Generate summary using LLM
        summary = self.llm.summarize(combined_content)
        
        # Calculate average importance
        avg_importance = sum(m.importance for m in memories) / len(memories)
        
        # Create summary memory
        summary_memory = self.store_memory(
            content=summary,
            memory_type=MemoryType.SEMANTIC,
            source=f"summary_{level}",
            importance=avg_importance,
            metadata={
                "summary_level": level,
                "source_memory_ids": [m.memory_id for m in memories],
                "memory_count": len(memories)
            }
        )
        
        return summary_memory
```

### 4. Adaptive Importance

Learn importance from user feedback:

```python
class AdaptiveImportance:
    """
    Adjust importance based on usage patterns
    """
    
    def adjust_importance(self, memory_id: str):
        """
        Increase importance if memory is frequently accessed
        """
        memory = self.memory_store.get_memory(memory_id)
        
        # Access-based adjustment
        if memory.access_count > 10:
            new_importance = min(
                1.0,
                memory.importance + 0.1
            )
            memory.importance = new_importance
            self.memory_store.store_memory(memory)
    
    def learn_from_feedback(
        self,
        memory_id: str,
        helpful: bool
    ):
        """
        User feedback on memory usefulness
        """
        memory = self.memory_store.get_memory(memory_id)
        
        if helpful:
            memory.importance = min(1.0, memory.importance + 0.2)
        else:
            memory.importance = max(0.0, memory.importance - 0.2)
        
        self.memory_store.store_memory(memory)
```

### 5. Multi-Modal Memories

Store different types of content:

```python
class MultiModalMemory:
    """
    Handle different content types
    """
    
    def store_code_memory(
        self,
        code: str,
        language: str,
        explanation: str
    ):
        """Store code snippets"""
        content = f"```{language}\n{code}\n```\n\n{explanation}"
        
        self.ltm.store_memory(
            content=content,
            memory_type=MemoryType.PROCEDURAL,
            tags=["code", language],
            metadata={
                "content_type": "code",
                "language": language
            }
        )
    
    def store_structured_memory(
        self,
        data: Dict,
        description: str
    ):
        """Store structured data"""
        content = f"{description}\n\nData: {json.dumps(data, indent=2)}"
        
        self.ltm.store_memory(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            tags=["structured_data"],
            metadata={
                "content_type": "structured",
                "data": data
            }
        )
```

---

## Performance Considerations

### 1. Embedding Optimization

**Batch Processing**:
```python
def store_memories_batch(self, contents: List[str]):
    """
    Store multiple memories efficiently
    """
    # Generate embeddings in batch
    embeddings = self.embedding_manager.embed_batch(contents)
    
    # Store all at once
    memories = []
    for content, embedding in zip(contents, embeddings):
        memory = Memory(
            content=content,
            embedding=embedding,
            # ... other fields
        )
        memories.append(memory)
    
    self.memory_store.store_batch(memories)
```

**Caching**:
```python
from functools import lru_cache

class CachedEmbeddingManager:
    def __init__(self):
        self.cache = {}
    
    def embed_text(self, text: str) -> List[float]:
        # Check cache
        if text in self.cache:
            return self.cache[text]
        
        # Generate and cache
        embedding = self.model.encode(text)
        self.cache[text] = embedding.tolist()
        
        return self.cache[text]
```

### 2. Database Indexing

Add indexes for faster queries:

```sql
-- Index on memory type
CREATE INDEX idx_memory_type ON memories(memory_type);

-- Index on strength for filtering
CREATE INDEX idx_strength ON memories(strength);

-- Index on created_at for temporal queries
CREATE INDEX idx_created_at ON memories(created_at);

-- Composite index for common query patterns
CREATE INDEX idx_type_strength ON memories(memory_type, strength);
```

### 3. Vector Search Optimization

Use approximate nearest neighbor search for large datasets:

```python
import faiss
import numpy as np

class FAISSMemoryIndex:
    """
    Fast approximate similarity search using FAISS
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.memory_map = {}
    
    def add_memories(self, memories: List[Memory]):
        """Add memories to FAISS index"""
        embeddings = np.array([m.embedding for m in memories]).astype('float32')
        self.index.add(embeddings)
        
        # Map index positions to memory IDs
        start_idx = len(self.memory_map)
        for i, memory in enumerate(memories):
            self.memory_map[start_idx + i] = memory.memory_id
    
    def search(self, query_embedding: List[float], k: int = 5):
        """Fast k-NN search"""
        query_vec = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vec, k)
        
        # Map back to memory IDs
        memory_ids = [self.memory_map[idx] for idx in indices[0]]
        return memory_ids, distances[0]
```

### 4. Memory Pagination

For large result sets:

```python
def get_memories_paginated(
    self,
    page: int = 1,
    page_size: int = 100,
    memory_type: Optional[MemoryType] = None
) -> Dict:
    """
    Retrieve memories with pagination
    """
    offset = (page - 1) * page_size
    
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    # Count total
    count_query = "SELECT COUNT(*) FROM memories"
    if memory_type:
        count_query += f" WHERE memory_type = '{memory_type.value}'"
    
    total = cursor.execute(count_query).fetchone()[0]
    
    # Fetch page
    query = "SELECT * FROM memories"
    if memory_type:
        query += f" WHERE memory_type = '{memory_type.value}'"
    query += f" LIMIT {page_size} OFFSET {offset}"
    
    rows = cursor.execute(query).fetchall()
    conn.close()
    
    memories = [self._row_to_memory(row) for row in rows]
    
    return {
        "memories": memories,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": (total + page_size - 1) // page_size
    }
```

### 5. Monitoring and Metrics

Track system performance:

```python
class MemoryMetrics:
    """
    Monitor memory system performance
    """
    
    def __init__(self):
        self.metrics = {
            "retrieval_times": [],
            "storage_times": [],
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def record_retrieval(self, duration: float):
        self.metrics["retrieval_times"].append(duration)
    
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        retrieval_times = self.metrics["retrieval_times"]
        
        return {
            "avg_retrieval_time": np.mean(retrieval_times),
            "p95_retrieval_time": np.percentile(retrieval_times, 95),
            "cache_hit_rate": (
                self.metrics["cache_hits"] / 
                (self.metrics["cache_hits"] + self.metrics["cache_misses"])
            )
        }
```

---

## Usage Examples

### Example 1: Basic Memory Storage and Retrieval

```python
from src.memory.long_term_memory import LongTermMemory
from src.memory.memory_store import MemoryStore
from src.memory.memory_types import MemoryType
from src.retrieval.embeddings import EmbeddingManager

# Initialize components
memory_store = MemoryStore("data/memory/ltm.db")
embedding_manager = EmbeddingManager()
ltm = LongTermMemory(memory_store, embedding_manager)

# Store a fact
ltm.store_memory(
    content="The user prefers Python over JavaScript",
    memory_type=MemoryType.SEMANTIC,
    source="user_preference",
    tags=["preference", "programming"],
    importance=0.8
)

# Retrieve relevant memories
memories = ltm.retrieve_memories(
    query="What programming language should I use?",
    k=3
)

for memory in memories:
    print(f"Memory: {memory.content}")
    print(f"Importance: {memory.importance}")
    print(f"Created: {memory.created_at}")
    print("---")
```

### Example 2: RAG with Memory

```python
from src.pipeline.rag_with_memory import RAGWithMemory

# Initialize pipeline
pipeline = RAGWithMemory(
    retriever=document_retriever,
    llm_manager=llm,
    long_term_memory=ltm,
    top_k=5,
    memory_k=3
)

# Query with memory
response = pipeline.query(
    query_text="How do I implement caching in Python?",
    session_id="user_123",
    store_interaction=True
)

print(f"Answer: {response.answer}")
print(f"\nDocuments used: {len(response.sources)}")
print(f"Memories used: {len(response.memories_used)}")

# View memories used
for mem in response.memories_used:
    print(f"- {mem['content'][:100]}...")
```

### Example 3: Session-Based Conversations

```python
# First conversation
response1 = pipeline.query(
    "I'm working on a web scraping project",
    session_id="project_session_1"
)

# Later conversation
response2 = pipeline.query(
    "What was I working on?",
    session_id="project_session_1"
)

# The system remembers from memory:
# "I'm working on a web scraping project"
```

### Example 4: Adding Important Facts

```python
# User shares important information
pipeline.add_fact(
    fact="The production API endpoint is https://api.company.com/v2",
    importance=0.9,
    tags=["api", "production", "endpoint"]
)

# Later, when asked about API
response = pipeline.query("What's the API endpoint?")
# Memory system retrieves the stored fact
```

### Example 5: Memory Statistics

```python
stats = pipeline.get_memory_stats()

print(f"Total memories: {stats['total_memories']}")
print(f"Episodic: {stats['episodic_memories']}")
print(f"Semantic: {stats['semantic_memories']}")
print(f"Average importance: {stats['average_importance']:.2f}")
print(f"Average strength: {stats['average_strength']:.2f}")
```

---

## Conclusion

Long-term memory is a powerful addition to RAG systems, enabling:

1. **Persistent Context**: Remember information across sessions
2. **Personalization**: Adapt to user preferences and history
3. **Knowledge Accumulation**: Build expertise over time
4. **Efficient Retrieval**: Access relevant past information quickly
5. **Enhanced Responses**: Combine documents with experiential knowledge

### Key Takeaways

- Use appropriate memory types (episodic vs. semantic)
- Implement importance scoring for selective storage
- Apply decay and consolidation for memory management
- Integrate seamlessly with RAG pipelines
- Monitor and optimize performance
- Respect privacy and data retention policies

### Further Reading

- Cognitive Science: Human memory systems
- Vector Databases: FAISS, Pinecone, Weaviate
- LLM Context Management: Token optimization
- RAG Advanced Patterns: Multi-hop reasoning
- Privacy Engineering: Data retention and deletion

---

## References

This implementation is based on:
- Cognitive science principles of human memory
- RAG system best practices from research papers
- Production-grade vector database architectures
- Real-world LLM application patterns

For more details, see the codebase at `/Users/saikapilkumar/Documents/PersonalProjects/context_enginnering/rag_ltm/`
