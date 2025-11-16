from typing import List, Optional, Dict
from dataclasses import dataclass
from langchain_core.prompts import PromptTemplate

from .rag_pipeline import RAGPipeline, RAGResponse
from src.memory.long_term_memory import LongTermMemory
from src.memory.memory_types import MemoryType
from src.generation.prompts import RAG_WITH_MEMORY_SYSTEM, RAG_WITH_MEMORY_PROMPT
import time

@dataclass
class MemoryRAGResponse(RAGResponse):
    """RAG response with memory information"""
    memories_used: List[Dict]
    memory_retrieval_time: float

class RAGWithMemory(RAGPipeline):
    """RAG pipeline with long-term memory integration"""
    
    def __init__(
        self,
        retriever,
        llm_manager,
        long_term_memory: LongTermMemory,
        top_k: int = 5,
        memory_k: int = 3,
        max_context_length: int = 3000
    ):
        super().__init__(retriever, llm_manager, top_k, max_context_length)
        self.ltm = long_term_memory
        self.memory_k = memory_k
    
    def query(
        self,
        query_text: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        top_k: Optional[int] = None,
        temperature: float = 0.7,
        include_sources: bool = True,
        store_interaction: bool = True
    ) -> MemoryRAGResponse:
        """Process a query through the RAG pipeline with memory"""
        start_time = time.time()
        
        # Retrieval from documents
        retrieval_start = time.time()
        k = top_k or self.top_k
        search_results = self.retriever.retrieve(query_text, k=k)
        retrieval_time = time.time() - retrieval_start
        
        # Retrieve relevant memories (filtered by session/user if provided)
        memory_start = time.time()
        relevant_memories = self.ltm.retrieve_memories(
            query_text, 
            k=self.memory_k,
            session_id=session_id,
            user_id=user_id
        )
        memory_retrieval_time = time.time() - memory_start
        
        # Assemble context
        context_texts = self._assemble_context(search_results)
        memory_texts = [m.content for m in relevant_memories]
        
        # Generate response with memory context
        generation_start = time.time()
        # Format memories and context with numbering
        formatted_memories = "\n".join([f"[M{i}] {mem}" for i, mem in enumerate(memory_texts, 1)]) if memory_texts else "None"
        formatted_context = "\n".join([f"[C{i}] {ctx}" for i, ctx in enumerate(context_texts, 1)])
        prompt_template = PromptTemplate.from_template(RAG_WITH_MEMORY_PROMPT)
        prompt = prompt_template.format(
            memories=formatted_memories,
            context=formatted_context,
            query=query_text
        )
        llm_response = self.llm_manager.generate(
            prompt=prompt,
            system_message=RAG_WITH_MEMORY_SYSTEM,
            temperature=temperature
        )
        generation_time = time.time() - generation_start
        
        # Store interaction as episodic memory
        if store_interaction:
            interaction_content = f"Q: {query_text}\nA: {llm_response.content}"
            metadata = {"query": query_text}
            if session_id:
                metadata["session_id"] = session_id
            if user_id:
                metadata["user_id"] = user_id
            
            self.ltm.store_memory(
                content=interaction_content,
                memory_type=MemoryType.EPISODIC,
                source=f"conversation_{session_id}" if session_id else "conversation",
                metadata=metadata
            )
        
        # Format sources and memories
        sources = self._format_sources(search_results) if include_sources else []
        memories_used = [
            {
                "content": m.content,
                "type": m.memory_type.value,
                "importance": m.importance,
                "created_at": m.created_at.isoformat()
            }
            for m in relevant_memories
        ]
        
        total_time = time.time() - start_time
        
        return MemoryRAGResponse(
            answer=llm_response.content,
            sources=sources,
            query=query_text,
            tokens_used=llm_response.tokens_used,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            metadata={
                "num_sources": len(search_results),
                "num_memories": len(relevant_memories),
                "model": llm_response.model,
                "session_id": session_id,
                "user_id": user_id
            },
            memories_used=memories_used,
            memory_retrieval_time=memory_retrieval_time
        )
    
    def get_memory_stats(self) -> Dict:
        """Get memory system statistics"""
        return self.ltm.get_memory_stats()
    
    def add_fact(
        self,
        fact: str,
        importance: float = 0.8,
        tags: Optional[List[str]] = None
    ):
        """Add a semantic fact to long-term memory"""
        self.ltm.store_memory(
            content=fact,
            memory_type=MemoryType.SEMANTIC,
            source="user_fact",
            tags=tags or [],
            importance=importance,
            metadata={"user_marked_important": True}
        )
