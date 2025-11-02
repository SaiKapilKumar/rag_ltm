from typing import List, Optional, Dict
from dataclasses import dataclass
import time

from src.retrieval.retriever import DocumentRetriever, SearchResult
from src.generation.llm_manager import LLMManager
from src.generation.prompts import PromptTemplate

@dataclass
class RAGResponse:
    """RAG pipeline response"""
    answer: str
    sources: List[Dict]
    query: str
    tokens_used: int
    retrieval_time: float
    generation_time: float
    total_time: float
    metadata: Dict

class RAGPipeline:
    """Basic RAG pipeline"""
    
    def __init__(
        self,
        retriever: DocumentRetriever,
        llm_manager: LLMManager,
        top_k: int = 5,
        max_context_length: int = 3000
    ):
        self.retriever = retriever
        self.llm_manager = llm_manager
        self.top_k = top_k
        self.max_context_length = max_context_length
    
    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        temperature: float = 0.7,
        include_sources: bool = True
    ) -> RAGResponse:
        """Process a query through the RAG pipeline"""
        start_time = time.time()
        
        # Retrieval
        retrieval_start = time.time()
        k = top_k or self.top_k
        search_results = self.retriever.retrieve(query_text, k=k)
        retrieval_time = time.time() - retrieval_start
        
        # Assemble context
        context_texts = self._assemble_context(search_results)
        
        # Generate response
        generation_start = time.time()
        prompt = PromptTemplate.format_rag_prompt(query_text, context_texts)
        llm_response = self.llm_manager.generate(
            prompt=prompt,
            system_message=PromptTemplate.RAG_SYSTEM_MESSAGE,
            temperature=temperature
        )
        generation_time = time.time() - generation_start
        
        # Format sources
        sources = self._format_sources(search_results) if include_sources else []
        
        total_time = time.time() - start_time
        
        return RAGResponse(
            answer=llm_response.content,
            sources=sources,
            query=query_text,
            tokens_used=llm_response.tokens_used,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            metadata={
                "num_sources": len(search_results),
                "model": llm_response.model
            }
        )
    
    def _assemble_context(self, search_results: List[SearchResult]) -> List[str]:
        """Assemble context from search results"""
        context_texts = []
        total_length = 0
        
        for result in search_results:
            text = result.document.content
            text_length = len(text)
            
            if total_length + text_length > self.max_context_length:
                remaining = self.max_context_length - total_length
                if remaining > 100:
                    text = text[:remaining] + "..."
                    context_texts.append(text)
                break
            
            context_texts.append(text)
            total_length += text_length
        
        return context_texts
    
    def _format_sources(self, search_results: List[SearchResult]) -> List[Dict]:
        """Format sources for response"""
        sources = []
        for i, result in enumerate(search_results, 1):
            sources.append({
                "id": i,
                "content": result.document.content,
                "score": result.score,
                "metadata": result.document.metadata
            })
        return sources
    
    def add_documents(
        self,
        texts: List[str],
        doc_ids: List[str],
        metadatas: Optional[List[Dict]] = None
    ):
        """Add documents to the pipeline"""
        print(f"🔄 Pipeline: Adding {len(texts)} document(s) to RAG system...")
        start_time = time.time()
        
        self.retriever.add_documents(texts, doc_ids, metadatas)
        
        total_time = time.time() - start_time
        print(f"✅ Pipeline: Document(s) successfully added in {total_time:.3f}s")
