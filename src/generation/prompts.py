from typing import List

class PromptTemplate:
    """Template for RAG prompts"""
    
    RAG_SYSTEM_MESSAGE = """You are a helpful AI assistant that answers questions based on the provided context.
Always use the context to inform your answers. If the context doesn't contain relevant information, say so clearly.
Cite specific parts of the context when possible."""
    
    RAG_WITH_MEMORY_SYSTEM = """You are a helpful AI assistant with memory of past conversations.
Answer questions based on:
1. The provided context from documents
2. Your memory of previous interactions
3. Your general knowledge

Always prioritize context and memory over general knowledge. Be clear about the source of your information."""
    
    @staticmethod
    def format_rag_prompt(query: str, context: List[str], include_instructions: bool = True) -> str:
        """Format RAG prompt with query and context"""
        prompt_parts = []
        
        if include_instructions:
            prompt_parts.append("Use the following context to answer the question. Be concise and accurate.")
            prompt_parts.append("")
        
        prompt_parts.append("CONTEXT:")
        for i, ctx in enumerate(context, 1):
            prompt_parts.append(f"[{i}] {ctx}")
        
        prompt_parts.append("")
        prompt_parts.append(f"QUESTION: {query}")
        prompt_parts.append("")
        prompt_parts.append("ANSWER:")
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def format_rag_with_memory_prompt(
        query: str,
        context: List[str],
        memories: List[str],
        include_instructions: bool = True
    ) -> str:
        """Format RAG prompt with context and memories"""
        prompt_parts = []
        
        if include_instructions:
            prompt_parts.append("Use the context and memories to answer the question accurately.")
            prompt_parts.append("")
        
        if memories:
            prompt_parts.append("RELEVANT MEMORIES:")
            for i, mem in enumerate(memories, 1):
                prompt_parts.append(f"[M{i}] {mem}")
            prompt_parts.append("")
        
        prompt_parts.append("CONTEXT:")
        for i, ctx in enumerate(context, 1):
            prompt_parts.append(f"[C{i}] {ctx}")
        
        prompt_parts.append("")
        prompt_parts.append(f"QUESTION: {query}")
        prompt_parts.append("")
        prompt_parts.append("ANSWER:")
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def format_summarization_prompt(text: str, max_words: int = 100) -> str:
        """Format prompt for summarization"""
        return f"""Summarize the following text in no more than {max_words} words:

TEXT:
{text}

SUMMARY:"""
    
    @staticmethod
    def format_memory_extraction_prompt(conversation: str) -> str:
        """Extract key facts for memory storage"""
        return f"""Extract key facts and information from this conversation that should be remembered long-term.
Focus on: user preferences, important details, factual information, and context.

CONVERSATION:
{conversation}

KEY FACTS TO REMEMBER (one per line):"""
