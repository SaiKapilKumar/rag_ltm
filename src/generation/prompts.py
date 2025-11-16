# System Messages
RAG_SYSTEM_MESSAGE = """You are a helpful AI assistant that answers questions based on the provided context.
Always use the context to inform your answers. If the context doesn't contain relevant information, say so clearly.
Cite specific parts of the context when possible."""

RAG_WITH_MEMORY_SYSTEM = """You are a helpful AI assistant with memory of past conversations.
Answer questions based on:
1. The provided context from documents
2. Your memory of previous interactions
3. Your general knowledge

Always prioritize context and memory over general knowledge. Be clear about the source of your information."""

# Basic RAG Prompt
RAG_PROMPT = """Use the following context to answer the question. Be concise and accurate.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

# RAG with Memory Prompt
RAG_WITH_MEMORY_PROMPT = """Use the context and memories to answer the question accurately.

RELEVANT MEMORIES:
{memories}

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

# Summarization Prompt
SUMMARIZATION_PROMPT = """Summarize the following text in no more than {max_words} words:

TEXT:
{text}

SUMMARY:"""

# Memory Extraction Prompt
MEMORY_EXTRACTION_PROMPT = """Extract key facts and information from this conversation that should be remembered long-term.
Focus on: user preferences, important details, factual information, and context.

CONVERSATION:
{conversation}

KEY FACTS TO REMEMBER (one per line):"""


# ============================================================================
# Quick access functions for common use cases
# ============================================================================

def get_rag_prompt(use_memory: bool = False) -> str:
    """Get appropriate RAG prompt based on whether memory is used"""
    return RAG_WITH_MEMORY_PROMPT if use_memory else RAG_PROMPT

def get_system_message(use_memory: bool = False) -> str:
    """Get appropriate system message based on whether memory is used"""
    return RAG_WITH_MEMORY_SYSTEM if use_memory else RAG_SYSTEM_MESSAGE
