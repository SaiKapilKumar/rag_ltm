from abc import ABC, abstractmethod
from typing import Optional, Dict
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """LLM response with metadata"""
    content: str
    model: str
    tokens_used: int
    finish_reason: str
    metadata: Dict

class LLMBase(ABC):
    """Abstract LLM interface"""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """Generate response from prompt"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass
