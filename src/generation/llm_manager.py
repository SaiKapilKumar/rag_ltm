import os
from typing import Optional
from .llm import LLMResponse
from .azure_openai_llm import AzureOpenAILLM

class LLMManager:
    """Manages multiple LLM providers with fallback"""
    
    def __init__(self, provider: str = "azure_openai", model: str = "gpt-4.1"):
        self.provider = provider
        self.model = model
        self.providers = {}
        
        if provider == "azure_openai" and os.getenv("AZURE_OPENAI_API_KEY"):
            self.providers["azure_openai"] = AzureOpenAILLM(model=model)
            self.primary_provider = "azure_openai"
        else:
            raise ValueError(f"Provider {provider} not configured or API key missing")
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """Generate response using primary provider"""
        provider = self.providers.get(self.primary_provider)
        if not provider:
            raise ValueError("No provider available")
        
        return provider.generate(
            prompt=prompt,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def count_tokens(self, text: str, provider_name: Optional[str] = None) -> int:
        """Count tokens using specified or primary provider"""
        provider_name = provider_name or self.primary_provider
        provider = self.providers.get(provider_name)
        if not provider:
            return len(text) // 4
        return provider.count_tokens(text)
