import os
from openai import AzureOpenAI
import tiktoken
from typing import Optional
from .llm import LLMBase, LLMResponse

class AzureOpenAILLM(LLMBase):
    """Azure OpenAI LLM implementation"""
    
    def __init__(self, model: str = "gpt-4.1"):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE")
        )
        self.model = model
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", model)
        try:
            self.encoder = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Azure OpenAI"""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.deployment_name,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            finish_reason=response.choices[0].finish_reason,
            metadata={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0
            }
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        return len(self.encoder.encode(text))
