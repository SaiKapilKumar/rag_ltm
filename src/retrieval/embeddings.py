from typing import List
from sentence_transformers import SentenceTransformer
import os
from openai import AzureOpenAI

class EmbeddingManager:
    """Manages embedding generation for documents and queries"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", provider: str = "sentence-transformers"):
        self.provider = provider
        self.model_name = model_name
        
        if provider == "sentence-transformers":
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        elif provider == "azure_openai":
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT")
            )
            self.deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "embeddings")
            self.dimension = 1536
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if self.provider == "sentence-transformers":
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        elif self.provider == "azure_openai":
            response = self.client.embeddings.create(
                input=texts,
                model=self.deployment
            )
            return [item.embedding for item in response.data]
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension
