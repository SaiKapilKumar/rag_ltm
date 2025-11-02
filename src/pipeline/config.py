from dataclasses import dataclass
import os
import yaml

@dataclass
class PipelineConfig:
    """Configuration for RAG pipeline"""
    
    # Retrieval
    vector_db_type: str = "faiss"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    similarity_threshold: float = 0.7
    
    # Generation
    llm_provider: str = "azure_openai"
    llm_model: str = "gpt-4.1"
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # Context
    max_context_length: int = 3000
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Pipeline
    enable_memory: bool = True
    memory_retrieval_k: int = 3
    document_retrieval_k: int = 5
    
    # Memory
    storage_backend: str = "sqlite"
    db_path: str = "../data/memory/ltm.db"
    decay_enabled: bool = True
    decay_rate: float = 0.01
    consolidation_interval: int = 86400
    importance_threshold: float = 0.3
    max_memories: int = 10000

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from YAML file"""
        if not os.path.exists(path):
            return cls()
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        if 'retrieval' in data:
            config.vector_db_type = data['retrieval'].get('vector_db_type', config.vector_db_type)
            config.embedding_model = data['retrieval'].get('embedding_model', config.embedding_model)
            config.top_k = data['retrieval'].get('top_k', config.top_k)
            config.similarity_threshold = data['retrieval'].get('similarity_threshold', config.similarity_threshold)
            config.chunk_size = data['retrieval'].get('chunk_size', config.chunk_size)
            config.chunk_overlap = data['retrieval'].get('chunk_overlap', config.chunk_overlap)
        
        if 'generation' in data:
            config.llm_provider = data['generation'].get('llm_provider', config.llm_provider)
            config.llm_model = data['generation'].get('model', config.llm_model)
            config.temperature = data['generation'].get('temperature', config.temperature)
            config.max_tokens = data['generation'].get('max_tokens', config.max_tokens)
        
        if 'pipeline' in data:
            config.enable_memory = data['pipeline'].get('enable_memory', config.enable_memory)
            config.memory_retrieval_k = data['pipeline'].get('memory_retrieval_k', config.memory_retrieval_k)
            config.document_retrieval_k = data['pipeline'].get('document_retrieval_k', config.document_retrieval_k)
        
        if 'memory' in data:
            config.storage_backend = data['memory'].get('storage_backend', config.storage_backend)
            config.db_path = data['memory'].get('db_path', config.db_path)
            config.decay_enabled = data['memory'].get('decay_enabled', config.decay_enabled)
            config.decay_rate = data['memory'].get('decay_rate', config.decay_rate)
            config.consolidation_interval = data['memory'].get('consolidation_interval', config.consolidation_interval)
            config.importance_threshold = data['memory'].get('importance_threshold', config.importance_threshold)
            config.max_memories = data['memory'].get('max_memories', config.max_memories)
        
        return config
