from typing import List, Dict
from dataclasses import dataclass
import uuid

@dataclass
class Chunk:
    """Represents a text chunk"""
    content: str
    chunk_id: str
    document_id: str
    metadata: Dict
    start_char: int
    end_char: int

class TextChunker:
    """Splits documents into chunks"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, document_id: str, metadata: Dict = None) -> List[Chunk]:
        """Split text into overlapping chunks"""
        if metadata is None:
            metadata = {}
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            if end < text_length and text[end] not in [' ', '\n', '.', '!', '?']:
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_content = text[start:end].strip()
            
            if chunk_content:
                chunk = Chunk(
                    content=chunk_content,
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    metadata=metadata,
                    start_char=start,
                    end_char=end
                )
                chunks.append(chunk)
            
            start = end - self.overlap if end < text_length else end
        
        return chunks
