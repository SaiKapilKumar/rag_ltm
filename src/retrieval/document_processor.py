from pathlib import Path
import io
from pypdf import PdfReader

class DocumentProcessor:
    """Process different document types and extract text"""
    
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF file bytes"""
        try:
            pdf_file = io.BytesIO(file_content)
            reader = PdfReader(pdf_file)
            
            text_parts = []
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(f"[Page {page_num + 1}]\n{text}")
                except Exception as e:
                    print(f"Error extracting text from page {page_num + 1}: {str(e)}")
                    continue
            
            return "\n\n".join(text_parts)
        except Exception as e:
            raise ValueError(f"Failed to process PDF: {str(e)}")
    
    @staticmethod
    def extract_text_from_txt(file_content: bytes) -> str:
        """Extract text from plain text file"""
        try:
            return file_content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return file_content.decode("latin-1")
            except Exception as e:
                raise ValueError(f"Failed to decode text file: {str(e)}")
    
    @staticmethod
    def extract_text_from_md(file_content: bytes) -> str:
        """Extract text from markdown file"""
        return DocumentProcessor.extract_text_from_txt(file_content)
    
    @staticmethod
    def process_file(file_content: bytes, filename: str) -> str:
        """Process file based on extension and return extracted text"""
        file_extension = Path(filename).suffix.lower()
        
        if file_extension == ".pdf":
            return DocumentProcessor.extract_text_from_pdf(file_content)
        elif file_extension in [".txt", ".text"]:
            return DocumentProcessor.extract_text_from_txt(file_content)
        elif file_extension in [".md", ".markdown"]:
            return DocumentProcessor.extract_text_from_md(file_content)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported types: .pdf, .txt, .md")
    
    @staticmethod
    def get_supported_extensions() -> list:
        """Return list of supported file extensions"""
        return [".pdf", ".txt", ".md", ".text", ".markdown"]
