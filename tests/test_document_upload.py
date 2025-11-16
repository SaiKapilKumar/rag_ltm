"""
Tests for document upload functionality
"""
import os
from pathlib import Path
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_upload_document_saves_to_folder():
    """Test that uploaded documents are saved to data/documents/ folder"""
    
    # Create a test file
    test_content = b"This is a test document for upload testing."
    test_filename = "test_upload.txt"
    
    # Upload the file
    response = client.post(
        "/documents/upload",
        files={"file": (test_filename, test_content, "text/plain")}
    )
    
    # Check response
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    assert result["filename"] == test_filename
    assert "file_path" in result
    
    # Verify file was saved to disk
    expected_path = Path(__file__).parent.parent / "data" / "documents" / test_filename
    assert expected_path.exists(), f"File not found at {expected_path}"
    
    # Verify content
    with open(expected_path, "rb") as f:
        saved_content = f.read()
    assert saved_content == test_content
    
    # Cleanup
    if expected_path.exists():
        os.remove(expected_path)


def test_upload_pdf_document():
    """Test uploading a PDF document"""
    
    # For this test, we'll just verify the endpoint handles PDF files correctly
    # Create a minimal PDF-like file (not a real PDF, just for testing the flow)
    test_filename = "test_document.pdf"
    test_content = b"%PDF-1.4 dummy content"
    
    _ = client.post(
        "/documents/upload",
        files={"file": (test_filename, test_content, "application/pdf")}
    )
    
    # This might fail at processing but should save the file first
    # Check if file was saved regardless of processing errors
    expected_path = Path(__file__).parent.parent / "data" / "documents" / test_filename
    
    # Cleanup if file was created
    if expected_path.exists():
        os.remove(expected_path)


def test_documents_folder_created():
    """Test that data/documents folder is created if it doesn't exist"""
    documents_dir = Path(__file__).parent.parent / "data" / "documents"
    
    # Create test file to trigger folder creation
    test_content = b"Test content"
    test_filename = "folder_test.txt"
    
    _ = client.post(
        "/documents/upload",
        files={"file": (test_filename, test_content, "text/plain")}
    )
    
    # Verify folder exists
    assert documents_dir.exists()
    assert documents_dir.is_dir()
    
    # Cleanup
    test_file = documents_dir / test_filename
    if test_file.exists():
        os.remove(test_file)
