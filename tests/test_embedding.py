import pytest
import os
import shutil
from unittest.mock import patch, MagicMock
from src.rag.embedding import create_vector_db

@patch("src.rag.embedding.os.path.exists")
@patch("src.rag.embedding.os.getenv")
@patch("builtins.print")
def test_create_vector_db_no_docs_dir(mock_print, mock_getenv, mock_exists):
    mock_exists.return_value = False
    
    create_vector_db("fake_dir", "fake_db")
    
    mock_exists.assert_called_with("fake_dir")
    mock_print.assert_called_with("Directory fake_dir not found. Run scrape_data.py first.")

@patch("src.rag.embedding.os.path.exists")
@patch("src.rag.embedding.os.getenv")
@patch("builtins.print")
def test_create_vector_db_no_api_key(mock_print, mock_getenv, mock_exists):
    mock_exists.return_value = True
    mock_getenv.return_value = None
    
    create_vector_db("fake_dir", "fake_db")
    
    mock_getenv.assert_called_with("OPENAI_API_KEY")
    mock_print.assert_any_call("Error: OPENAI_API_KEY not found in environment variables.")

@patch("src.rag.embedding.os.path.exists")
@patch("src.rag.embedding.os.getenv")
@patch("src.rag.embedding.DirectoryLoader")
@patch("builtins.print")
def test_create_vector_db_no_docs(mock_print, mock_loader, mock_getenv, mock_exists):
    mock_exists.return_value = True
    mock_getenv.return_value = "fake-api-key"
    
    loader_instance = MagicMock()
    loader_instance.load.return_value = []
    mock_loader.return_value = loader_instance
    
    create_vector_db("fake_dir", "fake_db")
    
    mock_print.assert_any_call("No documents found.")

@patch("src.rag.embedding.os.path.exists")
@patch("src.rag.embedding.os.getenv")
@patch("src.rag.embedding.DirectoryLoader")
@patch("src.rag.embedding.RecursiveCharacterTextSplitter")
@patch("src.rag.embedding.OpenAIEmbeddings")
@patch("src.rag.embedding.FAISS")
@patch("builtins.print")
def test_create_vector_db_success(mock_print, mock_faiss, mock_embeddings, mock_splitter, mock_loader, mock_getenv, mock_exists):
    mock_exists.return_value = True
    mock_getenv.return_value = "fake-api-key"
    
    # Mock documents
    mock_doc = MagicMock()
    mock_doc.page_content = "Source: https://example.com\nSome content."
    mock_doc.metadata = {}
    
    loader_instance = MagicMock()
    loader_instance.load.return_value = [mock_doc]
    mock_loader.return_value = loader_instance
    
    # Mock splitter
    splitter_instance = MagicMock()
    splitter_instance.split_documents.return_value = ["chunk1", "chunk2"]
    mock_splitter.return_value = splitter_instance
    
    # Mock FAISS
    faiss_instance = MagicMock()
    mock_faiss.from_documents.return_value = faiss_instance
    
    create_vector_db("fake_dir", "fake_db")
    
    assert mock_doc.metadata['source'] == "https://example.com"
    splitter_instance.split_documents.assert_called_once_with([mock_doc])
    mock_faiss.from_documents.assert_called_once()
    faiss_instance.save_local.assert_called_once_with("fake_db")
    mock_print.assert_any_call("Vector database created successfully in fake_db")
