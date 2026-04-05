"""Tests for text splitter."""

import pytest
from app.documents.text_splitter import TextSplitter
from app.documents.base import LoadedDocument


class TestTextSplitter:
    """Tests for TextSplitter class."""
    
    def test_split_short_text(self):
        """Test splitting text shorter than chunk size."""
        splitter = TextSplitter(chunk_size=1000)
        text = "This is a short text."
        
        chunks = splitter.split_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_by_paragraph(self):
        """Test splitting on paragraph boundaries."""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=0)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        
        chunks = splitter.split_text(text)
        
        assert len(chunks) >= 1
    
    def test_split_long_text(self):
        """Test splitting text longer than chunk size."""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
        text = "A" * 100
        
        chunks = splitter.split_text(text)
        
        assert len(chunks) >= 2
    
    def test_chunk_overlap(self):
        """Test that chunks have overlap."""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "First sentence here. Second sentence here. Third sentence here."
        
        chunks = splitter.split_text(text)
        
        # With overlap, chunks should share some text
        if len(chunks) > 1:
            # The second chunk should contain some overlap from first
            assert len(chunks[1]) > 10
    
    def test_empty_text(self):
        """Test splitting empty text."""
        splitter = TextSplitter()
        chunks = splitter.split_text("")
        
        assert len(chunks) == 0
    
    def test_whitespace_only(self):
        """Test splitting whitespace-only text."""
        splitter = TextSplitter()
        chunks = splitter.split_text("   \n\n   ")
        
        assert len(chunks) == 0
    
    def test_split_document(self):
        """Test splitting a LoadedDocument."""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
        doc = LoadedDocument(
            content="First sentence here. Second sentence here.",
            source="test.txt",
            metadata={"author": "test"},
        )
        
        chunks = splitter.split_document(doc)
        
        assert len(chunks) >= 1
        assert all(c.document_id == doc.document_id for c in chunks)
        assert all("source" in c.metadata for c in chunks)
    
    def test_chunk_indices(self):
        """Test that chunk indices are correct."""
        splitter = TextSplitter(chunk_size=20, chunk_overlap=0)
        doc = LoadedDocument(
            content="A" * 60,
            source="test.txt",
        )
        
        chunks = splitter.split_document(doc)
        
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
    
    def test_preserve_metadata(self):
        """Test that metadata is preserved in chunks."""
        splitter = TextSplitter(chunk_size=50)
        doc = LoadedDocument(
            content="Some content here.",
            source="test.txt",
            metadata={"key": "value"},
        )
        
        chunks = splitter.split_document(doc, additional_metadata={"extra": "data"})
        
        assert chunks[0].metadata["key"] == "value"
        assert chunks[0].metadata["extra"] == "data"
        assert chunks[0].metadata["source"] == "test.txt"
