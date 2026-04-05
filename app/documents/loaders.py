"""
Document loaders for various file types.
"""

import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiofiles

from app.core.exceptions import DocumentParsingError
from app.core.logging import get_logger
from app.documents.base import BaseDocumentLoader, LoadedDocument, document_loader_registry

logger = get_logger(__name__)


class TextLoader(BaseDocumentLoader):
    """Loader for plain text files."""
    
    @property
    def supported_extensions(self) -> list[str]:
        return [".txt", ".text"]
    
    async def load(self, file_path: Path) -> LoadedDocument:
        """Load a text file."""
        try:
            async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
                content = await f.read()
            
            stat = file_path.stat()
            
            return LoadedDocument(
                content=content,
                source=str(file_path),
                metadata={
                    "filename": file_path.name,
                    "file_type": "text",
                    "file_size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                },
            )
        except Exception as e:
            raise DocumentParsingError(f"Failed to load text file: {e}", original_error=e)
    
    async def load_from_bytes(
        self,
        content: bytes,
        filename: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> LoadedDocument:
        """Load text from bytes."""
        try:
            text_content = content.decode("utf-8")
            return LoadedDocument(
                content=text_content,
                source=filename,
                metadata={
                    "filename": filename,
                    "file_type": "text",
                    "file_size": len(content),
                    **(metadata or {}),
                },
            )
        except Exception as e:
            raise DocumentParsingError(f"Failed to parse text content: {e}", original_error=e)


class MarkdownLoader(BaseDocumentLoader):
    """Loader for Markdown files."""
    
    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown"]
    
    async def load(self, file_path: Path) -> LoadedDocument:
        """Load a Markdown file."""
        try:
            async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
                content = await f.read()
            
            stat = file_path.stat()
            
            return LoadedDocument(
                content=content,
                source=str(file_path),
                metadata={
                    "filename": file_path.name,
                    "file_type": "markdown",
                    "file_size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                },
            )
        except Exception as e:
            raise DocumentParsingError(f"Failed to load markdown file: {e}", original_error=e)
    
    async def load_from_bytes(
        self,
        content: bytes,
        filename: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> LoadedDocument:
        """Load markdown from bytes."""
        try:
            text_content = content.decode("utf-8")
            return LoadedDocument(
                content=text_content,
                source=filename,
                metadata={
                    "filename": filename,
                    "file_type": "markdown",
                    "file_size": len(content),
                    **(metadata or {}),
                },
            )
        except Exception as e:
            raise DocumentParsingError(f"Failed to parse markdown content: {e}", original_error=e)


class PDFLoader(BaseDocumentLoader):
    """
    Loader for PDF files.
    
    Note: Requires pypdf to be installed (included in [all] extras).
    """
    
    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]
    
    async def load(self, file_path: Path) -> LoadedDocument:
        """Load a PDF file."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise DocumentParsingError(
                "pypdf is required for PDF support. Install with: pip install pypdf"
            )
        
        try:
            reader = PdfReader(str(file_path))
            
            text_content = ""
            for page in reader.pages:
                text_content += page.extract_text() + "\n\n"
            
            stat = file_path.stat()
            
            return LoadedDocument(
                content=text_content.strip(),
                source=str(file_path),
                metadata={
                    "filename": file_path.name,
                    "file_type": "pdf",
                    "file_size": stat.st_size,
                    "num_pages": len(reader.pages),
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                },
            )
        except ImportError:
            raise
        except Exception as e:
            raise DocumentParsingError(f"Failed to load PDF file: {e}", original_error=e)
    
    async def load_from_bytes(
        self,
        content: bytes,
        filename: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> LoadedDocument:
        """Load PDF from bytes."""
        try:
            from pypdf import PdfReader
            import io
        except ImportError:
            raise DocumentParsingError(
                "pypdf is required for PDF support. Install with: pip install pypdf"
            )
        
        try:
            reader = PdfReader(io.BytesIO(content))
            
            text_content = ""
            for page in reader.pages:
                text_content += page.extract_text() + "\n\n"
            
            return LoadedDocument(
                content=text_content.strip(),
                source=filename,
                metadata={
                    "filename": filename,
                    "file_type": "pdf",
                    "file_size": len(content),
                    "num_pages": len(reader.pages),
                    **(metadata or {}),
                },
            )
        except ImportError:
            raise
        except Exception as e:
            raise DocumentParsingError(f"Failed to parse PDF content: {e}", original_error=e)


# Register default loaders
document_loader_registry.register(TextLoader())
document_loader_registry.register(MarkdownLoader())
document_loader_registry.register(PDFLoader())
