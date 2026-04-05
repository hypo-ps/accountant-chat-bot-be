"""
Qdrant vector database implementation.

Qdrant is a high-performance vector database written in Rust.
Supports both self-hosted and cloud deployments.

Setup (Docker):
    docker run -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant

Setup (Cloud):
    1. Create account at https://cloud.qdrant.io
    2. Create a cluster
    3. Get API key and URL
"""

from typing import Any, Optional
from uuid import uuid4

from app.core.exceptions import VectorDBConnectionError, VectorDBError
from app.core.logging import get_logger
from app.vectordb.base import BaseVectorDB, Document, SearchResult

logger = get_logger(__name__)


class QdrantVectorDB(BaseVectorDB):
    """
    Qdrant vector database implementation.
    
    Supports:
    - Local/Docker deployment
    - Qdrant Cloud
    - Advanced filtering
    - Payload storage
    """
    
    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: int = 6333,
        url: Optional[str] = None,  # For cloud: https://xxx.cloud.qdrant.io
        api_key: Optional[str] = None,  # For cloud authentication
        vector_size: int = 1536,  # Must match your embedding dimension!
        distance: str = "Cosine",  # Cosine, Euclid, or Dot
        **kwargs: Any,
    ) -> None:
        super().__init__(collection_name, **kwargs)
        self.host = host
        self.port = port
        self.url = url
        self.api_key = api_key
        self.vector_size = vector_size
        self.distance = distance
        self.client = None
    
    async def initialize(self) -> None:
        """Initialize Qdrant client and create collection if needed."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except ImportError:
            raise VectorDBError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )
        
        try:
            # Connect to Qdrant
            if self.url:
                # Cloud deployment
                self.client = QdrantClient(url=self.url, api_key=self.api_key)
                logger.info("Connected to Qdrant Cloud", url=self.url)
            else:
                # Local/Docker deployment
                self.client = QdrantClient(host=self.host, port=self.port)
                logger.info("Connected to Qdrant", host=self.host, port=self.port)
            
            # Check if collection exists, create if not
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance[self.distance.upper()],
                    ),
                )
                logger.info(
                    "Created Qdrant collection",
                    collection=self.collection_name,
                    vector_size=self.vector_size,
                )
            else:
                logger.info("Using existing Qdrant collection", collection=self.collection_name)
                
        except Exception as e:
            raise VectorDBConnectionError(f"Failed to connect to Qdrant: {e}", original_error=e)
    
    async def add_documents(
        self,
        documents: list[Document],
        embeddings: Optional[list[list[float]]] = None,
    ) -> list[str]:
        """Add documents to Qdrant."""
        if not self.client:
            raise VectorDBError("Qdrant not initialized")
        
        if not embeddings or len(embeddings) != len(documents):
            raise VectorDBError("Embeddings required for all documents")
        
        try:
            from qdrant_client.http import models
            
            points = [
                models.PointStruct(
                    id=doc.id,
                    vector=embeddings[i],
                    payload={
                        "content": doc.content,
                        **doc.metadata,
                    },
                )
                for i, doc in enumerate(documents)
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            
            logger.info("Documents added to Qdrant", count=len(documents))
            return [doc.id for doc in documents]
            
        except Exception as e:
            raise VectorDBError(f"Failed to add documents: {e}", original_error=e)
    
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Search for similar documents in Qdrant."""
        if not self.client:
            raise VectorDBError("Qdrant not initialized")
        
        try:
            from qdrant_client.http import models
            
            # Build filter if provided
            qdrant_filter = None
            if filter_metadata:
                conditions = [
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                    for key, value in filter_metadata.items()
                ]
                qdrant_filter = models.Filter(must=conditions)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            
            search_results = []
            for result in results:
                payload = result.payload or {}
                content = payload.pop("content", "")
                doc = Document(
                    id=str(result.id),
                    content=content,
                    metadata=payload,
                )
                search_results.append(SearchResult(document=doc, score=result.score))
            
            return search_results

        except Exception as e:
            raise VectorDBError(f"Search failed: {e}", original_error=e)

    async def delete_documents(self, document_ids: list[str]) -> int:
        """Delete documents from Qdrant."""
        if not self.client:
            raise VectorDBError("Qdrant not initialized")

        try:
            from qdrant_client.http import models

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=document_ids),
            )
            logger.info("Documents deleted from Qdrant", count=len(document_ids))
            return len(document_ids)

        except Exception as e:
            raise VectorDBError(f"Delete failed: {e}", original_error=e)

    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID from Qdrant."""
        if not self.client:
            raise VectorDBError("Qdrant not initialized")

        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[document_id],
                with_payload=True,
            )

            if results:
                payload = results[0].payload or {}
                content = payload.pop("content", "")
                return Document(
                    id=str(results[0].id),
                    content=content,
                    metadata=payload,
                )
            return None

        except Exception as e:
            raise VectorDBError(f"Get document failed: {e}", original_error=e)

    async def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        embedding: Optional[list[float]] = None,
    ) -> bool:
        """Update a document in Qdrant."""
        if not self.client:
            raise VectorDBError("Qdrant not initialized")

        try:
            from qdrant_client.http import models

            # Get existing document
            existing = await self.get_document(document_id)
            if not existing:
                return False

            # Build updated payload
            new_payload = existing.metadata.copy()
            if content:
                new_payload["content"] = content
            if metadata:
                new_payload.update(metadata)

            # Update payload
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=new_payload,
                points=[document_id],
            )

            # Update vector if provided
            if embedding:
                self.client.update_vectors(
                    collection_name=self.collection_name,
                    points=[
                        models.PointVectors(id=document_id, vector=embedding)
                    ],
                )

            return True

        except Exception as e:
            logger.warning("Update document failed", document_id=document_id, error=str(e))
            return False

    async def count(self) -> int:
        """Get document count in Qdrant."""
        if not self.client:
            return 0

        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception:
            return 0

    async def clear(self) -> None:
        """Clear all documents from Qdrant collection."""
        if not self.client:
            return

        try:
            from qdrant_client.http import models

            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance[self.distance.upper()],
                ),
            )
            logger.info("Qdrant collection cleared", collection=self.collection_name)

        except Exception as e:
            raise VectorDBError(f"Clear failed: {e}", original_error=e)

    async def health_check(self) -> bool:
        """Check if Qdrant is accessible."""
        try:
            if self.client:
                self.client.get_collections()
                return True
            return False
        except Exception:
            return False
