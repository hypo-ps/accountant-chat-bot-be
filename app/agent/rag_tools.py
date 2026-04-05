"""
RAG-related tools for the agent.

These tools allow the agent to intelligently decide when to search
the knowledge base for relevant information.
"""

from typing import Any, Optional

from app.agent.tools import BaseTool, ToolResult
from app.core.logging import get_logger
from app.rag.retriever import Retriever

logger = get_logger(__name__)


class SearchKnowledgeBaseTool(BaseTool):
    """
    Tool for searching the knowledge base / document store.
    
    The agent will use this tool when it needs to find information
    from ingested documents rather than relying on its training data.
    """
    
    def __init__(self, retriever: Retriever, top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k
    
    @property
    def name(self) -> str:
        return "search_knowledge_base"
    
    @property
    def description(self) -> str:
        return """Search the internal knowledge base for relevant information.
Use this tool when:
- The user asks about company-specific information (policies, procedures, products)
- The user references documents, manuals, or internal resources
- You need factual information that may be in the document store
- The user asks "according to..." or "what does the policy say..."

Do NOT use this tool for:
- General knowledge questions (math, science, history)
- Casual conversation or greetings
- Creative writing or brainstorming
- Questions you can confidently answer from training data"""
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant documents. Be specific and include key terms."
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-10). Default is 5.",
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, query: str, num_results: int = 5) -> ToolResult:
        """Search the knowledge base and return relevant documents."""
        try:
            logger.info("Searching knowledge base", query=query, top_k=num_results)
            
            results = await self.retriever.retrieve(
                query=query,
                top_k=min(num_results, self.top_k),
            )
            
            if not results:
                return ToolResult(
                    success=True,
                    output="No relevant documents found in the knowledge base.",
                    data={"results": [], "count": 0}
                )
            
            # Format results for the LLM
            formatted_results = []
            output_parts = [f"Found {len(results)} relevant document(s):\n"]
            
            for i, result in enumerate(results, 1):
                source = result.document.metadata.get("source", "Unknown")
                score = result.score
                content = result.document.content
                
                # Truncate long content
                if len(content) > 500:
                    content = content[:500] + "..."
                
                output_parts.append(f"[{i}] Source: {source} (relevance: {score:.2f})")
                output_parts.append(f"    {content}\n")
                
                formatted_results.append({
                    "source": source,
                    "score": score,
                    "content": result.document.content,
                    "metadata": result.document.metadata,
                })
            
            return ToolResult(
                success=True,
                output="\n".join(output_parts),
                data={"results": formatted_results, "count": len(results)}
            )
            
        except Exception as e:
            logger.error("Knowledge base search failed", error=str(e))
            return ToolResult(
                success=False,
                output=f"Failed to search knowledge base: {str(e)}",
                error=str(e)
            )


class GetDocumentTool(BaseTool):
    """
    Tool for retrieving a specific document by ID.
    
    Useful when the agent needs more context from a previously found document.
    """
    
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
    
    @property
    def name(self) -> str:
        return "get_document"
    
    @property
    def description(self) -> str:
        return """Retrieve the full content of a specific document by its ID.
Use this when you need more context from a document found via search_knowledge_base."""
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The document ID to retrieve"
                }
            },
            "required": ["document_id"]
        }
    
    async def execute(self, document_id: str) -> ToolResult:
        """Retrieve a document by ID."""
        try:
            document = await self.retriever.vectordb.get_document(document_id)
            
            if not document:
                return ToolResult(
                    success=False,
                    output=f"Document not found: {document_id}",
                    error="Document not found"
                )
            
            return ToolResult(
                success=True,
                output=f"Document: {document.metadata.get('source', 'Unknown')}\n\n{document.content}",
                data={
                    "document_id": document.id,
                    "content": document.content,
                    "metadata": document.metadata,
                }
            )
            
        except Exception as e:
            logger.error("Get document failed", document_id=document_id, error=str(e))
            return ToolResult(
                success=False,
                output=f"Failed to retrieve document: {str(e)}",
                error=str(e)
            )
