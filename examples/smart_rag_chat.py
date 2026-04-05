#!/usr/bin/env python3
"""
Smart RAG Chat Example

Demonstrates how the agent intelligently decides when to use RAG:
- General questions → Direct LLM response
- Document-specific questions → Searches knowledge base first

PREREQUISITES:
1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant
2. Set OPENAI_API_KEY in .env
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from app.llm.factory import create_llm
from app.rag.embeddings import create_embeddings
from app.rag.retriever import Retriever
from app.agent.rag_agent import RAGAgent
from app.vectordb.qdrant import QdrantVectorDB
from app.documents.base import LoadedDocument
from app.documents.text_splitter import TextSplitter


# Sample company documents to ingest
COMPANY_DOCUMENTS = [
    {
        "content": """
        VACATION POLICY
        
        All full-time employees are entitled to:
        - 15 days of paid vacation per year for the first 3 years
        - 20 days of paid vacation per year after 3 years
        - 25 days of paid vacation per year after 5 years
        
        Vacation requests must be submitted at least 2 weeks in advance.
        Unused vacation days can be carried over (max 5 days).
        """,
        "source": "hr_policies/vacation.txt"
    },
    {
        "content": """
        EXPENSE REIMBURSEMENT POLICY
        
        Employees can submit expense reports for:
        - Travel expenses (flights, hotels, meals up to $75/day)
        - Office supplies (requires manager approval over $100)
        - Professional development (up to $2000/year)
        
        Submit expense reports within 30 days of incurring the expense.
        Receipts required for all expenses over $25.
        """,
        "source": "hr_policies/expenses.txt"
    },
]


async def main():
    print("=" * 70)
    print("Smart RAG Chat - Agent Decides When to Use Knowledge Base")
    print("=" * 70)
    
    # Initialize components
    print("\n[1] Initializing components...")
    llm = create_llm()
    embeddings = create_embeddings()
    
    # Connect to Qdrant
    vectordb = QdrantVectorDB(
        collection_name="company_docs",
        host="localhost",
        port=6333,
        vector_size=1536,
    )
    await vectordb.initialize()
    print("    [OK] Connected to Qdrant")
    
    # Create retriever and ingest documents
    retriever = Retriever(vectordb=vectordb, embeddings=embeddings, top_k=3)
    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
    
    print("\n[2] Ingesting company documents...")
    for doc in COMPANY_DOCUMENTS:
        loaded = LoadedDocument(content=doc["content"], source=doc["source"])
        chunks = splitter.split_document(loaded)
        await retriever.add_documents(chunks)
        print(f"    [OK] Ingested: {doc['source']}")
    
    # Create RAG-enabled agent
    print("\n[3] Creating RAG Agent...")
    agent = RAGAgent(llm=llm, retriever=retriever)
    
    # Test questions - mix of general and document-specific
    test_questions = [
        # General question - should NOT use RAG
        ("What is 15% of 200?", "General math - no RAG needed"),
        
        # Document-specific - SHOULD use RAG
        ("How many vacation days do new employees get?", "Policy question - needs RAG"),
        
        # General question - should NOT use RAG
        ("What's the capital of France?", "General knowledge - no RAG needed"),
        
        # Document-specific - SHOULD use RAG
        ("What's the daily meal limit for travel expenses?", "Policy question - needs RAG"),
        
        # Greeting - should NOT use RAG
        ("Hello, how are you?", "Greeting - no RAG needed"),
    ]
    
    print("\n[4] Testing smart RAG decisions...")
    print("=" * 70)
    
    for question, description in test_questions:
        print(f"\n[{description}]")
        print(f"   Q: {question}")
        
        response = await agent.chat(message=question, use_tools=True)
        
        # Check if RAG was used
        rag_used = any(
            tc.get("tool") == "search_knowledge_base" 
            for tc in response.tool_calls_made
        )
        
        print(f"   RAG Used: {'YES' if rag_used else 'NO'}")
        print(f"   A: {response.content[:200]}{'...' if len(response.content) > 200 else ''}")
        
        if rag_used:
            for tc in response.tool_calls_made:
                if tc.get("tool") == "search_knowledge_base":
                    print(f"   Search query: \"{tc['arguments'].get('query', '')}\"")
        
        print("-" * 70)
    
    print("\nDemo complete!")
    print("\nThe agent automatically decided when to search the knowledge base")
    print("based on the nature of each question.")


if __name__ == "__main__":
    asyncio.run(main())
