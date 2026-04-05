#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) example.

This example shows how to:
1. Initialize the RAG pipeline with Qdrant
2. Ingest documents
3. Query with context retrieval

PREREQUISITES:
1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant
2. Set OPENAI_API_KEY in .env
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from app.llm.factory import create_llm
from app.rag.embeddings import create_embeddings
from app.rag.pipeline import RAGPipeline
from app.vectordb.qdrant import QdrantVectorDB


# Sample documents about accounting
SAMPLE_DOCUMENTS = [
    {
        "text": """
        Depreciation is an accounting method of allocating the cost of a tangible
        asset over its useful life. Businesses depreciate long-term assets for both
        tax and accounting purposes. The most common methods include:

        1. Straight-line depreciation: Equal expense each year
        2. Double declining balance: Accelerated depreciation
        3. Units of production: Based on usage or output
        4. Sum of years' digits: Another accelerated method
        """,
        "source": "accounting_basics.txt",
    },
    {
        "text": """
        Accounts Receivable (AR) represents money owed to a company by its customers
        for goods or services delivered but not yet paid for. It is recorded as a
        current asset on the balance sheet. Key metrics include:

        - Days Sales Outstanding (DSO): Average collection period
        - Aging Schedule: Categorizes receivables by age
        - Bad Debt Expense: Estimated uncollectible accounts
        - Allowance for Doubtful Accounts: Contra asset account
        """,
        "source": "receivables_guide.txt",
    },
    {
        "text": """
        The Generally Accepted Accounting Principles (GAAP) are a set of rules and
        standards used for financial reporting in the United States. Key principles:

        1. Revenue Recognition: When to record revenue
        2. Matching Principle: Match expenses with related revenues
        3. Full Disclosure: Report all relevant financial information
        4. Consistency: Use same methods across periods
        5. Materiality: Report significant items
        """,
        "source": "gaap_overview.txt",
    },
]


async def main():
    """Run a RAG example."""
    print("=" * 60)
    print("Accountant Chatbot - RAG Example with Qdrant")
    print("=" * 60)

    # Initialize components
    print("\n[1] Initializing components...")
    llm = create_llm()
    embeddings = create_embeddings()

    # Connect to Qdrant (make sure it's running!)
    print("\n[2] Connecting to Qdrant...")
    vectordb = QdrantVectorDB(
        collection_name="rag_example",
        host="localhost",
        port=6333,
        vector_size=1536,  # OpenAI embedding dimension
    )
    await vectordb.initialize()
    print("    [OK] Connected to Qdrant")

    # Create RAG pipeline
    rag_pipeline = RAGPipeline(
        llm=llm,
        vectordb=vectordb,
        embeddings=embeddings,
        chunk_size=500,
        chunk_overlap=50,
        top_k=3,
    )

    # Ingest sample documents
    print("\n[3] Ingesting documents...")
    for doc in SAMPLE_DOCUMENTS:
        result = await rag_pipeline.ingest_text(
            text=doc["text"],
            source=doc["source"],
        )
        print(f"    [OK] Ingested: {doc['source']} -> {result['chunks_created']} chunks")

    # Query the RAG pipeline
    print("\n[4] Querying with RAG...")
    print("-" * 60)

    questions = [
        "What are the different methods of depreciation?",
        "How do you calculate Days Sales Outstanding?",
        "What is the matching principle in GAAP?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")

        result = await rag_pipeline.query(
            question=question,
            include_sources=True,
        )

        print(f"\nAnswer: {result['answer'][:500]}...")
        print(f"\nSources used: {len(result.get('sources', []))}")
        if result.get('sources'):
            for src in result['sources'][:2]:
                print(f"  - {src['metadata'].get('source', 'unknown')} (score: {src['score']:.3f})")

        print("-" * 60)

    # Show Qdrant stats
    count = await vectordb.count()
    print(f"\nTotal vectors in Qdrant: {count}")

    print("\n" + "=" * 60)
    print("RAG example complete!")


if __name__ == "__main__":
    asyncio.run(main())
