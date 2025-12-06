"""
Kerala Ayurveda RAG System - Part A Implementation
Implements document chunking, retrieval, and Q&A with citations
"""

import os
from typing import List, Dict, Tuple
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import re

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class Citation:
    """Structured citation with document and section information"""
    doc_id: str
    section_id: str
    content_snippet: str
    relevance_score: float


@dataclass
class QueryResponse:
    """Response structure with answer and citations"""
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[str]


class AyurvedaRAGSystem:
    """
    RAG system designed specifically for Kerala Ayurveda content.

    Design decisions:
    1. Hybrid chunking strategy: Semantic (headers) + fixed-size with overlap
    2. Embeddings-based retrieval (more semantic than BM25 for medical content)
    3. Retrieves 5 chunks, uses top 3 in prompt to balance context and relevance
    4. Citations include doc_id + section_id for traceable references
    """

    def __init__(self, content_dir: str = ".", persist_dir: str = "./chroma_db"):
        self.content_dir = Path(content_dir)
        self.persist_dir = persist_dir

        # Configure MegaLLM for embeddings and LLM
        megallm_api_key = os.getenv("MEGALLM_API_KEY")

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=megallm_api_key,
            openai_api_base="https://ai.megallm.io/v1"
        )

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=megallm_api_key,
            openai_api_base="https://ai.megallm.io/v1"
        )

        self.vectorstore = None
        self.documents = []

        # Chunking configuration
        self.chunk_sizes = {
            'faq': 400,      # FAQs are Q&A pairs, keep them together
            'product': 500,  # Products have structured sections
            'guide': 800,    # Guides need more context
            'default': 600   # General articles
        }

    def detect_document_type(self, filename: str) -> str:
        """Detect document type for adaptive chunking"""
        if 'faq' in filename.lower():
            return 'faq'
        elif 'product' in filename.lower():
            return 'product'
        elif 'guide' in filename.lower() or 'dosha' in filename.lower():
            return 'guide'
        return 'default'

    def chunk_document(self, content: str, doc_id: str, doc_type: str) -> List[Document]:
        """
        Chunk document with adaptive strategy based on type.

        Strategy:
        - FAQs: Smaller chunks (400 chars) to keep Q&A pairs together
        - Products: Medium chunks (500 chars) for product sections
        - Guides: Larger chunks (800 chars) for conceptual content
        - Overlap: 100 chars to maintain context at boundaries
        """
        chunk_size = self.chunk_sizes.get(doc_type, 600)

        # Use recursive splitter with markdown-aware splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=100,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        # Split and create documents with metadata
        chunks = splitter.split_text(content)
        documents = []

        for i, chunk in enumerate(chunks):
            # Extract section header if present
            section_match = re.search(r'^#+ (.+?)$', chunk, re.MULTILINE)
            section_id = section_match.group(1) if section_match else f"section_{i}"

            doc = Document(
                page_content=chunk,
                metadata={
                    "doc_id": doc_id,
                    "section_id": section_id,
                    "doc_type": doc_type,
                    "chunk_index": i
                }
            )
            documents.append(doc)

        return documents

    def load_and_index_content(self):
        """
        Load all content files and build vector index.

        Handles:
        - Markdown files (.md)
        - CSV product catalog
        """
        print("Loading and indexing content...")

        # Load markdown files
        md_files = list(self.content_dir.glob("*.md"))
        for md_file in md_files:
            doc_id = md_file.stem
            doc_type = self.detect_document_type(doc_id)

            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            chunks = self.chunk_document(content, doc_id, doc_type)
            self.documents.extend(chunks)
            print(f"  Loaded {md_file.name}: {len(chunks)} chunks ({doc_type} type)")

        # Load CSV product catalog
        csv_file = self.content_dir / "products_catalog.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                # Create rich text representation of product
                product_text = f"""
Product: {row['name']} (ID: {row['product_id']})
Category: {row['category']}
Format: {row['format']}
Target Concerns: {row['target_concerns']}
Key Herbs: {row['key_herbs']}
Contraindications: {row['contraindications_short']}
Tags: {row['internal_tags']}
"""
                doc = Document(
                    page_content=product_text,
                    metadata={
                        "doc_id": f"catalog_{row['product_id']}",
                        "section_id": row['name'],
                        "doc_type": "product_catalog",
                        "product_id": row['product_id']
                    }
                )
                self.documents.append(doc)

            print(f"  Loaded products_catalog.csv: {len(df)} products")

        # Build vector store
        print(f"\nBuilding vector index with {len(self.documents)} total chunks...")
        self.vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        print("Index built successfully!")

    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k relevant chunks using semantic similarity.

        Returns chunks with similarity scores for citation ranking.
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call load_and_index_content() first.")

        # Retrieve with scores
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        return results

    def answer_user_query(self, query: str) -> QueryResponse:
        """
        Answer user query with citations.

        Process:
        1. Retrieve 5 most relevant chunks
        2. Select top 3 for context (balance relevance vs. context length)
        3. Build prompt with retrieved context
        4. Generate answer
        5. Attach citations with doc_id and section_id

        Returns:
            QueryResponse with answer, citations, and retrieved chunks
        """
        # Retrieve relevant chunks
        retrieved = self.retrieve_relevant_chunks(query, k=5)

        # Use top 3 for generation (empirically good balance)
        top_chunks = retrieved[:3]

        # Build context from retrieved chunks
        context_parts = []
        for i, (doc, score) in enumerate(top_chunks, 1):
            context_parts.append(
                f"[Source {i}: {doc.metadata['doc_id']} - {doc.metadata['section_id']}]\n"
                f"{doc.page_content}\n"
            )

        context = "\n---\n".join(context_parts)

        # Create prompt following Kerala Ayurveda style guidelines
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert assistant for Kerala Ayurveda. Answer questions using ONLY the provided context.

Style guidelines:
- Warm & reassuring, like a calm practitioner
- Grounded & precise - no vague claims
- Use phrases like "traditionally used to support...", "may help maintain..."
- NEVER claim to diagnose, treat, cure, or prevent diseases
- Always include gentle safety notes when relevant
- Encourage consultation with qualified practitioners

IMPORTANT:
- Only use information from the provided sources
- If the answer isn't in the sources, say so clearly
- Include specific citations in your answer using [Source X] notation
- Be concise but complete"""),
            ("user", """Context from Kerala Ayurveda knowledge base:

{context}

Question: {query}

Please provide a helpful answer based on the context above. Include [Source X] citations in your response.""")
        ])

        # Generate answer
        chain = prompt_template | self.llm
        response = chain.invoke({"context": context, "query": query})
        answer = response.content

        # Build citations
        citations = []
        for doc, score in top_chunks:
            citation = Citation(
                doc_id=doc.metadata['doc_id'],
                section_id=doc.metadata['section_id'],
                content_snippet=doc.page_content[:200] + "...",
                relevance_score=score
            )
            citations.append(citation)

        # Get all retrieved chunks for analysis
        retrieved_chunks = [doc.page_content for doc, _ in retrieved]

        return QueryResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks
        )


def main():
    """Example usage of the RAG system"""
    import dotenv
    dotenv.load_dotenv()

    # Initialize system
    rag = AyurvedaRAGSystem()

    # Load and index content
    rag.load_and_index_content()

    # Example queries
    test_queries = [
        "What are the key benefits of Ashwagandha tablets?",
        "Are there any contraindications for Triphala?",
        "Can Ayurveda help with stress and sleep?"
    ]

    print("\n" + "="*80)
    print("TESTING QUERIES")
    print("="*80)

    for query in test_queries:
        print(f"\n\nQuery: {query}")
        print("-" * 80)

        response = rag.answer_user_query(query)

        print(f"\nAnswer:\n{response.answer}")

        print(f"\n\nCitations:")
        for i, citation in enumerate(response.citations, 1):
            print(f"\n  [{i}] {citation.doc_id} - {citation.section_id}")
            print(f"      Relevance: {citation.relevance_score:.3f}")
            print(f"      Snippet: {citation.content_snippet[:150]}...")


if __name__ == "__main__":
    main()
