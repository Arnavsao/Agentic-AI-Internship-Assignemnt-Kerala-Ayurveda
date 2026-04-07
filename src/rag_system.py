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
import pypdf
import chromadb

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from src.key_manager import GeminiKeyManager


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

    def __init__(self, content_dir: str = "data", persist_dir: str = "./chroma_db"):
        self.content_dir = Path(content_dir)
        self.persist_dir = persist_dir

        # Key manager handles multi-key rotation on quota exhaustion
        self.key_manager = GeminiKeyManager()

        # Local HuggingFace embeddings (no API needed, fast, reliable)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
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
        elif 'pdf' in filename.lower():
            return 'guide'  # PDFs are long-form content — use larger chunks
        return 'default'

    def load_pdf_document(self, pdf_path: Path) -> str:
        """
        Extract full text from a PDF using pypdf.
        Joins all pages with double newlines for clean paragraph separation.
        """
        reader = pypdf.PdfReader(str(pdf_path))
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_text.append(f"[Page {i + 1}]\n{text.strip()}")
        return "\n\n".join(pages_text)

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

        # recursive splitter with markdown-aware splitting
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

        On subsequent startups, reuses the persisted ChromaDB index to avoid
        re-embedding all documents (which was causing the Streamlit endless reload).
        Only rebuilds from scratch if the collection is empty or missing.

        Handles:
        - Markdown files (.md)
        - PDF documents (.pdf)
        - CSV product catalog
        """
        persist_path = str(Path(self.persist_dir).resolve())
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=persist_path)

        # ── Fast path: reuse existing persisted index ──────────────────────
        try:
            existing = chroma_client.get_collection("ayurveda_rag")
            count = existing.count()
            if count > 0:
                print(f"Reusing persisted ChromaDB index ({count} chunks already indexed).")
                self.vectorstore = Chroma(
                    client=chroma_client,
                    collection_name="ayurveda_rag",
                    embedding_function=self.embeddings,
                )
                return
        except Exception:
            pass  # Collection doesn't exist yet — fall through to full build

        # ── Slow path: build from scratch (first run only) ─────────────────
        print("Building vector index for the first time — this takes ~30s...")

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

        # Load PDF files
        pdf_files = list(self.content_dir.glob("*.pdf"))
        for pdf_file in pdf_files:
            doc_id = pdf_file.stem
            doc_type = 'guide'  # PDFs are long-form — use 800-char chunks
            try:
                content = self.load_pdf_document(pdf_file)
                if content.strip():
                    chunks = self.chunk_document(content, doc_id, doc_type)
                    self.documents.extend(chunks)
                    print(f"  Loaded {pdf_file.name}: {len(chunks)} chunks (pdf/guide type)")
                else:
                    print(f"  WARNING: {pdf_file.name} produced no extractable text (may be scanned image PDF)")
            except Exception as e:
                print(f"  ERROR loading {pdf_file.name}: {e}")

        # Load CSV product catalog
        csv_file = self.content_dir / "products_catalog.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
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

        print(f"\nEmbedding and indexing {len(self.documents)} total chunks...")
        self.vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            client=chroma_client,
            collection_name="ayurveda_rag"
        )
        print("Index built and persisted successfully!")

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

        # Generate answer with automatic key rotation on quota exhaustion
        def create_llm(api_key):
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,
                google_api_key=api_key
            )

        def invoke(llm):
            chain = prompt_template | llm
            return chain.invoke({"context": context, "query": query})

        response = self.key_manager.invoke_with_rotation(create_llm, invoke)
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
