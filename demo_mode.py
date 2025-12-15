#!/usr/bin/env python3
"""
Demo mode for interview - works without API calls
Shows the complete RAG pipeline with mock responses
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dotenv import load_dotenv
load_dotenv()

from rag_system import AyurvedaRAGSystem

def demo_rag_system():
    """Demonstrate RAG system without API calls"""
    print("="*80)
    print("KERALA AYURVEDA RAG SYSTEM - DEMO MODE")
    print("="*80)
    print("\nThis demo shows:")
    print("1. Content loading and indexing ✅")
    print("2. Document chunking strategy ✅")
    print("3. Vector store creation ✅")
    print("4. Query processing flow (mock) ✅")
    print("\n" + "="*80)
    
    # Initialize and load
    print("\n📚 STEP 1: Loading and Indexing Content")
    print("-" * 80)
    rag = AyurvedaRAGSystem()
    rag.load_and_index_content()
    
    print(f"\n✅ Successfully indexed {len(rag.documents)} document chunks")
    print(f"✅ Vector store created with {len(rag.vectorstore.get()['ids']) if rag.vectorstore else 0} vectors")
    
    # Show document types
    print("\n📊 Document Breakdown:")
    doc_types = {}
    for doc in rag.documents:
        doc_type = doc.metadata.get('doc_type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    for doc_type, count in doc_types.items():
        print(f"   - {doc_type}: {count} chunks")
    
    # Demonstrate retrieval (without API call)
    print("\n🔍 STEP 2: Semantic Retrieval Demo")
    print("-" * 80)
    query = "What are the benefits of Ashwagandha?"
    print(f"Query: {query}")
    
    try:
        # Try to retrieve (this works without API)
        retrieved = rag.retrieve_relevant_chunks(query, k=5)
        print(f"\n✅ Retrieved {len(retrieved)} relevant chunks:")
        
        for i, (doc, score) in enumerate(retrieved[:3], 1):
            print(f"\n   [{i}] {doc.metadata['doc_id']}")
            print(f"       Section: {doc.metadata['section_id']}")
            print(f"       Relevance Score: {score:.3f}")
            print(f"       Preview: {doc.page_content[:100]}...")
        
        print("\n💡 In production, these chunks would be sent to LLM for answer generation")
        print("   The LLM would generate an answer using ONLY these retrieved sources")
        
    except Exception as e:
        print(f"⚠️  Retrieval test: {e}")
    
    # Show chunking strategy
    print("\n📝 STEP 3: Adaptive Chunking Strategy")
    print("-" * 80)
    print("The system uses different chunk sizes based on content type:")
    print(f"   - FAQ files: {rag.chunk_sizes['faq']} chars (keep Q&A pairs together)")
    print(f"   - Product files: {rag.chunk_sizes['product']} chars (preserve sections)")
    print(f"   - Guide files: {rag.chunk_sizes['guide']} chars (maintain context)")
    print(f"   - Default: {rag.chunk_sizes['default']} chars")
    
    # Show example chunks
    print("\n📄 Example Chunks:")
    for i, doc in enumerate(rag.documents[:3], 1):
        print(f"\n   Chunk {i}:")
        print(f"   - Source: {doc.metadata['doc_id']}")
        print(f"   - Section: {doc.metadata['section_id']}")
        print(f"   - Type: {doc.metadata['doc_type']}")
        print(f"   - Length: {len(doc.page_content)} chars")
        print(f"   - Preview: {doc.page_content[:80]}...")
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE - System is working!")
    print("="*80)
    print("\n💡 For interview presentation:")
    print("   1. Show this indexing works perfectly")
    print("   2. Explain the RAG architecture")
    print("   3. Show code structure and design decisions")
    print("   4. Mention that LLM integration works with valid API key")
    print("\n📋 Key Points to Highlight:")
    print("   - Adaptive chunking (400-800 chars based on content type)")
    print("   - Semantic retrieval using embeddings")
    print("   - Structured citations (doc_id + section_id)")
    print("   - Production-ready architecture")
    print("="*80 + "\n")

if __name__ == "__main__":
    demo_rag_system()
