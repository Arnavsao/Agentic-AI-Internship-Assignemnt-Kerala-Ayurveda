#!/usr/bin/env python3
"""
Test script to verify the project works for interview demo
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dotenv import load_dotenv
load_dotenv()

def test_indexing():
    """Test that indexing works"""
    print("="*80)
    print("TEST 1: Content Indexing")
    print("="*80)
    
    try:
        from rag_system import AyurvedaRAGSystem
        
        rag = AyurvedaRAGSystem()
        rag.load_and_index_content()
        
        print(f"✅ SUCCESS: Indexed {len(rag.documents)} documents")
        print(f"✅ Vector store created: {rag.vectorstore is not None}")
        return rag
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_query(rag):
    """Test a query (requires API key)"""
    print("\n" + "="*80)
    print("TEST 2: Query Processing")
    print("="*80)
    
    if rag is None:
        print("❌ Skipping - indexing failed")
        return
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        print("⚠️  WARNING: GOOGLE_API_KEY not set or is placeholder")
        print("   The system can index content but cannot answer queries without API key")
        print("   For interview demo, you can:")
        print("   1. Show that indexing works (✅ above)")
        print("   2. Explain the query flow")
        print("   3. Use a valid API key for live demo")
        return
    
    try:
        query = "What are the benefits of Ashwagandha?"
        print(f"Query: {query}")
        print("Processing...")
        
        response = rag.answer_user_query(query)
        
        print(f"\n✅ Answer generated:")
        print(f"{response.answer[:200]}...")
        print(f"\n✅ Citations: {len(response.citations)}")
        for i, cit in enumerate(response.citations[:2], 1):
            print(f"   [{i}] {cit.doc_id} - {cit.section_id} (score: {cit.relevance_score:.2f})")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("\nThis is likely due to:")
        print("  1. Invalid API key")
        print("  2. Network issues")
        print("  3. API rate limits")
        import traceback
        traceback.print_exc()

def main():
    print("\n🧪 TESTING KERALA AYURVEDA RAG SYSTEM")
    print("="*80)
    
    # Test indexing
    rag = test_indexing()
    
    # Test query
    test_query(rag)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if rag:
        print("✅ Content indexing: WORKING")
        print("✅ Vector store: CREATED")
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key and api_key != "your_google_api_key_here":
            print("✅ API Key: CONFIGURED")
        else:
            print("⚠️  API Key: NEEDS VALID KEY FOR QUERIES")
        print("\n💡 For interview:")
        print("   - Indexing works perfectly (can show this)")
        print("   - Explain the RAG pipeline architecture")
        print("   - Show code structure and design decisions")
    else:
        print("❌ Indexing failed - check errors above")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
