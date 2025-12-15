"""
Local Demo Script for Kerala Ayurveda RAG + Agent System
Run this to test the system locally
"""

import os
from dotenv import load_dotenv
from src.rag_system import AyurvedaRAGSystem
from src.agent_workflow import ArticleWorkflowOrchestrator

# Load environment variables
load_dotenv()

def demo_rag_system():
    """Demo the RAG Q&A system"""
    print("=" * 80)
    print("KERALA AYURVEDA RAG SYSTEM - Q&A Demo")
    print("=" * 80)

    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n❌ ERROR: GOOGLE_API_KEY not found in .env file")
        print("Please add your Gemini API key to .env file:")
        print("GOOGLE_API_KEY=your_api_key_here")
        return

    print("\n📚 Loading Kerala Ayurveda knowledge base...")
    rag = AyurvedaRAGSystem()
    rag.load_and_index_content()
    print("✓ Knowledge base loaded successfully!")

    # Example questions
    questions = [
        "What are the benefits of Ashwagandha?",
        "Are there any contraindications for Triphala?",
        "What is the Stress Support Program?"
    ]

    print("\n" + "=" * 80)
    print("TESTING RAG SYSTEM WITH SAMPLE QUESTIONS")
    print("=" * 80)

    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 80}")
        print(f"Question {i}: {question}")
        print(f"{'─' * 80}")

        response = rag.answer_user_query(question)

        print(f"\n📝 Answer:")
        print(response.answer)

        print(f"\n📚 Sources ({len(response.citations)} citations):")
        for j, citation in enumerate(response.citations, 1):
            print(f"  {j}. {citation.doc_id} (Section: {citation.section_id})")
            print(f"     Relevance: {citation.relevance_score:.1%}")
            print(f"     Snippet: {citation.content_snippet[:100]}...")

    # Interactive mode
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("Ask your own questions (type 'quit' to exit):")

    while True:
        question = input("\n🌿 Your question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("\n✓ Goodbye!")
            break

        if not question:
            continue

        response = rag.answer_user_query(question)
        print(f"\n📝 Answer:")
        print(response.answer)
        print(f"\n📚 Sources: {len(response.citations)} citations")


def demo_agent_workflow():
    """Demo the multi-agent blog generation system"""
    print("\n" + "=" * 80)
    print("MULTI-AGENT BLOG GENERATION SYSTEM - Demo")
    print("=" * 80)

    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n❌ ERROR: GOOGLE_API_KEY not found in .env file")
        return

    print("\n🤖 Initializing agent workflow...")
    workflow = ArticleWorkflowOrchestrator()
    print("✓ Agents initialized!")

    # Example topic
    topic = "Benefits of Ashwagandha for Stress Management"
    target_audience = "Health-conscious professionals dealing with workplace stress"
    tone = "informative yet approachable"

    print(f"\n📋 Generating blog article...")
    print(f"   Topic: {topic}")
    print(f"   Audience: {target_audience}")
    print(f"   Tone: {tone}")

    result = workflow.run(
        topic=topic,
        target_audience=target_audience,
        desired_tone=tone
    )

    print("\n" + "=" * 80)
    print("GENERATED ARTICLE")
    print("=" * 80)
    print(result.final_article)

    print("\n" + "=" * 80)
    print("WORKFLOW METADATA")
    print("=" * 80)
    print(f"Total processing time: {result.total_time:.2f}s")
    print(f"Outline generation: {result.outline_time:.2f}s")
    print(f"Writing: {result.writing_time:.2f}s")
    print(f"Fact checking: {result.fact_checking_time:.2f}s")
    print(f"Tone editing: {result.tone_editing_time:.2f}s")


def main():
    """Main demo function"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║           KERALA AYURVEDA RAG + AGENTIC AI SYSTEM                           ║
║           Internship Assignment Demo                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    print("Choose a demo:")
    print("1. RAG Q&A System (with interactive mode)")
    print("2. Multi-Agent Blog Generation")
    print("3. Both")
    print("4. Exit")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == "1":
        demo_rag_system()
    elif choice == "2":
        demo_agent_workflow()
    elif choice == "3":
        demo_rag_system()
        print("\n" * 2)
        demo_agent_workflow()
    elif choice == "4":
        print("\n✓ Goodbye!")
    else:
        print("\n❌ Invalid choice. Please run again.")


if __name__ == "__main__":
    main()
