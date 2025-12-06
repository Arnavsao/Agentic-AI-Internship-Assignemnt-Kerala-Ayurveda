"""
Demo script showing example queries and expected outputs
Used for Part A requirement: "Run or reason on 2-3 example queries"
"""

from typing import Dict, List

# Example outputs based on the Kerala Ayurveda corpus
# These show what the system would produce


class DemoExamples:
    """Documented example queries with expected outputs and analysis"""

    @staticmethod
    def example_1_ashwagandha_benefits() -> Dict:
        """
        Query: What are the key benefits of Ashwagandha tablets?

        Expected retrieval:
        - product_ashwagandha_tablets_internal (primary source)
        - ayurveda_foundations (supporting context on stress support)
        - content_style_and_tone_guide (for appropriate phrasing)

        Retrieved sections:
        1. "Traditional Positioning" from Ashwagandha product dossier
        2. "Key Messages for Content" from same
        3. "Safety & Precautions" section
        """
        return {
            "query": "What are the key benefits of Ashwagandha tablets?",

            "expected_retrieved_docs": [
                {
                    "doc_id": "product_ashwagandha_tablets_internal",
                    "section_id": "Traditional Positioning",
                    "relevance": 0.95,
                    "content_snippet": "In Ayurveda, Ashwagandha is traditionally used to: Support the body's ability to adapt to stress, Promote calmness and emotional balance..."
                },
                {
                    "doc_id": "product_ashwagandha_tablets_internal",
                    "section_id": "Key Messages for Content",
                    "relevance": 0.89,
                    "content_snippet": "Stress resilience, not sedation. Emphasise adaptation, steadiness, and recovery..."
                },
                {
                    "doc_id": "ayurveda_foundations",
                    "section_id": "How we position Ayurveda at Kerala Ayurveda",
                    "relevance": 0.76,
                    "content_snippet": "We focus on supporting natural processes... traditionally used to support..."
                }
            ],

            "expected_answer": """Ashwagandha Stress Balance Tablets are traditionally used in Ayurveda to support the body's ability to adapt to stress and promote emotional balance [Source: product_ashwagandha_tablets_internal - Traditional Positioning].

Key benefits include:
- Supporting stress resilience and the body's natural stress response
- Promoting calmness and emotional steadiness
- Supporting restful sleep, especially for those feeling "wired and tired"
- Helping maintain strength and stamina during demanding periods

The product is positioned as daily support for stress resilience, not as sedation [Source: product_ashwagandha_tablets_internal - Key Messages]. It works best alongside healthy routines like reduced screens before bed, calming evening rituals, and adequate rest.

**Important safety note:** People with thyroid disorders, autoimmune conditions, or those on long-term medication should consult their healthcare provider before use. This product is not a substitute for medical treatment or mental health care [Source: product_ashwagandha_tablets_internal - Safety & Precautions].""",

            "potential_failure_modes": [
                {
                    "failure": "Claiming to cure anxiety or insomnia",
                    "why": "Product dossier explicitly avoids disease claims, but LLM might overgeneralize from 'stress' to 'anxiety disorder'",
                    "mitigation": "Strong prompt instructions + fact-checking agent to flag disease claims"
                },
                {
                    "failure": "Missing safety warnings for thyroid conditions",
                    "why": "If chunking splits the contraindications section poorly, safety info might not be retrieved",
                    "mitigation": "Ensure safety sections are kept together in chunking; fact-checker verifies safety language present"
                },
                {
                    "failure": "Adding dosing instructions not in corpus",
                    "why": "Common knowledge about Ashwagandha might leak from training data",
                    "mitigation": "Explicit instruction to not provide dosing; fact-checker flags specific medical instructions"
                }
            ]
        }

    @staticmethod
    def example_2_triphala_contraindications() -> Dict:
        """
        Query: Are there any contraindications or precautions for Triphala?

        Expected retrieval:
        - product_triphala_capsules_internal (primary source)
        - products_catalog.csv (supporting product info)

        Retrieved sections:
        1. "Safety & Precautions" from Triphala product dossier
        2. Catalog entry for Triphala
        """
        return {
            "query": "Are there any contraindications or precautions for Triphala?",

            "expected_retrieved_docs": [
                {
                    "doc_id": "product_triphala_capsules_internal",
                    "section_id": "Safety & Precautions (Content Version)",
                    "relevance": 0.97,
                    "content_snippet": "Not intended to diagnose, treat, cure, or prevent any disease. Individuals with: Chronic digestive disease, Recent abdominal surgery, Pregnancy or breastfeeding..."
                },
                {
                    "doc_id": "catalog_KA-P001",
                    "section_id": "Triphala Capsules",
                    "relevance": 0.84,
                    "content_snippet": "Contraindications: Consult doctor in chronic digestive disease, pregnancy, or post-surgery"
                },
                {
                    "doc_id": "ayurveda_foundations",
                    "section_id": "Kerala Ayurveda Content Boundaries",
                    "relevance": 0.72,
                    "content_snippet": "Always add a gentle safety note for herbs and therapies..."
                }
            ],

            "expected_answer": """Yes, there are important precautions to be aware of before using Triphala Capsules [Source: product_triphala_capsules_internal - Safety & Precautions].

**Individuals who should consult a healthcare provider before use:**
- Those with chronic digestive disease
- People who have had recent abdominal surgery
- Those who are pregnant or breastfeeding

Triphala is a traditional Ayurvedic formulation for digestive support, and experiences may vary from person to person [Source: product_triphala_capsules_internal]. It is not intended to diagnose, treat, cure, or prevent any disease.

If you have a medical condition or take prescription medication, please consult your healthcare provider before adding Triphala to your routine [Source: catalog_KA-P001]. This ensures the supplement is appropriate for your individual circumstances.""",

            "potential_failure_modes": [
                {
                    "failure": "Generic 'consult your doctor' without specific contraindications",
                    "why": "LLM might play it safe and be too vague instead of citing specific warnings from corpus",
                    "mitigation": "Prompt instructs to be specific; retrieval should surface detailed precautions section"
                },
                {
                    "failure": "Omitting pregnancy/breastfeeding warning",
                    "why": "If only the CSV is retrieved (shorter contraindications), detailed warnings might be missed",
                    "mitigation": "Retrieve more chunks (k=5) to increase chance of getting detailed product dossier"
                },
                {
                    "failure": "Adding contraindications not in corpus (e.g., kidney disease)",
                    "why": "General Ayurveda knowledge from training data",
                    "mitigation": "Strict grounding requirement; fact-checker verifies each claim has source"
                }
            ]
        }

    @staticmethod
    def example_3_ayurveda_stress_sleep() -> Dict:
        """
        Query: Can Ayurveda help with stress and sleep?

        Expected retrieval:
        - faq_general_ayurveda_patients (Q3 directly addresses this)
        - treatment_stress_support_program (clinic program details)
        - product_ashwagandha_tablets_internal (specific herb for stress/sleep)

        Retrieved sections:
        1. FAQ Q3: "Can Ayurveda help with stress and sleep?"
        2. Stress Support Program overview
        3. Ashwagandha product positioning
        """
        return {
            "query": "Can Ayurveda help with stress and sleep?",

            "expected_retrieved_docs": [
                {
                    "doc_id": "faq_general_ayurveda_patients",
                    "section_id": "3. Can Ayurveda help with stress and sleep?",
                    "relevance": 0.98,
                    "content_snippet": "Ayurveda approaches stress and sleep through: Daily routines, Food choices, Herbs like Ashwagandha, Therapies such as oil massages..."
                },
                {
                    "doc_id": "treatment_stress_support_program",
                    "section_id": "Core Components",
                    "relevance": 0.87,
                    "content_snippet": "Supporting people who experience ongoing stress, mental fatigue, and difficulty unwinding..."
                },
                {
                    "doc_id": "product_ashwagandha_tablets_internal",
                    "section_id": "Traditional Positioning",
                    "relevance": 0.85,
                    "content_snippet": "Support the body's ability to adapt to stress, Promote calmness, Help maintain restful sleep..."
                }
            ],

            "expected_answer": """Yes, Ayurveda offers supportive approaches for stress and sleep, though it's important to understand it as complementary support rather than a replacement for medical care [Source: faq_general_ayurveda_patients - Q3].

**How Ayurveda approaches stress and sleep:**

1. **Daily routines** - Consistent sleep-wake timing and grounding practices
2. **Food choices** - Eating patterns that feel calming and supportive
3. **Herbal support** - Herbs like Ashwagandha, traditionally used to support the body's stress response [Source: product_ashwagandha_tablets_internal]
4. **Therapies** - Oil massages (Abhyanga), head oil applications (like Shirodhara), and calming practices [Source: treatment_stress_support_program]

At Kerala Ayurveda clinics, the Stress Support Program combines traditional therapies with personalized routine suggestions to help people unwind and build resilience to everyday stress [Source: treatment_stress_support_program - Core Components].

**Important note:** Ayurvedic support can complement, but not replace, professional mental health care when needed [Source: faq_general_ayurveda_patients]. If you experience severe or persistent mood changes or significant functional impairment, please seek support from a qualified mental health professional.""",

            "potential_failure_modes": [
                {
                    "failure": "Overpromising results or timelines",
                    "why": "User asking 'can it help' might lead LLM to be overly optimistic to be helpful",
                    "mitigation": "Style guide emphasizes 'may help', 'support' language; tone checker catches overclaims"
                },
                {
                    "failure": "Not emphasizing 'complementary' nature strongly enough",
                    "why": "Multiple sources mention both benefits AND limitations, but LLM might emphasize benefits more",
                    "mitigation": "System prompt includes strong disclaimer requirements; fact-checker verifies safety language"
                },
                {
                    "failure": "Missing the 'seek professional help' disclaimer",
                    "why": "Might get lost if only FAQ chunk is used without treatment program safety note",
                    "mitigation": "Retrieve k=5 chunks to get multiple sources with safety language; guardrail requires disclaimer for mental health topics"
                }
            ]
        }

    @staticmethod
    def get_all_examples() -> List[Dict]:
        """Get all demo examples"""
        return [
            DemoExamples.example_1_ashwagandha_benefits(),
            DemoExamples.example_2_triphala_contraindications(),
            DemoExamples.example_3_ayurveda_stress_sleep()
        ]


def print_example_analysis():
    """Print formatted analysis of example queries"""
    examples = DemoExamples.get_all_examples()

    print("="*80)
    print("PART A: EXAMPLE QUERY ANALYSIS")
    print("="*80)

    for i, example in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i}")
        print(f"{'='*80}\n")

        print(f"Query: {example['query']}\n")

        print("EXPECTED RETRIEVED DOCUMENTS:")
        print("-" * 80)
        for doc in example['expected_retrieved_docs']:
            print(f"\n  [{doc['relevance']:.2f}] {doc['doc_id']}")
            print(f"  Section: {doc['section_id']}")
            print(f"  Snippet: {doc['content_snippet'][:100]}...")

        print("\n\nEXPECTED ANSWER:")
        print("-" * 80)
        print(example['expected_answer'])

        print("\n\nPOTENTIAL FAILURE MODES:")
        print("-" * 80)
        for j, failure in enumerate(example['potential_failure_modes'], 1):
            print(f"\n  {j}. {failure['failure']}")
            print(f"     Why: {failure['why']}")
            print(f"     Mitigation: {failure['mitigation']}")

        print("\n")


if __name__ == "__main__":
    print_example_analysis()
