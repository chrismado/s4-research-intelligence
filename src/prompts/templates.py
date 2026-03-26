"""
Prompt templates for the S4 Research Intelligence system.

These are carefully engineered for documentary research — not generic Q&A.
The system prompt establishes the assistant as a research analyst, not a
chatbot. The retrieval prompt forces structured output with citations,
contradiction detection, and confidence scoring.
"""

SYSTEM_PROMPT = """You are a documentary research analyst assisting with "S4: The Bob Lazar Story,"
a feature-length documentary investigating Bob Lazar's claims about reverse-engineering
extraterrestrial technology at a facility known as S4 near Area 51.

Your role is to:
1. Answer research questions using ONLY the provided source material
2. Cite every claim with [Source: filename] references
3. Flag contradictions between sources explicitly
4. Distinguish between verified facts, credible claims, and unverified assertions
5. Extract timeline events with dates when available
6. Rate your confidence based on source quality and corroboration

You are NOT an advocate for or against Lazar's claims. You are a neutral research tool
that helps the documentary team navigate complex, often contradictory source material
with rigorous attribution.

When sources disagree, present both sides with citations. Never synthesize claims
from different sources into a single unsourced statement."""


RESEARCH_QUERY_PROMPT = """Based on the following source material, answer the research question.

## Source Material
{context}

## Research Question
{question}

## Instructions
Respond in the following JSON structure:
{{
    "answer": "Your synthesized answer with inline [Source: filename] citations for every claim.",
    "sources_used": ["list of source filenames actually cited in your answer"],
    "contradictions": [
        {{
            "claim_a": "What source A says",
            "source_a": "filename_a",
            "claim_b": "What source B says (contradicts A)",
            "source_b": "filename_b",
            "explanation": "Why these contradict and which has stronger provenance"
        }}
    ],
    "timeline_events": [
        {{
            "date": "YYYY-MM-DD or YYYY-MM or YYYY (as precise as source allows)",
            "description": "What happened",
            "source": "filename",
            "confidence": 0.0-1.0
        }}
    ],
    "confidence": 0.0-1.0,
    "reasoning": "Brief chain-of-thought explaining how you arrived at your answer, which sources you weighted most heavily and why."
}}

CRITICAL RULES:
- If the source material does not contain information to answer the question, say so explicitly. Do NOT hallucinate.
- Every factual claim MUST have a [Source: filename] citation.
- Confidence should reflect: number of corroborating sources, source reliability, and directness of evidence.
- Timeline dates should only be included when explicitly stated in sources, not inferred."""


CONTRADICTION_DETECTION_PROMPT = """Analyze the following excerpts from different sources about the same topic.
Identify any contradictions, inconsistencies, or notable differences in their accounts.

## Excerpts
{excerpts}

## Topic
{topic}

For each contradiction found, provide:
1. The specific conflicting claims
2. Which sources make each claim
3. Analysis of which source is more likely reliable and why
4. Whether the contradiction is factual (dates, events) or interpretive (opinions, analysis)

If no contradictions are found, explicitly state that the sources are consistent."""


TIMELINE_EXTRACTION_PROMPT = """Extract a chronological timeline from the following source material.
Only include events with explicit dates or date ranges mentioned in the text.
Do not infer or estimate dates.

## Source Material
{context}

## Focus Area
{focus}

Return a JSON array of events:
[
    {{
        "date": "As precise as the source states (YYYY, YYYY-MM, or YYYY-MM-DD)",
        "description": "What happened",
        "source": "filename",
        "confidence": 0.0-1.0,
        "category": "personal | professional | government | scientific | media"
    }}
]"""
