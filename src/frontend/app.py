"""
S4 Research Intelligence — Streamlit Frontend

A demo-ready interface for querying the documentary research corpus.
Designed for live interview demos: "let me show you how this works."
"""

import streamlit as st
from loguru import logger

from config.settings import settings
from src.ingestion.vectorstore import VectorStore
from src.models.documents import SourceType
from src.models.queries import ResearchQuery
from src.retrieval.pipeline import ResearchPipeline


# --- Page config ---
st.set_page_config(
    page_title="S4 Research Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Session state initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = []


@st.cache_resource
def get_vector_store():
    """Initialize vector store (cached across reruns)."""
    return VectorStore()


@st.cache_resource
def get_pipeline():
    """Initialize RAG pipeline (cached across reruns)."""
    store = get_vector_store()
    return ResearchPipeline(vector_store=store)


def render_sidebar():
    """Render the sidebar with filters and corpus info."""
    st.sidebar.title("S4 Research Intelligence")
    st.sidebar.caption("RAG-powered documentary research assistant")
    st.sidebar.divider()

    # Corpus stats
    store = get_vector_store()
    st.sidebar.metric("Chunks in Corpus", store.count)
    st.sidebar.metric("Embedding Model", settings.embedding_model.split("/")[-1])
    st.sidebar.metric("LLM", settings.llm_model.split(":")[0])

    st.sidebar.divider()

    # Source type filter
    st.sidebar.subheader("Filters")
    source_filter = st.sidebar.multiselect(
        "Source Types",
        options=[st.value for st in SourceType],
        format_func=lambda x: x.replace("_", " ").title(),
        default=None,
        help="Filter retrieval to specific source types",
    )

    top_k = st.sidebar.slider("Sources to retrieve", 3, 15, 5)
    include_contradictions = st.sidebar.checkbox("Detect contradictions", value=True)

    st.sidebar.divider()

    # Conversation controls
    if st.sidebar.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_context = []
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption(
        "Built for **S4: The Bob Lazar Story**  \n"
        "Source-weighted retrieval | Contradiction detection | Timeline extraction"
    )

    return source_filter, top_k, include_contradictions


def render_sources(sources):
    """Render source cards with relevance/reliability scores."""
    if not sources:
        return

    st.subheader("Sources")
    cols = st.columns(min(len(sources), 3))
    for i, src in enumerate(sources):
        with cols[i % 3]:
            source_type_display = src.source_type.value.replace("_", " ").title()
            reliability_color = "green" if src.reliability_score >= 0.8 else "orange" if src.reliability_score >= 0.6 else "red"

            st.markdown(f"""
**{src.title}**
- **Type:** {source_type_display}
- **Author:** {src.author or 'Unknown'}
- **Date:** {src.date_created or 'Unknown'}
- **Relevance:** {src.relevance_score:.1%}
- **Reliability:** :{reliability_color}[{src.reliability_score:.0%}]
- **Combined:** **{src.combined_score:.1%}**
            """)
            with st.expander("Excerpt"):
                st.text(src.excerpt)


def render_contradictions(contradictions):
    """Render contradiction alerts with side-by-side comparison."""
    if not contradictions:
        return

    st.subheader("Contradictions Detected")
    for c in contradictions:
        with st.container(border=True):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**{c.source_a}**")
                st.info(c.claim_a)
            with col_b:
                st.markdown(f"**{c.source_b}**")
                st.warning(c.claim_b)
            st.caption(f"Analysis: {c.explanation}")


def render_timeline(timeline):
    """Render timeline events."""
    if not timeline:
        return

    st.subheader("Timeline")
    for event in sorted(timeline, key=lambda e: e.date or ""):
        date_str = event.date or "Unknown date"
        confidence_pct = f"{event.confidence:.0%}"
        st.markdown(f"**{date_str}** - {event.description} *(from {event.source}, confidence: {confidence_pct})*")


def build_conversation_prompt(question: str) -> str:
    """Build a question with conversation context for multi-turn support."""
    if not st.session_state.conversation_context:
        return question

    # Include last 3 turns of context
    recent = st.session_state.conversation_context[-3:]
    context_summary = "\n".join(
        f"Q: {turn['question']}\nA: {turn['answer'][:200]}..."
        for turn in recent
    )
    return (
        f"Previous conversation context:\n{context_summary}\n\n"
        f"Follow-up question: {question}"
    )


def main():
    source_filter, top_k, include_contradictions = render_sidebar()

    # Chat interface
    st.title("S4 Research Intelligence")
    st.caption("Query the documentary research corpus with source-weighted retrieval")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "response" in msg:
                resp = msg["response"]
                render_sources(resp.sources)
                render_contradictions(resp.contradictions)
                render_timeline(resp.timeline)

    # Chat input
    if question := st.chat_input("Ask a research question about S4, Bob Lazar, Element 115..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Build query with conversation context
        enriched_question = build_conversation_prompt(question)

        # Build research query
        source_types = [SourceType(st_val) for st_val in source_filter] if source_filter else None
        rq = ResearchQuery(
            question=enriched_question,
            source_types=source_types,
            top_k=top_k,
            include_contradictions=include_contradictions,
        )

        # Execute query
        with st.chat_message("assistant"):
            with st.spinner("Searching corpus and generating response..."):
                try:
                    pipeline = get_pipeline()
                    response = pipeline.query(rq)

                    # Display answer
                    st.markdown(response.answer)

                    # Confidence indicator
                    conf_color = "green" if response.confidence > 0.7 else "orange" if response.confidence > 0.4 else "red"
                    st.markdown(f"**Confidence:** :{conf_color}[{response.confidence:.0%}]")

                    if response.reasoning:
                        with st.expander("Reasoning trace"):
                            st.text(response.reasoning)

                    # Render detailed panels
                    render_sources(response.sources)
                    render_contradictions(response.contradictions)
                    render_timeline(response.timeline)

                    # Store in session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "response": response,
                    })

                    # Update conversation context for multi-turn
                    st.session_state.conversation_context.append({
                        "question": question,
                        "answer": response.answer,
                    })

                except Exception as e:
                    error_msg = f"Error querying pipeline: {e}"
                    st.error(error_msg)
                    logger.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {e}",
                    })


if __name__ == "__main__":
    main()
