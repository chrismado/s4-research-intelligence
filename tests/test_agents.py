"""Tests for the multi-agent orchestration layer."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.state import (
    AgentCall,
    CorpusResult,
    FactCheckResult,
    ResearchState,
    TimelineResult,
    TraceEntry,
)

# --- State schema tests ---


class TestResearchState:
    """Test the ResearchState TypedDict and related schemas."""

    def test_empty_state(self):
        state: ResearchState = {
            "query": "test",
            "query_type": "",
            "research_plan": [],
            "corpus_results": [],
            "cross_ref_results": {},
            "timeline_results": [],
            "fact_check_results": [],
            "synthesis": "",
            "confidence": 0.0,
            "sources_cited": [],
            "retries": 0,
            "max_retries": 2,
            "trace": [],
            "trace_id": "test_trace",
        }
        assert state["query"] == "test"
        assert state["retries"] == 0

    def test_agent_call_schema(self):
        call: AgentCall = {
            "agent": "corpus_search",
            "reason": "Foundation retrieval",
            "depends_on": [],
            "priority": 0,
        }
        assert call["agent"] == "corpus_search"
        assert call["priority"] == 0

    def test_corpus_result_schema(self):
        result: CorpusResult = {
            "content": "Test content about S4",
            "source_file": "test.txt",
            "source_type": "interview_transcript",
            "relevance_score": 0.85,
            "reliability_score": 0.70,
            "combined_score": 0.80,
            "excerpt": "Test content...",
        }
        assert result["relevance_score"] == 0.85

    def test_fact_check_result_schema(self):
        result: FactCheckResult = {
            "claim": "Lazar worked at Los Alamos",
            "verdict": "DISPUTED",
            "confidence": 0.65,
            "supporting_sources": [],
            "contradicting_sources": [],
            "reasoning": "Evidence is mixed.",
        }
        assert result["verdict"] == "DISPUTED"

    def test_timeline_result_schema(self):
        result: TimelineResult = {
            "date": "1988",
            "description": "Lazar claims to have started at S4",
            "source": "interview.txt",
            "confidence": 0.7,
            "category": "professional",
        }
        assert result["date"] == "1988"

    def test_trace_entry_schema(self):
        entry: TraceEntry = {
            "node": "corpus_search",
            "action": "search",
            "timestamp": "2024-01-01T00:00:00Z",
            "duration_ms": 150.5,
            "inputs": {"query": "test"},
            "outputs": {"result_count": 5},
            "error": None,
        }
        assert entry["duration_ms"] == 150.5
        assert entry["error"] is None


# --- Agent node tests (with mocked LLM and vector store) ---


class TestCorpusSearchNode:
    """Test the corpus search agent node."""

    @patch("src.agents.corpus_search.llm_call")
    @patch("src.agents.corpus_search.corpus_search_tool")
    @patch("src.agents.corpus_search.assemble_context_tool")
    def test_returns_results(self, mock_context, mock_search, mock_llm):
        from src.agents.corpus_search import corpus_search_node

        mock_search.return_value = [
            {
                "content": "Lazar worked at S4",
                "source_file": "interview.txt",
                "source_type": "interview_transcript",
                "relevance_score": 0.9,
                "reliability_score": 0.7,
                "combined_score": 0.83,
                "excerpt": "Lazar worked at S4",
            }
        ]
        mock_context.return_value = "--- Source 1 ---\nLazar worked at S4"
        mock_llm.return_value = {
            "findings": "Lazar claimed to work at S4.",
            "key_claims": [
                {"claim": "Worked at S4", "source": "interview.txt", "reliability": 0.7},
            ],
            "gaps": [],
            "confidence": 0.8,
        }

        state: ResearchState = {
            "query": "Where did Lazar work?",
            "trace": [],
        }

        result = corpus_search_node(state)

        assert len(result["corpus_results"]) == 1
        assert result["corpus_results"][0]["source_file"] == "interview.txt"
        assert len(result["trace"]) == 1

    @patch("src.agents.corpus_search.corpus_search_tool")
    def test_handles_no_results(self, mock_search):
        from src.agents.corpus_search import corpus_search_node

        mock_search.return_value = []

        state: ResearchState = {"query": "nonexistent topic", "trace": []}
        result = corpus_search_node(state)

        assert result["corpus_results"] == []
        assert result["trace"][0]["error"] == "No results found"


class TestCrossReferenceNode:
    """Test the cross-reference agent node."""

    @patch("src.agents.cross_reference.llm_call")
    @patch("src.agents.cross_reference.filtered_search_tool")
    @patch("src.agents.cross_reference.assemble_context_tool")
    def test_produces_report(self, mock_context, mock_search, mock_llm):
        from src.agents.cross_reference import cross_reference_node

        mock_search.return_value = [
            {
                "content": "Evidence text",
                "source_file": "gov_doc.pdf",
                "source_type": "government_document",
                "relevance_score": 0.8,
                "reliability_score": 0.95,
                "combined_score": 0.85,
            }
        ]
        mock_context.return_value = "Context text"
        mock_llm.return_value = {
            "claim": "Lazar worked at S4",
            "corroborating": [{"source": "interview.txt", "evidence": "He said so"}],
            "contradicting": [{"source": "gov_doc.pdf", "evidence": "No record"}],
            "unresolved": [],
            "summary": "Mixed evidence.",
        }

        state: ResearchState = {
            "query": "Did Lazar work at S4?",
            "corpus_results": [{"content": "test", "source_file": "test.txt"}],
            "trace": [],
        }

        result = cross_reference_node(state)

        assert result["cross_ref_results"]["claim"] == "Lazar worked at S4"
        assert len(result["cross_ref_results"]["corroborating"]) == 1
        assert len(result["cross_ref_results"]["contradicting"]) == 1

    def test_skips_without_corpus_results(self):
        from src.agents.cross_reference import cross_reference_node

        state: ResearchState = {
            "query": "test",
            "corpus_results": [],
            "trace": [],
        }

        result = cross_reference_node(state)
        assert "No corpus results" in result["cross_ref_results"]["summary"]


class TestTimelineNode:
    """Test the timeline agent node."""

    @patch("src.agents.timeline_agent.llm_call")
    @patch("src.agents.timeline_agent.assemble_context_tool")
    def test_extracts_events(self, mock_context, mock_llm):
        from src.agents.timeline_agent import timeline_node

        mock_context.return_value = "Timeline context"
        mock_llm.return_value = {
            "events": [
                {
                    "date": "1989",
                    "description": "First public disclosure",
                    "source": "interview.txt",
                    "confidence": 0.9,
                    "category": "media",
                },
                {
                    "date": "1988",
                    "description": "Started at S4",
                    "source": "interview.txt",
                    "confidence": 0.7,
                    "category": "professional",
                },
            ],
            "conflicts": [],
            "span": "1988-1989",
        }

        state: ResearchState = {
            "query": "Timeline of Lazar's disclosures",
            "corpus_results": [{"content": "test", "source_file": "test.txt"}],
            "trace": [],
        }

        result = timeline_node(state)

        assert len(result["timeline_results"]) == 2
        # Should be sorted chronologically
        assert result["timeline_results"][0]["date"] == "1988"
        assert result["timeline_results"][1]["date"] == "1989"

    @patch("src.agents.timeline_agent.corpus_search_tool")
    def test_handles_no_evidence(self, mock_search):
        from src.agents.timeline_agent import timeline_node

        mock_search.return_value = []

        state: ResearchState = {
            "query": "test",
            "corpus_results": [],
            "trace": [],
        }

        result = timeline_node(state)
        assert result["timeline_results"] == []


class TestFactCheckNode:
    """Test the fact-check agent node."""

    @patch("src.agents.fact_checker.llm_call")
    @patch("src.agents.fact_checker.filtered_search_tool")
    @patch("src.agents.fact_checker.corpus_search_tool")
    @patch("src.agents.fact_checker.assemble_context_tool")
    def test_produces_verdict(self, mock_context, mock_corpus, mock_filtered, mock_llm):
        from src.agents.fact_checker import fact_check_node

        mock_corpus.return_value = [
            {
                "content": "Evidence",
                "source_file": "test.txt",
                "source_type": "interview_transcript",
                "relevance_score": 0.8,
                "reliability_score": 0.7,
                "combined_score": 0.75,
                "excerpt": "Evidence",
            }
        ]
        mock_filtered.return_value = []
        mock_context.return_value = "Context"
        mock_llm.return_value = {
            "claim": "Lazar worked at Los Alamos",
            "verdict": "DISPUTED",
            "confidence": 0.6,
            "supporting_sources": [{"source": "interview.txt", "evidence": "He claims so"}],
            "contradicting_sources": [],
            "reasoning": "Only single source claim.",
        }

        state: ResearchState = {
            "query": "Is it true that Lazar worked at Los Alamos?",
            "corpus_results": [],
            "trace": [],
        }

        result = fact_check_node(state)

        assert len(result["fact_check_results"]) == 1
        assert result["fact_check_results"][0]["verdict"] == "DISPUTED"
        assert result["fact_check_results"][0]["confidence"] == 0.6

    @patch("src.agents.fact_checker.corpus_search_tool")
    @patch("src.agents.fact_checker.filtered_search_tool")
    def test_unverifiable_when_no_evidence(self, mock_filtered, mock_corpus):
        from src.agents.fact_checker import fact_check_node

        mock_corpus.return_value = []
        mock_filtered.return_value = []

        state: ResearchState = {
            "query": "Did aliens land in Roswell?",
            "corpus_results": [],
            "trace": [],
        }

        result = fact_check_node(state)
        assert result["fact_check_results"][0]["verdict"] == "UNVERIFIABLE"


# --- Orchestrator tests ---


class TestOrchestrator:
    """Test the orchestrator graph construction and routing."""

    def test_graph_builds_successfully(self):
        from src.agents.orchestrator import build_research_graph

        graph = build_research_graph()
        compiled = graph.compile()
        assert compiled is not None

    @patch("src.agents.orchestrator.llm_call")
    def test_analyze_query_node(self, mock_llm):
        from src.agents.orchestrator import analyze_query_node

        mock_llm.return_value = {
            "query_type": "verification",
            "agents": [
                {
                    "agent": "corpus_search",
                    "reason": "Search corpus",
                    "depends_on": [],
                    "priority": 0,
                },
                {
                    "agent": "fact_check",
                    "reason": "Verify claim",
                    "depends_on": ["corpus_search"],
                    "priority": 1,
                },
            ],
            "reasoning": "This is a verification query.",
        }

        state: ResearchState = {
            "query": "Is it true that Lazar worked at Los Alamos?",
            "trace": [],
        }

        result = analyze_query_node(state)

        assert result["query_type"] == "verification"
        assert len(result["research_plan"]) >= 1
        # corpus_search should always be included
        agent_names = {a["agent"] for a in result["research_plan"]}
        assert "corpus_search" in agent_names

    @patch("src.agents.orchestrator.llm_call")
    def test_analyze_query_adds_corpus_search_if_missing(self, mock_llm):
        from src.agents.orchestrator import analyze_query_node

        mock_llm.return_value = {
            "query_type": "timeline",
            "agents": [
                {"agent": "timeline", "reason": "Extract events", "depends_on": [], "priority": 0},
            ],
            "reasoning": "Timeline query.",
        }

        state: ResearchState = {"query": "Timeline test", "trace": []}
        result = analyze_query_node(state)

        agent_names = [a["agent"] for a in result["research_plan"]]
        assert "corpus_search" in agent_names

    def test_routing_functions(self):
        from src.agents.orchestrator import (
            should_cross_reference,
            should_fact_check,
            should_retry,
            should_timeline,
        )

        # Test cross-reference routing
        state_with_all: ResearchState = {
            "research_plan": [
                {"agent": "corpus_search"},
                {"agent": "cross_reference"},
                {"agent": "timeline"},
                {"agent": "fact_check"},
            ],
        }
        assert should_cross_reference(state_with_all) == "cross_reference"
        assert should_timeline(state_with_all) == "timeline"
        assert should_fact_check(state_with_all) == "fact_check"

        # Test minimal plan routing
        state_minimal: ResearchState = {
            "research_plan": [{"agent": "corpus_search"}],
        }
        assert should_cross_reference(state_minimal) == "synthesize"

        # Test retry logic
        state_low_conf: ResearchState = {
            "retries": 0,
            "max_retries": 2,
            "confidence": 0.2,
        }
        assert should_retry(state_low_conf) == "corpus_search"

        state_high_conf: ResearchState = {
            "retries": 0,
            "max_retries": 2,
            "confidence": 0.8,
        }
        from langgraph.graph import END

        assert should_retry(state_high_conf) == END

        state_max_retries: ResearchState = {
            "retries": 2,
            "max_retries": 2,
            "confidence": 0.2,
        }
        assert should_retry(state_max_retries) == END


# --- Observability tests ---


class TestTracer:
    """Test the research tracer."""

    def test_tracer_creates_trace_id(self):
        from src.observability.tracer import ResearchTracer

        tracer = ResearchTracer()
        assert tracer.trace_id.startswith("trace_")

    def test_tracer_custom_id(self):
        from src.observability.tracer import ResearchTracer

        tracer = ResearchTracer(trace_id="custom_123")
        assert tracer.trace_id == "custom_123"

    def test_span_recording(self):
        from src.observability.tracer import ResearchTracer

        tracer = ResearchTracer()
        span = tracer.start_span("test_node", "test_action", {"key": "value"})
        entry = tracer.end_span(span, outputs={"result": "ok"})

        assert entry["node"] == "test_node"
        assert entry["action"] == "test_action"
        assert entry["duration_ms"] >= 0
        assert entry["outputs"]["result"] == "ok"
        assert entry["error"] is None

    def test_span_with_error(self):
        from src.observability.tracer import ResearchTracer

        tracer = ResearchTracer()
        span = tracer.start_span("error_node", "fail")
        entry = tracer.end_span(span, error="Something went wrong")

        assert entry["error"] == "Something went wrong"

    def test_log_decision(self):
        from src.observability.tracer import ResearchTracer

        tracer = ResearchTracer()
        tracer.log_decision("orchestrator", "use_all_agents", "Complex query")

        assert len(tracer.entries) == 1
        assert tracer.entries[0]["action"] == "decision"

    def test_trace_summary(self):
        from src.observability.tracer import ResearchTracer

        tracer = ResearchTracer()
        span = tracer.start_span("node1", "action1")
        tracer.end_span(span)
        tracer.log_decision("node2", "skip")

        summary = tracer.get_summary()
        assert summary["trace_id"] == tracer.trace_id
        assert summary["num_decisions"] == 1
        assert len(summary["entries"]) == 2


class TestMetrics:
    """Test the metrics tracking."""

    def test_agent_metrics(self):
        from src.observability.metrics import AgentMetrics

        metrics = AgentMetrics(agent_name="corpus_search")
        metrics.tool_calls = 3
        metrics.llm_calls = 1
        metrics.complete(results_count=5)

        d = metrics.to_dict()
        assert d["agent_name"] == "corpus_search"
        assert d["tool_calls"] == 3
        assert d["results_count"] == 5
        assert d["duration_ms"] >= 0
        assert d["error"] is None

    def test_agent_metrics_failure(self):
        from src.observability.metrics import AgentMetrics

        metrics = AgentMetrics(agent_name="fact_check")
        metrics.fail("Connection timeout")

        d = metrics.to_dict()
        assert d["error"] == "Connection timeout"

    def test_pipeline_metrics(self):
        from src.observability.metrics import PipelineMetrics

        pm = PipelineMetrics(trace_id="test_trace", query="test query")
        m1 = pm.start_agent("corpus_search")
        m1.tool_calls = 2
        m1.llm_calls = 1
        m1.complete(results_count=5)

        m2 = pm.start_agent("fact_check")
        m2.tool_calls = 3
        m2.llm_calls = 1
        m2.complete(results_count=1)

        pm.complete(confidence=0.75)

        d = pm.to_dict()
        assert d["total_tool_calls"] == 5
        assert d["total_llm_calls"] == 2
        assert d["final_confidence"] == 0.75
        assert len(d["agents"]) == 2


# --- Tool wrapper tests ---


class TestTools:
    """Test tool wrappers (with mocked dependencies)."""

    @patch("src.agents.tools._get_llm_client")
    def test_llm_call_parses_json(self, mock_client_fn):
        from src.agents.tools import llm_call

        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": '{"answer": "test", "confidence": 0.8}'}
        }
        mock_client_fn.return_value = mock_client

        result = llm_call("system", "user prompt")
        assert result["answer"] == "test"
        assert result["confidence"] == 0.8

    @patch("src.agents.tools._get_llm_client")
    def test_llm_call_handles_non_json(self, mock_client_fn):
        from src.agents.tools import llm_call

        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "Not JSON at all"}}
        mock_client_fn.return_value = mock_client

        result = llm_call("system", "user prompt")
        assert result["parse_error"] is True
        assert "Not JSON" in result["raw_response"]


# --- API model tests ---


class TestAPIModels:
    """Test the new API models for agent endpoints."""

    def test_agent_query_model(self):
        from src.api.routes import AgentQuery

        q = AgentQuery(question="What did Lazar claim?")
        assert q.question == "What did Lazar claim?"

    def test_agent_query_max_length(self):
        from pydantic import ValidationError

        from src.api.routes import AgentQuery

        with pytest.raises(ValidationError):
            AgentQuery(question="x" * 10001)

    def test_agent_response_model(self):
        from src.api.routes import AgentResponse

        r = AgentResponse(
            synthesis="Test answer",
            confidence=0.75,
            query_type="factual",
            research_plan=[],
            trace_id="test_123",
        )
        assert r.confidence == 0.75
        assert r.trace_id == "test_123"
