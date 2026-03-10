"""Tests for the two-phase split pipeline (orchestrator → narrator)."""

import pytest
from unittest.mock import MagicMock, patch

from gm_agent.mcp.base import ToolDef, ToolParameter, ToolResult
from gm_agent.models.base import LLMBackend, LLMResponse, Message, ToolCall
from gm_agent.split import (
    ORCHESTRATOR_PROMPT,
    SYNTHESIZE_TOOL,
    RetrievalResult,
    SplitPipeline,
    SplitResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockLLM(LLMBackend):
    """Mock LLM that returns pre-configured responses in order."""

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self._index = 0
        self.calls: list[tuple[list[Message], list[ToolDef] | None, float | None]] = []

    def chat(self, messages, tools=None, thinking=None, temperature=None):
        self.calls.append((messages, tools, temperature))
        if self._index < len(self.responses):
            resp = self.responses[self._index]
            self._index += 1
            return resp
        return self.responses[-1]

    def get_model_name(self) -> str:
        return "mock-retrieval" if "retrieval" in str(id(self)) else "mock-model"

    def is_available(self) -> bool:
        return True


class MockMCP:
    """Mock MCPClient with RAG + dice + encounter tools."""

    def __init__(self, tool_results: dict[str, ToolResult] | None = None):
        self._results = tool_results or {}
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self) -> list[ToolDef]:
        return [
            ToolDef(
                name="lookup",
                description="Look up a named entity by type",
                parameters=[
                    ToolParameter(name="type", type="string", description="Entity type"),
                    ToolParameter(name="name", type="string", description="Name"),
                ],
            ),
            ToolDef(
                name="search_content",
                description="Search content",
                parameters=[
                    ToolParameter(name="query", type="string", description="Query"),
                ],
            ),
            ToolDef(
                name="roll_dice",
                description="Roll dice in PF2e notation",
                parameters=[
                    ToolParameter(name="expression", type="string", description="Dice expression"),
                ],
            ),
            ToolDef(
                name="roll_check",
                description="Roll a check with modifier vs DC",
                parameters=[
                    ToolParameter(name="modifier", type="integer", description="Modifier"),
                    ToolParameter(name="dc", type="integer", description="DC"),
                ],
            ),
            ToolDef(
                name="evaluate_encounter",
                description="Evaluate encounter threat",
                parameters=[
                    ToolParameter(name="creature_levels", type="string", description="Levels"),
                ],
            ),
        ]

    def call_tool(self, name: str, args: dict) -> ToolResult:
        self.calls.append((name, args))
        if name in self._results:
            return self._results[name]
        return ToolResult(success=True, data=f"Mock result for {name}({args})")

    def close(self):
        pass


def _make_pipeline(
    retrieval_responses: list[LLMResponse],
    synthesis_responses: list[LLMResponse],
    tool_results: dict[str, ToolResult] | None = None,
    verbose: bool = False,
) -> tuple[SplitPipeline, MockLLM, MockLLM, MockMCP]:
    """Create a SplitPipeline with configured mocks."""
    retrieval_llm = MockLLM(retrieval_responses)
    synthesis_llm = MockLLM(synthesis_responses)
    mcp = MockMCP(tool_results)

    pipeline = SplitPipeline(
        retrieval_llm=retrieval_llm,
        synthesis_llm=synthesis_llm,
        mcp=mcp,
        synthesis_system_prompt="You are a test assistant.",
        verbose=verbose,
    )
    return pipeline, retrieval_llm, synthesis_llm, mcp


# ---------------------------------------------------------------------------
# RetrievalResult / SplitResult dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_retrieval_result_creation(self):
        rr = RetrievalResult(
            tool_calls=[("lookup_creature", {"name": "goblin"}, "Goblin data")],
            duration_ms=100.0,
            model="test-model",
        )
        assert len(rr.tool_calls) == 1
        assert rr.model == "test-model"

    def test_split_result_creation(self):
        rr = RetrievalResult(tool_calls=[], duration_ms=50.0, model="m1")
        sr = SplitResult(
            response="Hello",
            retrieval_rounds=[rr],
            synthesis_model="m2",
            total_duration_ms=200.0,
        )
        assert sr.response == "Hello"
        assert sr.synthesis_model == "m2"

    def test_split_result_no_direct_tool_calls_field(self):
        """SplitResult should not have direct_tool_calls (removed in refactor)."""
        assert not hasattr(SplitResult, "direct_tool_calls") or "direct_tool_calls" not in SplitResult.__dataclass_fields__


# ---------------------------------------------------------------------------
# SYNTHESIZE_TOOL definition
# ---------------------------------------------------------------------------


class TestSynthesizeTool:
    def test_tool_name(self):
        assert SYNTHESIZE_TOOL.name == "synthesize"

    def test_no_parameters(self):
        assert SYNTHESIZE_TOOL.parameters == []

    def test_converts_to_openai_format(self):
        fmt = SYNTHESIZE_TOOL.to_openai_format()
        assert fmt["type"] == "function"
        assert fmt["function"]["name"] == "synthesize"
        # No required params
        props = fmt["function"]["parameters"].get("properties", {})
        assert len(props) == 0


# ---------------------------------------------------------------------------
# ORCHESTRATOR_PROMPT
# ---------------------------------------------------------------------------


class TestOrchestratorPrompt:
    def test_contains_synthesize_instruction(self):
        assert "synthesize" in ORCHESTRATOR_PROMPT

    def test_contains_encounter_guidance(self):
        assert "encounter" in ORCHESTRATOR_PROMPT.lower()

    def test_no_do_not_roll_dice(self):
        """Orchestrator should NOT say 'Do NOT roll dice' — it handles dice."""
        assert "Do NOT roll dice" not in ORCHESTRATOR_PROMPT

    def test_has_dedup_instruction(self):
        # Phrasing changed; check for the key concept
        assert "same tool" in ORCHESTRATOR_PROMPT

    def test_tells_not_to_answer(self):
        assert "Do NOT answer the question yourself" in ORCHESTRATOR_PROMPT


# ---------------------------------------------------------------------------
# Phase 1: Orchestrator
# ---------------------------------------------------------------------------


class TestOrchestratorPhase:
    def test_orchestrator_sees_all_tools_plus_synthesize(self):
        """Orchestrator should see ALL MCP tools + synthesize."""
        pipeline, ret_llm, _, _ = _make_pipeline(
            retrieval_responses=[LLMResponse(text="", tool_calls=[])],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
        )
        pipeline._orchestrate("test", [], max_iterations=1)

        tools = ret_llm.calls[0][1]
        tool_names = {t.name for t in tools}
        assert "lookup" in tool_names
        assert "search_content" in tool_names
        assert "roll_dice" in tool_names
        assert "roll_check" in tool_names
        assert "evaluate_encounter" in tool_names
        assert "synthesize" in tool_names

    def test_synthesize_stops_loop(self):
        """Calling synthesize should break out of the inner orchestration loop.

        The reflection loop may prompt the orchestrator again after synthesize,
        but with only a synthesize response queued for reflection rounds, no
        additional MCP tool calls should be made.
        """
        pipeline, _, _, mcp = _make_pipeline(
            retrieval_responses=[
                LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="c1", name="lookup_creature", args={"name": "goblin"}),
                        ToolCall(id="c2", name="search_rules", args={"query": "attack"}),
                        ToolCall(id="c3", name="search_content", args={"query": "goblin lore"}),
                        ToolCall(id="c4", name="synthesize", args={}),
                    ],
                ),
                # Reflection rounds: immediately synthesize again (no new tool calls)
                LLMResponse(text="", tool_calls=[
                    ToolCall(id="s2", name="synthesize", args={}),
                ]),
            ],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
        )
        result, _ = pipeline._orchestrate("goblin", [], max_iterations=5)
        assert len(mcp.calls) == 3
        assert mcp.calls[0] == ("lookup_creature", {"name": "goblin"})
        assert len(result.tool_calls) == 3

    def test_implicit_handoff_when_model_stops(self):
        """If model stops without calling synthesize, still hand off."""
        pipeline, _, _, mcp = _make_pipeline(
            retrieval_responses=[
                LLMResponse(
                    text="",
                    tool_calls=[ToolCall(id="c1", name="lookup_creature", args={"name": "goblin"})],
                ),
                LLMResponse(text="Done.", tool_calls=[]),
            ],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
        )
        result, _ = pipeline._orchestrate("goblin", [], max_iterations=5)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0][0] == "lookup_creature"

    def test_max_iterations_fallback(self):
        """Should stop after max_iterations even if model keeps calling tools."""
        pipeline, _, _, mcp = _make_pipeline(
            retrieval_responses=[
                LLMResponse(text="", tool_calls=[
                    ToolCall(id="c1", name="search_content", args={"query": "more 1"}),
                ]),
                LLMResponse(text="", tool_calls=[
                    ToolCall(id="c2", name="search_content", args={"query": "more 2"}),
                ]),
                LLMResponse(text="", tool_calls=[
                    ToolCall(id="c3", name="search_content", args={"query": "more 3"}),
                ]),
            ],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
        )
        result, _ = pipeline._orchestrate("endless", [], max_iterations=2)
        assert len(mcp.calls) == 2

    def test_executes_dice_tools(self):
        """Orchestrator should execute dice tools directly."""
        pipeline, _, _, mcp = _make_pipeline(
            retrieval_responses=[
                LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="d1", name="roll_dice", args={"expression": "1d20+7"}),
                        ToolCall(id="d2", name="roll_check", args={"modifier": 5, "dc": 15}),
                        ToolCall(id="d3", name="synthesize", args={}),
                    ],
                ),
            ],
            synthesis_responses=[LLMResponse(text="You rolled!", tool_calls=[])],
            tool_results={
                "roll_dice": ToolResult(success=True, data="d20: 14, total: 21"),
                "roll_check": ToolResult(success=True, data="d20: 10, total: 15, success"),
            },
        )
        result, _ = pipeline._orchestrate("Roll a d20+7", [], max_iterations=5)
        assert ("roll_dice", {"expression": "1d20+7"}) in mcp.calls
        assert len(result.tool_calls) == 2

    def test_mixes_rag_and_dice(self):
        """Orchestrator can mix search + dice in one pass."""
        pipeline, _, _, mcp = _make_pipeline(
            retrieval_responses=[
                LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="c1", name="lookup_creature", args={"name": "goblin"}),
                        ToolCall(id="d1", name="roll_check", args={"modifier": 7, "dc": 16}),
                        ToolCall(id="s1", name="synthesize", args={}),
                    ],
                ),
            ],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
            tool_results={
                "roll_check": ToolResult(success=True, data="d20: 14, total: 21, degree: success"),
            },
        )
        result, _ = pipeline._orchestrate("Goblin attacks AC 16", [], max_iterations=5)
        assert len(mcp.calls) == 2  # lookup_creature + roll_check
        assert len(result.tool_calls) == 2

    def test_dedup_works(self):
        """Duplicate tool calls should not hit MCP twice."""
        pipeline, _, _, mcp = _make_pipeline(
            retrieval_responses=[
                LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="c1", name="lookup_creature", args={"name": "goblin"}),
                        ToolCall(id="c2", name="lookup_creature", args={"name": "goblin"}),
                    ],
                ),
                LLMResponse(text="", tool_calls=[]),
            ],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
        )
        result, _ = pipeline._orchestrate("goblin", [], max_iterations=3)
        assert len(mcp.calls) == 1
        assert len(result.tool_calls) == 1

    def test_uses_orchestrator_prompt(self):
        """System prompt should include core and mode-appropriate content."""
        pipeline, ret_llm, _, _ = _make_pipeline(
            retrieval_responses=[LLMResponse(text="", tool_calls=[])],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
        )
        pipeline._orchestrate("test", [], max_iterations=1)
        system_msg = ret_llm.calls[0][0][0]
        assert system_msg.role == "system"
        # Default pipeline has campaign_mode=False → chat mode header, no campaign sections
        from gm_agent.split import _PROMPT_CORE, _CHAT_MODE_HEADER, _PROMPT_COMMON
        assert _PROMPT_CORE in system_msg.content
        assert _CHAT_MODE_HEADER in system_msg.content
        assert _PROMPT_COMMON in system_msg.content

    def test_uses_low_temperature(self):
        """Orchestrator should use RETRIEVAL_TEMPERATURE."""
        pipeline, ret_llm, _, _ = _make_pipeline(
            retrieval_responses=[LLMResponse(text="", tool_calls=[])],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
        )
        pipeline._orchestrate("test", [], max_iterations=1)
        _, _, temp = ret_llm.calls[0]
        assert temp == 0.0

    def test_includes_recent_turns(self):
        """Recent conversation turns should be included for pronoun resolution."""
        pipeline, ret_llm, _, _ = _make_pipeline(
            retrieval_responses=[LLMResponse(text="", tool_calls=[])],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
        )
        recent = [
            Message(role="user", content="Tell me about goblins."),
            Message(role="assistant", content="Goblins are small creatures."),
        ]
        pipeline._orchestrate("What about their tactics?", recent, max_iterations=1)
        messages = ret_llm.calls[0][0]
        # System + 2 recent + user query = 4
        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[1].content == "Tell me about goblins."

    def test_no_tool_calls_triggers_fallback(self):
        """If orchestrator makes no tool calls, a fallback search is triggered."""
        pipeline, _, _, mcp = _make_pipeline(
            retrieval_responses=[LLMResponse(text="No tools needed.", tool_calls=[])],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
        )
        result, _ = pipeline._orchestrate("test query", [], max_iterations=3)
        # Fallback search should have been triggered
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0][0] == "search_content"
        assert result.tool_calls[0][1] == {"query": "test query", "limit": 5}
        assert len(mcp.calls) == 1

    def test_llm_error_triggers_fallback(self):
        """If orchestrator LLM raises an exception, fallback search is triggered."""

        class ErrorLLM:
            def chat(self, messages, tools=None, temperature=None, **kw):
                raise ValueError("validation error for ToolCall")

            def get_model_name(self):
                return "error-model"

        mcp = MockMCP()
        pipeline = SplitPipeline(
            retrieval_llm=ErrorLLM(),
            synthesis_llm=MockLLM([]),
            mcp=mcp,
            synthesis_system_prompt="Test.",
        )
        result, _ = pipeline._orchestrate("treat wounds", [], max_iterations=3)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0][0] == "search_content"
        assert result.tool_calls[0][1] == {"query": "treat wounds", "limit": 5}

    def test_premature_synthesize_rejected(self):
        """Synthesize before MIN_TOOLS (2) calls is rejected, forcing more searching."""
        pipeline, ret_llm, _, mcp = _make_pipeline(
            retrieval_responses=[
                # Iteration 0: 1 call + synthesize → rejected (1 < 2)
                LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="c1", name="lookup_creature", args={"name": "goblin"}),
                        ToolCall(id="s1", name="synthesize", args={}),
                    ],
                ),
                # Iteration 1: 1 more call + synthesize → accepted (2 >= 2)
                LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="c2", name="search_lore", args={"query": "goblin"}),
                        ToolCall(id="s2", name="synthesize", args={}),
                    ],
                ),
            ],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
        )
        result, _ = pipeline._orchestrate("goblin", [], max_iterations=5)
        # Two tool calls should have executed before synthesize was accepted
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0][0] == "lookup_creature"
        assert result.tool_calls[1][0] == "search_lore"
        # One rejection message sent (at 1 tool call)
        all_messages = ret_llm.calls[-1][0]
        rejection_msgs = [
            m for m in all_messages
            if m.role == "tool" and "Only" in m.content and "tool call" in m.content
        ]
        assert len(rejection_msgs) == 1

    def test_implicit_handoff_nudge(self):
        """If model stops calling tools with too few results, nudge it to continue."""
        pipeline, ret_llm, _, mcp = _make_pipeline(
            retrieval_responses=[
                # Iteration 0: 1 tool call, then no more tools (implicit handoff attempt)
                LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="c1", name="lookup_creature", args={"name": "goblin"}),
                    ],
                ),
                # Iteration 1: nudged (1 < 2), returns no tools again
                LLMResponse(text="I think that's enough.", tool_calls=[]),
                # Iteration 2: nudged again (1 < 3), model searches more this time
                LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="c2", name="search_lore", args={"query": "goblin"}),
                        ToolCall(id="c3", name="search_content", args={"query": "goblin tactics"}),
                        ToolCall(id="s1", name="synthesize", args={}),
                    ],
                ),
            ],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
        )
        result, _ = pipeline._orchestrate("goblin", [], max_iterations=5)
        # All three tool calls should have executed
        assert len(result.tool_calls) == 3
        assert result.tool_calls[0][0] == "lookup_creature"
        assert result.tool_calls[1][0] == "search_lore"
        assert result.tool_calls[2][0] == "search_content"
        # Nudge messages should have been injected (twice: iterations 1 and 2)
        all_messages = ret_llm.calls[-1][0]
        nudge_msgs = [
            m for m in all_messages
            if m.role == "user" and "tool call" in m.content.lower()
        ]
        assert len(nudge_msgs) >= 1

    def test_implicit_handoff_allowed_on_last_iteration(self):
        """On the last iteration, implicit handoff is allowed even with few results."""
        pipeline, _, _, mcp = _make_pipeline(
            retrieval_responses=[
                LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="c1", name="lookup_creature", args={"name": "goblin"}),
                    ],
                ),
                # Iteration 1 (last): no tools → allowed to hand off
                LLMResponse(text="Done.", tool_calls=[]),
            ],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
        )
        result, _ = pipeline._orchestrate("goblin", [], max_iterations=2)
        assert len(result.tool_calls) == 1

    def test_synthesize_allowed_on_last_iteration(self):
        """On the final iteration, synthesize is always allowed even with few results."""
        pipeline, _, _, mcp = _make_pipeline(
            retrieval_responses=[
                # Only 1 tool call + synthesize, but max_iterations=1
                LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="c1", name="lookup_creature", args={"name": "goblin"}),
                        ToolCall(id="s1", name="synthesize", args={}),
                    ],
                ),
            ],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
        )
        result, _ = pipeline._orchestrate("goblin", [], max_iterations=1)
        # Should be allowed since it's the last iteration
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0][0] == "lookup_creature"


# ---------------------------------------------------------------------------
# Phase 2: Narrator (always toolless)
# ---------------------------------------------------------------------------


class TestNarratorPhase:
    def test_narrator_receives_no_tools(self):
        """Narrator should always receive tools=None."""
        pipeline, _, syn_llm, _ = _make_pipeline(
            retrieval_responses=[LLMResponse(text="", tool_calls=[])],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
        )
        retrievals = [RetrievalResult(tool_calls=[], duration_ms=10, model="m")]
        pipeline._narrate("test", retrievals, None, "")

        tools = syn_llm.calls[0][1]
        assert tools is None

    def test_narrator_answers_directly(self):
        """Narrator answers directly without tools."""
        pipeline, _, syn_llm, _ = _make_pipeline(
            retrieval_responses=[LLMResponse(text="", tool_calls=[])],
            synthesis_responses=[
                LLMResponse(text="Goblins are small creatures.", tool_calls=[]),
            ],
        )
        retrievals = [RetrievalResult(
            tool_calls=[("lookup_creature", {"name": "goblin"}, "Goblin data here")],
            duration_ms=50.0,
            model="mock-ret",
        )]
        text, _ = pipeline._narrate("What is a goblin?", retrievals, None, "")
        assert text == "Goblins are small creatures."

    def test_narrator_receives_reference_material(self):
        """Narrator should receive formatted reference material."""
        pipeline, _, syn_llm, _ = _make_pipeline(
            retrieval_responses=[LLMResponse(text="", tool_calls=[])],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
        )
        retrievals = [RetrievalResult(
            tool_calls=[("lookup_creature", {"name": "goblin"}, "**Goblin** Level -1")],
            duration_ms=50.0,
            model="mock",
        )]
        pipeline._narrate("goblin?", retrievals, None, "")

        user_msg = syn_llm.calls[0][0][-1]
        assert user_msg.role == "user"
        assert "## Reference Material" in user_msg.content
        assert "lookup_creature" in user_msg.content
        assert "**Goblin** Level -1" in user_msg.content

    def test_narrator_uses_synthesis_temperature(self):
        """Narrator should use SYNTHESIS_TEMPERATURE."""
        from gm_agent.config import SYNTHESIS_TEMPERATURE
        pipeline, _, syn_llm, _ = _make_pipeline(
            retrieval_responses=[LLMResponse(text="", tool_calls=[])],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
        )
        retrievals = [RetrievalResult(tool_calls=[], duration_ms=10, model="m")]
        pipeline._narrate("test", retrievals, None, "")

        _, _, temp = syn_llm.calls[0]
        assert temp == SYNTHESIS_TEMPERATURE

    def test_narrator_includes_conversation_history(self):
        """Narrator should include conversation history."""
        pipeline, _, syn_llm, _ = _make_pipeline(
            retrieval_responses=[LLMResponse(text="", tool_calls=[])],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
        )
        retrievals = [RetrievalResult(tool_calls=[], duration_ms=10, model="m")]
        history = [
            Message(role="user", content="Previous question"),
            Message(role="assistant", content="Previous answer"),
        ]
        pipeline._narrate("New question", retrievals, history, "")

        messages = syn_llm.calls[0][0]
        assert len(messages) == 4  # system + history(2) + user
        assert messages[1].content == "Previous question"
        assert messages[2].content == "Previous answer"

    def test_narrator_includes_extra_context(self):
        """Extra context should be prepended to user message."""
        pipeline, _, syn_llm, _ = _make_pipeline(
            retrieval_responses=[LLMResponse(text="", tool_calls=[])],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
        )
        retrievals = [RetrievalResult(tool_calls=[], duration_ms=10, model="m")]
        pipeline._narrate("question?", retrievals, None, "Campaign: Rise of Runelords")

        user_msg = syn_llm.calls[0][0][-1]
        assert user_msg.content.startswith("Campaign: Rise of Runelords")

    def test_narrator_empty_response_retry(self):
        """If narrator returns empty, pipeline should retry once."""
        pipeline, _, syn_llm, _ = _make_pipeline(
            retrieval_responses=[LLMResponse(text="", tool_calls=[])],
            synthesis_responses=[
                LLMResponse(text="", tool_calls=[]),
                LLMResponse(text="Here is the answer.", tool_calls=[]),
            ],
        )
        retrievals = [RetrievalResult(
            tool_calls=[("lookup_creature", {"name": "goblin"}, "Goblin data")],
            duration_ms=50, model="m",
        )]
        text, _ = pipeline._narrate("goblin?", retrievals, None, "")
        assert text == "Here is the answer."
        assert len(syn_llm.calls) == 2

    def test_narrator_empty_response_hard_fallback(self):
        """If retry also returns empty, return a generic error message."""
        pipeline, _, _, _ = _make_pipeline(
            retrieval_responses=[LLMResponse(text="", tool_calls=[])],
            synthesis_responses=[
                LLMResponse(text="", tool_calls=[]),
                LLMResponse(text="", tool_calls=[]),
            ],
        )
        retrievals = [RetrievalResult(
            tool_calls=[("lookup_creature", {"name": "goblin"}, "**Goblin** Level -1")],
            duration_ms=50, model="m",
        )]
        text, _ = pipeline._narrate("goblin?", retrievals, None, "")
        assert "wasn't able to formulate" in text

    def test_narrator_empty_response_no_results_fallback(self):
        """If empty response and no results, return generic fallback."""
        pipeline, _, _, _ = _make_pipeline(
            retrieval_responses=[LLMResponse(text="", tool_calls=[])],
            synthesis_responses=[
                LLMResponse(text="", tool_calls=[]),
                LLMResponse(text="   ", tool_calls=[]),
            ],
        )
        retrievals = [RetrievalResult(tool_calls=[], duration_ms=10, model="m")]
        text, _ = pipeline._narrate("test?", retrievals, None, "")
        assert "rephrasing" in text.lower()


# ---------------------------------------------------------------------------
# Dice in orchestrator (end-to-end through run())
# ---------------------------------------------------------------------------


class TestDiceInOrchestrator:
    def test_pure_dice_query(self):
        """Pure dice query: orchestrator rolls, synthesize, narrator narrates."""
        pipeline, _, _, mcp = _make_pipeline(
            retrieval_responses=[
                LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="d1", name="roll_check", args={"modifier": 8, "dc": 20}),
                        ToolCall(id="s1", name="synthesize", args={}),
                    ],
                ),
            ],
            synthesis_responses=[
                LLMResponse(text="You rolled a 22, that's a success!", tool_calls=[]),
            ],
            tool_results={
                "roll_check": ToolResult(success=True, data="d20: 14, total: 22, degree: success"),
            },
        )
        result = pipeline.run("Roll stealth +8 vs DC 20")
        assert result.response == "You rolled a 22, that's a success!"
        assert ("roll_check", {"modifier": 8, "dc": 20}) in mcp.calls

    def test_mixed_search_and_dice(self):
        """Mixed query: orchestrator searches + rolls, narrator narrates with all."""
        pipeline, _, _, mcp = _make_pipeline(
            retrieval_responses=[
                LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="c1", name="lookup_creature", args={"name": "goblin"}),
                        ToolCall(id="d1", name="roll_dice", args={"expression": "1d6+2"}),
                        ToolCall(id="s1", name="synthesize", args={}),
                    ],
                ),
            ],
            synthesis_responses=[
                LLMResponse(text="The goblin deals 5 damage!", tool_calls=[]),
            ],
            tool_results={
                "roll_dice": ToolResult(success=True, data="d6: 3, total: 5"),
            },
        )
        result = pipeline.run("Goblin attacks, roll 1d6+2 damage")
        assert result.response == "The goblin deals 5 damage!"
        assert len(result.retrieval_rounds[0].tool_calls) == 2

    def test_dice_results_in_reference_material(self):
        """Dice results from orchestrator should appear in narrator's Reference Material."""
        pipeline, _, syn_llm, _ = _make_pipeline(
            retrieval_responses=[
                LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="d1", name="roll_dice", args={"expression": "1d20"}),
                        ToolCall(id="s1", name="synthesize", args={}),
                    ],
                ),
            ],
            synthesis_responses=[
                LLMResponse(text="You rolled a 17!", tool_calls=[]),
            ],
            tool_results={
                "roll_dice": ToolResult(success=True, data="d20: 17, total: 17"),
            },
        )
        pipeline.run("Roll a d20")

        # Check what the narrator received
        user_msg = syn_llm.calls[0][0][-1]
        assert "## Reference Material" in user_msg.content
        assert "roll_dice" in user_msg.content
        assert "17" in user_msg.content

    def test_encounter_evaluation_in_orchestrator(self):
        """Encounter evaluation should run in orchestrator."""
        pipeline, _, _, mcp = _make_pipeline(
            retrieval_responses=[
                LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="e1", name="evaluate_encounter", args={"creature_levels": "[-1, -1, -1, -1]"}),
                        ToolCall(id="s1", name="synthesize", args={}),
                    ],
                ),
            ],
            synthesis_responses=[
                LLMResponse(text="This is a moderate encounter.", tool_calls=[]),
            ],
            tool_results={
                "evaluate_encounter": ToolResult(success=True, data="Threat: moderate, XP: 80"),
            },
        )
        result = pipeline.run("Evaluate 4 goblins vs level 1 party")
        assert result.response == "This is a moderate encounter."
        assert ("evaluate_encounter", {"creature_levels": "[-1, -1, -1, -1]"}) in mcp.calls


# ---------------------------------------------------------------------------
# Full pipeline run()
# ---------------------------------------------------------------------------


class TestSplitPipelineRun:
    def test_basic_run(self):
        """Full pipeline: orchestrator gets data, narrator answers."""
        pipeline, _, _, mcp = _make_pipeline(
            retrieval_responses=[
                LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="c1", name="lookup_creature", args={"name": "goblin"}),
                        ToolCall(id="s1", name="synthesize", args={}),
                    ],
                ),
            ],
            synthesis_responses=[
                LLMResponse(text="Goblins are level -1 creatures.", tool_calls=[]),
            ],
        )

        result = pipeline.run("What is a goblin?")

        assert isinstance(result, SplitResult)
        assert result.response == "Goblins are level -1 creatures."
        assert len(result.retrieval_rounds) == 1
        assert len(result.retrieval_rounds[0].tool_calls) == 1
        assert result.total_duration_ms > 0

    def test_run_with_conversation_history(self):
        """Pipeline should pass conversation history through."""
        pipeline, _, syn_llm, _ = _make_pipeline(
            retrieval_responses=[LLMResponse(text="", tool_calls=[])],
            synthesis_responses=[LLMResponse(text="Their AC is 16.", tool_calls=[])],
        )

        history = [
            Message(role="user", content="Tell me about goblins."),
            Message(role="assistant", content="Goblins are small."),
        ]
        result = pipeline.run("What's their AC?", conversation_history=history)
        assert result.response == "Their AC is 16."

    def test_run_empty_orchestration_triggers_fallback(self):
        """Pipeline should auto-search when orchestrator returns no tools."""
        pipeline, _, _, _ = _make_pipeline(
            retrieval_responses=[LLMResponse(text="No tools needed.", tool_calls=[])],
            synthesis_responses=[
                LLMResponse(text="Here's what I found.", tool_calls=[]),
            ],
        )
        result = pipeline.run("What is the meaning of life?")
        assert result.response == "Here's what I found."
        # Fallback search should have been triggered
        assert len(result.retrieval_rounds[0].tool_calls) == 1
        assert result.retrieval_rounds[0].tool_calls[0][0] == "search_content"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_format_retrieval_empty(self):
        rr = RetrievalResult(tool_calls=[], duration_ms=10, model="m")
        text = SplitPipeline._format_retrieval(rr)
        assert text == "No results found."

    def test_format_retrieval_single_tool(self):
        rr = RetrievalResult(
            tool_calls=[("lookup_creature", {"name": "goblin"}, "**Goblin** Level -1")],
            duration_ms=50,
            model="m",
        )
        text = SplitPipeline._format_retrieval(rr)
        assert "### lookup_creature(name='goblin')" in text
        assert "**Goblin** Level -1" in text

    def test_format_retrieval_multiple_tools(self):
        rr = RetrievalResult(
            tool_calls=[
                ("lookup_creature", {"name": "goblin"}, "Goblin data"),
                ("search_content", {"query": "goblin lore"}, "Lore data"),
            ],
            duration_ms=100,
            model="m",
        )
        text = SplitPipeline._format_retrieval(rr)
        assert "### lookup_creature" in text
        assert "### search_content" in text
        assert "Goblin data" in text
        assert "Lore data" in text

    def test_format_all_retrievals_empty(self):
        pipeline, _, _, _ = _make_pipeline(
            retrieval_responses=[],
            synthesis_responses=[],
        )
        rr = RetrievalResult(tool_calls=[], duration_ms=10, model="m")
        text = pipeline._format_all_retrievals([rr])
        assert text == ""

    def test_format_all_retrievals_with_data(self):
        pipeline, _, _, _ = _make_pipeline(
            retrieval_responses=[],
            synthesis_responses=[],
        )
        rr = RetrievalResult(
            tool_calls=[("lookup_creature", {"name": "goblin"}, "Data")],
            duration_ms=50,
            model="m",
        )
        text = pipeline._format_all_retrievals([rr])
        assert text.startswith("## Reference Material")
        assert "lookup_creature" in text

    def test_format_all_retrievals_skips_empty_rounds(self):
        pipeline, _, _, _ = _make_pipeline(
            retrieval_responses=[],
            synthesis_responses=[],
        )
        empty = RetrievalResult(tool_calls=[], duration_ms=10, model="m")
        full = RetrievalResult(
            tool_calls=[("search_content", {"query": "x"}, "Result")],
            duration_ms=50,
            model="m",
        )
        text = pipeline._format_all_retrievals([empty, full])
        assert "## Reference Material" in text
        assert "search_content" in text
        assert "No results found" not in text


# ---------------------------------------------------------------------------
# ChatAgent always creates pipeline
# ---------------------------------------------------------------------------


class TestChatAgentAlwaysPipeline:
    def test_pipeline_always_created(self):
        """ChatAgent should always create a pipeline."""
        mock_llm = MockLLM([LLMResponse(text="Hello", tool_calls=[])])
        agent = _make_chat_agent(mock_llm)
        assert agent._pipeline is not None
        agent.close()

    def test_no_split_mode_attribute(self):
        """ChatAgent should not have _split_mode attribute."""
        mock_llm = MockLLM([LLMResponse(text="Hello", tool_calls=[])])
        agent = _make_chat_agent(mock_llm)
        assert not hasattr(agent, "_split_mode")
        agent.close()

    def test_chat_uses_pipeline(self):
        """ChatAgent.chat() should use the pipeline."""
        # llm is both orchestrator and narrator: orchestrator (no tools) + narrator
        mock_llm = MockLLM([
            LLMResponse(text="", tool_calls=[]),        # orchestrator
            LLMResponse(text="Split answer.", tool_calls=[]),  # narrator
        ])
        agent = _make_chat_agent(mock_llm)
        response = agent.chat("test question")
        assert response == "Split answer."
        assert len(agent._messages) == 3  # system + user + assistant
        agent.close()

    def test_chat_maintains_history(self):
        """Chat should add user/assistant messages to history."""
        # 2 chat calls x 2 LLM calls each (orchestrator + narrator)
        mock_llm = MockLLM([
            LLMResponse(text="", tool_calls=[]),           # orchestrator 1
            LLMResponse(text="First answer.", tool_calls=[]),   # narrator 1
            LLMResponse(text="", tool_calls=[]),           # orchestrator 2
            LLMResponse(text="Second answer.", tool_calls=[]),  # narrator 2
        ])
        agent = _make_chat_agent(mock_llm)
        agent.chat("first")
        agent.chat("second")
        assert len(agent._messages) == 5  # system + user + assistant + user + assistant
        agent.close()


def _make_chat_agent(llm):
    """Create a ChatAgent, avoiding real backend initialization."""
    from gm_agent.chat import ChatAgent

    return ChatAgent(llm=llm)


# ---------------------------------------------------------------------------
# Config and factory
# ---------------------------------------------------------------------------


class TestConfig:
    def test_no_split_mode_config(self):
        """SPLIT_MODE should no longer exist in config."""
        import gm_agent.config as config
        assert not hasattr(config, "SPLIT_MODE")

    def test_retrieval_temperature_default(self):
        from gm_agent.config import RETRIEVAL_TEMPERATURE
        assert isinstance(RETRIEVAL_TEMPERATURE, float)


class TestFactory:
    def test_get_retrieval_backend_returns_backend(self):
        with patch("gm_agent.models.factory.LLM_BACKEND", "ollama"):
            with patch("gm_agent.models.factory.RETRIEVAL_BACKEND", None):
                with patch("gm_agent.models.factory.RETRIEVAL_MODEL", None):
                    from gm_agent.models.factory import get_retrieval_backend
                    backend = get_retrieval_backend()
                    assert isinstance(backend, LLMBackend)

    def test_no_synthesis_backend_function(self):
        """get_synthesis_backend should no longer exist."""
        import gm_agent.models.factory as factory_mod
        assert not hasattr(factory_mod, "get_synthesis_backend")


# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------


class TestVerboseOutput:
    def test_verbose_orchestrator(self, capsys):
        """Verbose mode should print orchestrator info."""
        pipeline, _, _, _ = _make_pipeline(
            retrieval_responses=[
                LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="c1", name="lookup_creature", args={"name": "goblin"}),
                        ToolCall(id="s1", name="synthesize", args={}),
                    ],
                ),
            ],
            synthesis_responses=[LLMResponse(text="Answer.", tool_calls=[])],
            verbose=True,
        )
        pipeline.run("goblin")
        captured = capsys.readouterr()
        assert "[Orchestrator]" in captured.out
        assert "lookup_creature" in captured.out


# ---------------------------------------------------------------------------
# SYNTHESIS_SYSTEM_PROMPT
# ---------------------------------------------------------------------------


class TestSynthesisPrompt:
    def test_synthesis_prompt_exists(self):
        from gm_agent.chat import SYNTHESIS_SYSTEM_PROMPT
        assert "GROUNDING" in SYNTHESIS_SYSTEM_PROMPT

    def test_synthesis_prompt_no_retrieve_more(self):
        """Synthesis prompt should not mention retrieve_more (narrator is toolless)."""
        from gm_agent.chat import SYNTHESIS_SYSTEM_PROMPT
        assert "retrieve_more" not in SYNTHESIS_SYSTEM_PROMPT

    def test_synthesis_prompt_says_no_tools(self):
        """Synthesis prompt should say no tools available."""
        from gm_agent.chat import SYNTHESIS_SYSTEM_PROMPT
        assert "No tools" in SYNTHESIS_SYSTEM_PROMPT or "no tools" in SYNTHESIS_SYSTEM_PROMPT.lower()

    def test_synthesis_prompt_no_tool_guidance(self):
        """Synthesis prompt should not contain tool selection guidance."""
        from gm_agent.chat import SYNTHESIS_SYSTEM_PROMPT
        assert "lookup_creature" not in SYNTHESIS_SYSTEM_PROMPT
        assert "search_content" not in SYNTHESIS_SYSTEM_PROMPT

    def test_synthesis_prompt_no_dice_tools(self):
        """Synthesis prompt should NOT list dice tools (handled by orchestrator)."""
        from gm_agent.chat import SYNTHESIS_SYSTEM_PROMPT
        assert "roll_dice" not in SYNTHESIS_SYSTEM_PROMPT
        assert "roll_check" not in SYNTHESIS_SYSTEM_PROMPT
        assert "evaluate_encounter" not in SYNTHESIS_SYSTEM_PROMPT

    def test_synthesis_prompt_mentions_results_already_performed(self):
        """Prompt should clarify no tools available (results already gathered)."""
        from gm_agent.chat import SYNTHESIS_SYSTEM_PROMPT
        assert "No tools available" in SYNTHESIS_SYSTEM_PROMPT

    def test_synthesis_prompt_has_do_not_fabricate(self):
        """Prompt should warn against fabricating dice results."""
        from gm_agent.chat import SYNTHESIS_SYSTEM_PROMPT
        assert "fabricate dice" in SYNTHESIS_SYSTEM_PROMPT

    def test_chat_system_prompt_has_query_tips(self):
        from gm_agent.chat import CHAT_SYSTEM_PROMPT
        assert "SEARCH QUERY TIPS" in CHAT_SYSTEM_PROMPT
        assert "1-3 key terms" in CHAT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# _stable_args_key helper
# ---------------------------------------------------------------------------


class TestStableArgsKey:
    def test_same_args_same_key(self):
        from gm_agent.split import _stable_args_key
        a = {"name": "goblin", "limit": 5}
        b = {"limit": 5, "name": "goblin"}
        assert _stable_args_key(a) == _stable_args_key(b)

    def test_different_args_different_key(self):
        from gm_agent.split import _stable_args_key
        a = {"name": "goblin"}
        b = {"name": "dragon"}
        assert _stable_args_key(a) != _stable_args_key(b)


# ---------------------------------------------------------------------------
# Removed concepts should not exist
# ---------------------------------------------------------------------------


class TestRemovedConcepts:
    def test_no_retrieve_more_tool(self):
        """RETRIEVE_MORE_TOOL should no longer exist."""
        import gm_agent.split as split_mod
        assert not hasattr(split_mod, "RETRIEVE_MORE_TOOL")

    def test_no_retrieval_prompt(self):
        """RETRIEVAL_PROMPT should no longer exist."""
        import gm_agent.split as split_mod
        assert not hasattr(split_mod, "RETRIEVAL_PROMPT")

    def test_no_narrator_tools_param(self):
        """SplitPipeline should not accept narrator_tools parameter."""
        import inspect
        sig = inspect.signature(SplitPipeline.__init__)
        assert "narrator_tools" not in sig.parameters

    def test_no_direct_tool_names_param(self):
        """SplitPipeline should not accept direct_tool_names parameter."""
        import inspect
        sig = inspect.signature(SplitPipeline.__init__)
        assert "direct_tool_names" not in sig.parameters

    def test_no_default_direct_tools_constant(self):
        """DEFAULT_DIRECT_TOOLS should no longer exist."""
        import gm_agent.split as split_mod
        assert not hasattr(split_mod, "DEFAULT_DIRECT_TOOLS")

    def test_no_tools_selection_prompt(self):
        """TOOLS_SELECTION_PROMPT should no longer exist."""
        import gm_agent.split as split_mod
        assert not hasattr(split_mod, "TOOLS_SELECTION_PROMPT")

    def test_no_retrieve_method(self):
        """SplitPipeline._retrieve() should no longer exist."""
        assert not hasattr(SplitPipeline, "_retrieve")

    def test_no_max_synthesis_retrievals_param(self):
        """run() should not accept max_synthesis_retrievals."""
        import inspect
        sig = inspect.signature(SplitPipeline.run)
        assert "max_synthesis_retrievals" not in sig.parameters
