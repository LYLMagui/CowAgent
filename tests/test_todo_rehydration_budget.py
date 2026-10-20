"""Regression tests for legacy todos, request budgets, and compaction recovery."""

import copy
import threading

import pytest

from agent.memory.conversation_store import ConversationStore
from agent.protocol.agent import Agent
from agent.protocol.agent_stream import AgentStreamExecutor
from agent.tools.todo_write.todo_write_tool import TodoWriteTool


def _message(role, text):
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _store(path, todos):
    store = ConversationStore(path, agent_id="owner")
    store.append_messages("session", [_message("user", "开始任务")])
    store.commit_todo_snapshot("session", "old-run", "old-call", todos)
    return store


def _executor(store, messages, window=32000):
    agent = Agent.__new__(Agent)
    agent.memory_manager = None
    agent.max_context_tokens = None
    agent.messages_lock = threading.RLock()
    agent.messages = messages
    agent._get_model_context_window = lambda: window
    agent._get_output_reserve_tokens = lambda: 1000
    executor = AgentStreamExecutor(agent, None, "", [TodoWriteTool()], messages=messages)
    executor.bind_todo_owner(store, "session", "owner", "run")
    return executor


def _injected_text(executor):
    message = next(m for m in reversed(executor._prepare_messages()) if m["role"] == "user")
    return message["content"][-1]["text"]


def test_legacy_twelve_items_can_be_restored_and_replaced(tmp_path):
    todos = [{"content": f"旧任务 {index}", "status": "pending"} for index in range(12)]
    store = _store(tmp_path / "index.db", todos)
    executor = _executor(store, [_message("user", "整理任务")])
    text = _injected_text(executor)
    assert all(item["content"] in text for item in todos)
    assert "12" in text and "10" in text
    assert store.get_todo_snapshot("session")["todos"] == todos

    tool = TodoWriteTool()
    replacement, explanation = tool._validate({"todos": todos[:10]})
    store.commit_todo_snapshot("session", "run", "replace", replacement, explanation)
    assert store.get_todo_snapshot("session")["version"] == 2
    assert len(store.get_todo_snapshot("session")["todos"]) == 10
    with pytest.raises(ValueError, match="at most 10"):
        tool._validate({"todos": todos})


@pytest.mark.parametrize("content", ["日志" * 20000, "log " * 20000], ids=["cjk", "ascii"])
def test_long_descriptions_are_bounded_without_changing_storage(tmp_path, content):
    todos = [{"content": content + str(i), "status": "pending"} for i in range(10)]
    store = _store(tmp_path / "index.db", todos)
    executor = _executor(store, [_message("user", "继续")])
    original = copy.deepcopy(executor.messages)
    text = _injected_text(executor)
    assert Agent._estimate_text_tokens(text) <= (32000 - 1000) // 20
    assert "…" in text
    assert "缩略" in text or "abbreviated" in text
    assert store.get_todo_snapshot("session")["todos"] == todos
    assert executor.messages == original
    assert _injected_text(executor) == text


def test_huge_legacy_list_uses_explicit_summary_without_blocking(tmp_path):
    todos = [{"content": f"任务{i}", "status": "completed"} for i in range(1000)]
    store = _store(tmp_path / "index.db", todos)
    executor = _executor(store, [_message("user", "清空旧计划")])
    text = _injected_text(executor)
    assert Agent._estimate_text_tokens(text) <= 1550
    assert "1000" in text
    assert "不要" in text or "Do not replace or rebuild" in text
    assert store.get_todo_snapshot("session")["todos"] == todos


def test_proactive_trim_reserves_todo_tokens(tmp_path):
    store = _store(tmp_path / "index.db", [{"content": "任务" * 5000, "status": "pending"}])
    messages = []
    for index in range(5):
        messages.extend([_message("user", "q" * 3500), _message("assistant", "a" * 3500)])
    executor = _executor(store, messages, window=10000)
    original = copy.deepcopy(messages)
    history_tokens = sum(executor._estimate_turn_tokens(t) for t in executor._identify_complete_turns())
    assert history_tokens < 9000
    assert history_tokens + executor._todo_context_tokens() > 9000
    executor._trim_messages()
    assert len(executor.messages) < len(original)
    assert executor._todo_context_tokens() > 0
    assert store.get_todo_snapshot("session")["version"] == 1


def test_overflow_reserves_todo_and_retries_when_only_todo_shrinks(tmp_path):
    store = _store(tmp_path / "index.db", [{"content": "任务" * 20000, "status": "in_progress"}])
    executor = _executor(store, [_message("user", "继续")], window=128000)
    before = executor._todo_context_tokens()
    error = "maximum context length is 12000 tokens (2000 in the completion)"
    assert executor._smart_compact_to_budget(error)
    after = executor._todo_context_tokens()
    assert after < before
    assert after <= 450
    system_tokens = executor.agent._estimate_message_tokens({"role": "system", "content": ""})
    assert executor._overflow_input_budget(error) == int((10000 - system_tokens) * 0.9) - after
    assert len(executor.messages) == 1


@pytest.mark.parametrize("todos", [[], [{"content": "超长任务" * 20000, "status": "pending"}]])
def test_tiny_provider_budget_omits_note_without_changing_snapshot(tmp_path, todos):
    store = _store(tmp_path / "index.db", todos)
    executor = _executor(store, [_message("user", "继续")])
    before = store.get_todo_snapshot("session")
    budget = executor._overflow_input_budget(
        "maximum context length is 2100 tokens (2000 in the completion)"
    )
    assert 0 < budget < 100
    assert executor._todo_context_text() == ""
    assert executor._prepare_messages() == executor.messages
    assert store.get_todo_snapshot("session") == before


def test_manual_compact_and_reopen_restore_latest_snapshot(tmp_path):
    path = tmp_path / "index.db"
    todos = [{"content": "修复完成后执行验证", "status": "in_progress"}]
    store = _store(path, todos)
    messages = []
    for i in range(4):
        messages.extend([_message("user", f"用户{i}"), _message("assistant", f"回复{i}")])
    executor = _executor(store, messages)
    assert executor.agent.compact_context(keep_recent_turns=1)["ok"]
    executor.messages = executor.agent.messages
    assert "修复完成后执行验证" in _injected_text(executor)
    reopened = ConversationStore(path, agent_id="owner")
    restored = _executor(reopened, [_message("user", "重启后继续")])
    assert "修复完成后执行验证" in _injected_text(restored)
    assert len(restored.messages[0]["content"]) == 1
