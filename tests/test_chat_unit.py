from unittest.mock import patch
from src.backend.routers.chat import (
    _compact_strategy_state,
    _cache_key,
    _build_llm_prompt,
    ChatMessage
)

def test_compact_strategy_state_names_resolved():
    live = {
        "leader": "LEC", 
        "positions": {"LEC": 1, "NOR": 2},
        "gaps": {"LEC": 0.0, "NOR": 1.4}, 
        "current_lap": 32, 
        "total_laps": 57
    }
    with patch("src.backend.routers.chat._load_laps_for_session", return_value=[]):
        state = _compact_strategy_state(live)
        assert state["leader_full_name"] == "Charles Leclerc"
        assert any(p["name"] == "Lando Norris" for p in state["top5_performance"])

def test_cache_key_changes_on_lap_change():
    k1 = _cache_key(
        session_id="s1", 
        user_prompt="who leads",
        category=None, 
        live_context={"current_lap": 10}
    )
    k2 = _cache_key(
        session_id="s1", 
        user_prompt="who leads",
        category=None, 
        live_context={"current_lap": 11}
    )
    assert k1 != k2

def test_strategy_prompt_contains_state():
    msgs = [ChatMessage(role="user", content="can LEC catch NOR?")]
    prompt = _build_llm_prompt(
        retrieved_context="", 
        messages=msgs,
        category="strategy",
        live_context={"leader": "LEC"}
    )
    assert "Charles Leclerc" in prompt
    assert "strategy" in prompt.lower()
