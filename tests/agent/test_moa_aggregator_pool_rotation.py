"""A 429/401/402 from the MoA aggregator must rotate the aggregator provider's pool (#132284).

With ``provider: moa`` the main loop's ``recover_with_credential_pool`` bound the virtual
provider's empty pool: the failed aggregator credential was never marked exhausted, no pooled
account was tried, and the turn sat in the 600s Retry-After backoff. The fix resolves the
REAL aggregator provider's pool (from the facade's ``last_aggregator_slot``) and rotates it —
without swapping the agent's virtual ``moa`` identity onto a real wire endpoint: the retry
re-resolves credentials per call via ``_slot_runtime`` → ``resolve_runtime_provider``.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent import moa_loop
from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.credential_pool import CredentialPool, PooledCredential
from agent.error_classifier import FailoverReason

AGG_BASE_URL = "https://api.anthropic.com"
AGG_MODEL = "claude-opus-4.6"


def _entry(i: int) -> PooledCredential:
    return PooledCredential(
        provider="anthropic", id=f"cred-{i}", label=f"acct-{i}", auth_type="api_key", priority=i,
        source="manual", access_token=f"sk-ant-{i}-1234567890", base_url=AGG_BASE_URL,
    )


def _pool() -> CredentialPool:
    return CredentialPool(provider="anthropic", entries=[_entry(0), _entry(1)])


def _MoAAgent(aggregator_provider: str = "anthropic"):
    """Agent bound to a MoA preset (virtual provider) with an anthropic aggregator."""
    agent = SimpleNamespace(
        log_prefix="", quiet_mode=True, provider="moa", requested_provider="moa",
        model="my-preset", base_url="moa://local", api_mode="chat_completions",
        api_key="moa-virtual-provider", _credential_pool=None,
        _fallback_chain=(), _fallback_index=0, _credential_pool_revert_id=None,
        _swap_credential=MagicMock(return_value=True),
        _is_entitlement_failure=lambda *a, **k: False,
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            last_aggregator_slot={"provider": aggregator_provider, "model": AGG_MODEL},
        ))),
    )
    return agent


def _seed_runtime_cache(monkeypatch, api_key: str) -> None:
    """Seed the ``_slot_runtime`` cache exactly as the aggregator call would have."""
    from hermes_constants import hermes_home_key

    monkeypatch.setattr(
        moa_loop, "_runtime_cache",
        {(hermes_home_key(), "anthropic", AGG_MODEL): (
            time.monotonic(), {"provider": "anthropic", "model": AGG_MODEL, "api_key": api_key,
                               "base_url": AGG_BASE_URL},
        )},
    )


def _clear_runtime_cache(monkeypatch) -> None:
    monkeypatch.setattr(moa_loop, "_runtime_cache", {})


def test_aggregator_429_benches_model_and_rotates(monkeypatch, tmp_path):
    pool = _pool()
    agent = _MoAAgent()
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pool)
    _seed_runtime_cache(monkeypatch, "sk-ant-0-1234567890")

    recovered, _ = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=True,
        classified_reason=FailoverReason.rate_limit,
    )
    assert recovered is True
    entries = pool.entries()
    # Anthropic per-model 429: a model cooldown, not a credential-wide bench — the
    # credential stays available for sibling models.
    assert entries[0].last_status is None
    assert AGG_MODEL in (entries[0].model_cooldowns or {})
    assert entries[1].last_status is None and not (entries[1].model_cooldowns or {})
    # The benched key's cached runtime resolution is dropped so the retry re-resolves.
    assert moa_loop.peek_slot_runtime_api_key("anthropic", AGG_MODEL) is None
    # The agent's virtual identity is NEVER swapped onto a real endpoint.
    agent._swap_credential.assert_not_called()
    assert agent.provider == "moa" and agent.base_url == "moa://local"


def test_aggregator_429_without_cached_hint_attributes_via_pool(monkeypatch, tmp_path):
    """No ``_slot_runtime`` entry (cold cache): attribution falls to the pool's own
    selection, which is the same entry the aggregator request would have picked."""
    pool = _pool()
    agent = _MoAAgent()
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pool)
    _clear_runtime_cache(monkeypatch)

    recovered, _ = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=True,
        classified_reason=FailoverReason.rate_limit,
    )
    assert recovered is True
    assert AGG_MODEL in (pool.entries()[0].model_cooldowns or {})


def test_aggregator_401_rotates_via_auth_branch(monkeypatch, tmp_path):
    pool = _pool()
    agent = _MoAAgent()
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pool)
    _seed_runtime_cache(monkeypatch, "sk-ant-0-1234567890")

    recovered, _ = recover_with_credential_pool(
        agent, status_code=401, has_retried_429=False,
        classified_reason=FailoverReason.auth,
    )
    assert recovered is True
    # api_key entries cannot refresh: the failed entry is benched, not a sibling.
    assert pool.entries()[0].last_status is not None
    assert pool.entries()[1].last_status is None
    agent._swap_credential.assert_not_called()
    assert moa_loop.peek_slot_runtime_api_key("anthropic", AGG_MODEL) is None


def test_aggregator_402_rotates_via_billing_branch(monkeypatch, tmp_path):
    pool = _pool()
    agent = _MoAAgent()
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pool)
    _seed_runtime_cache(monkeypatch, "sk-ant-0-1234567890")

    recovered, _ = recover_with_credential_pool(
        agent, status_code=402, has_retried_429=False,
        classified_reason=FailoverReason.billing,
    )
    assert recovered is True
    assert pool.entries()[0].last_status is not None
    assert pool.entries()[1].last_status is None
    agent._swap_credential.assert_not_called()


def test_empty_aggregator_pool_degrades_gracefully(monkeypatch, tmp_path):
    agent = _MoAAgent()
    empty_pool = CredentialPool(provider="anthropic", entries=[])
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: empty_pool)
    _clear_runtime_cache(monkeypatch)

    recovered, _ = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=True,
        classified_reason=FailoverReason.rate_limit,
    )
    assert recovered is False  # degrade to today's behaviour: surface the error


def test_no_aggregator_slot_falls_back_gracefully(monkeypatch, tmp_path):
    agent = _MoAAgent()
    agent.client.chat.completions.last_aggregator_slot = None
    monkeypatch.setattr("agent.credential_pool.load_pool", MagicMock())

    recovered, _ = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=True,
        classified_reason=FailoverReason.rate_limit,
    )
    assert recovered is False


def test_already_bound_aggregator_pool_does_not_double_rotate(monkeypatch, tmp_path):
    """Defensive (#132284 layer 7): if the agent somehow already carries the aggregator
    provider's pool, the MoA branch steps aside — the normal path's mismatch guard owns it."""
    bound = _pool()
    agent = _MoAAgent()
    agent._credential_pool = bound
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: bound)
    _seed_runtime_cache(monkeypatch, "sk-ant-0-1234567890")

    recovered, _ = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=True,
        classified_reason=FailoverReason.rate_limit,
    )
    # provider="moa" vs pool="anthropic": the mismatch guard declines to mutate.
    assert recovered is False
    assert all(e.last_status is None and not (e.model_cooldowns or {}) for e in bound.entries())
    agent._swap_credential.assert_not_called()


def test_non_moa_agent_unchanged(monkeypatch, tmp_path):
    """Control: a plain anthropic main-model agent keeps using agent._credential_pool."""
    plain = SimpleNamespace(
        provider="anthropic", model="claude-opus-4.6", base_url=AGG_BASE_URL,
        api_key="sk-ant-0-1234567890", _credential_pool=_pool(),
        _credential_pool_entry_id="cred-0", log_prefix="", quiet_mode=True,
        _credential_pool_revert_id=None,
        _swap_credential=MagicMock(return_value=True),
        _is_entitlement_failure=lambda *a, **k: False,
    )
    recovered, _ = recover_with_credential_pool(
        plain, status_code=402, has_retried_429=False,
        classified_reason=FailoverReason.billing,
    )
    assert recovered is True
    plain._swap_credential.assert_called_once()
    assert plain._swap_credential.call_args.args[0].id == "cred-1"
