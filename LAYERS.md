# Layers from issue #51083

**Title:** auxiliary vision sends unsupported temperature to OpenAI gpt-5.5

**Issue summary:** When `auxiliary.vision.provider=auto` and the main model is `openai-api/gpt-5.5`,
Hermes routes `vision_analyze` through OpenAI. The default `vision_temperature=0.1` is passed in the
call to `async_call_llm`, which forwards it to the OpenAI Chat Completions endpoint. The endpoint
rejects the request with HTTP 400:

> `Unsupported value: 'temperature' does not support 0.1 with this model. Only the default (1) value is supported.`

**Root cause:** `gpt-5.5` family models (across all providers: openai-api, openai-codex, openrouter,
custom) only accept the provider's default `temperature=1`. The codebase already has a
`_fixed_temperature_for_model()` directive that OMITS `temperature` for Kimi/Moonshot models and pins
a specific value for Arcee Trinity Large Thinking. However, **the gpt-5.5 family is not in that
list** — so `_build_call_kwargs()` passes the caller's `temperature=0.1` straight through to the
API, which 400s.

The existing fallback retry path (`Auxiliary vision (async): provider rejected temperature; retrying once without it`)
DOES catch this error and retry without temperature (line ~5423 in `agent/auxiliary_client.py`),
but it costs a wasted round-trip AND the issue's debug log shows the fallback chain also timed out,
so the retry path isn't enough — the user-facing error in `tools.vision_tools.py` line ~1011
(`Error analyzing image: Error code: 400 - ...`) is what they see.

**The fix must** pre-emptively OMIT `temperature` for gpt-5.5 family models so the call is correct
on the first attempt, eliminating the wasted 400 round-trip AND making the auxiliary call succeed
unconditionally for the supported case.

## Layers

1. **Direct OpenAI API route (`openai-api` provider)**
   → `tools/vision_tools.py:954` sets `vision_temperature = 0.1`
   → passed to `async_call_llm` → `_build_call_kwargs` → `_fixed_temperature_for_model` returns `None`
   → `temperature=0.1` reaches `client.chat.completions.create(...)`
   → OpenAI returns HTTP 400 "does not support 0.1, only the default (1) is supported"

2. **OpenRouter / custom OpenAI-compatible endpoints that proxy gpt-5.5**
   → Same code path, same outcome. The OpenAI-side contract applies regardless of proxy.

3. **Codex OAuth route (`openai-codex` provider, gpt-5.5)**
   → Already handled by `_is_codex_gpt55()` (compaction threshold override), but `_fixed_temperature_for_model`
     does NOT consult `_is_codex_gpt55` — only the broader family check should. Codex OAuth also
     rejects non-default temperature for gpt-5.5 reasoning.

4. **Dated / variant gpt-5.5 slugs (`gpt-5.5-pro`, `gpt-5.5-2026-04-23`, `openai/gpt-5.5` via OpenRouter)**
   → Same problem. Family detection must use prefix matching, not exact slug equality.

## Edge cases

- **OpenAI-prefixed model names from OpenRouter (`openai/gpt-5.5`)**
  → The bare-model extraction (`rsplit('/', 1)[-1]`) already strips the provider prefix.
  → Match `bare == "gpt-5.5"` OR `bare.startswith("gpt-5.5-")` OR `bare.startswith("gpt-5.5.")`.

- **Older gpt-5.x models (gpt-5.4, gpt-5, gpt-5-mini)**
  → Issue only mentions gpt-5.5, but gpt-5.5 reasoning family may share the contract with
    `o-series` (o1/o3/o4). Conservative scope: match only gpt-5.5 family for this fix to avoid
    breaking gpt-5.x models that DO accept custom temperature.

- **Caller explicitly sets `temperature=1` for gpt-5.5**
  → This is the workaround from the issue. After the fix, the directive returns OMIT_TEMPERATURE
    which means we strip the key entirely. The provider's default is 1, so the user-visible behavior
    is identical. No regression.

- **Other auxiliary tasks (compression, titles) using the same model**
  → They go through the same `_build_call_kwargs` → same fix. They were also broken when calling
    gpt-5.5 with custom temperature. The fix benefits them too — sibling call paths covered per
    AGENTS.md ("sibling call paths included — not just the one site the reporter hit").

- **Async retry path in vision_tools.py:980-1001 (image-size retry)**
  → Uses the same `call_kwargs` dict (which now has temperature stripped). Works correctly.

- **Test isolation: tests must call `_build_call_kwargs` / `_fixed_temperature_for_model` directly**
  → No live API calls. Unit-test the directive, not the network round-trip.
