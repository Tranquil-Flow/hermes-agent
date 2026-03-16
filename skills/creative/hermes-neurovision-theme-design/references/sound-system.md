# Sound System — Full Reference

Hermes NeuroVision has a lightweight, zero-external-dependency sound engine.
It runs on two tiers so it works everywhere.

---

## Sound Tiers

| Tier | Platform | What it plays |
|------|----------|---------------|
| 1    | Everywhere | curses.beep() (terminal bell) and curses.flash() (screen flash) |
| 2    | macOS only | afplay (audio files) and say (text-to-speech, voice: Whisper) |

On non-macOS: `type='file'` and `type='say'` are silently skipped.
`type='bell'` and `type='flash'` work on all platforms.

---

## SoundCue — The Atomic Unit

```python
from hermes_neurovision.sound import SoundCue

SoundCue(
    name="my-cue-name",  # unique slug used to reference this cue
    type="bell",         # 'bell' | 'flash' | 'say' | 'file'
    value="",            # text for 'say', file path for 'file', "" for bell/flash
    volume=0.5,          # 0.0–1.0 (multiplied by master sound_volume)
    priority=5,          # higher overrides lower; prevents pile-ups on rapid events
)
```

### Type Reference

| type    | What happens                          | Platform   |
|---------|---------------------------------------|------------|
| bell    | Terminal BEL character / curses.beep()| All        |
| flash   | Screen flash via curses.flash()       | All        |
| say     | macOS TTS via `say -v Whisper <text>` | macOS only |
| file    | Audio file via `afplay -v <vol> <path>`| macOS only |

### Cooldown

The SoundEngine enforces a 0.5s minimum interval between plays of the same
cue name. Rapid event bursts (e.g. multiple llm_chunk events) won't stack up.

---

## Attaching Sound to Your Plugin

Implement `sound_cues()` returning a dict mapping event_kind strings to SoundCues:

```python
from hermes_neurovision.sound import SoundCue

def sound_cues(self):
    return {
        # Simple: terminal bell on errors
        "error": SoundCue("error-bell", "bell", "", volume=1.0, priority=10),

        # Screen flash on tool calls (subtle, works everywhere)
        "tool_call": SoundCue("tool-flash", "flash", "", volume=0.4, priority=5),

        # macOS: TTS whisper for dramatic moments
        "agent_start": SoundCue("wake", "say", "online", volume=0.7, priority=8),

        # macOS: audio file for bloom events
        "memory_save": SoundCue("bloom-chime", "file", "/path/to/chime.wav",
                                volume=0.3, priority=6),
    }
```

The engine automatically gates all output through `state.tune.sound_enabled`
and scales volume by `state.tune.sound_volume`. You don't need to check these.

### Event Kinds You Can Map

```
agent_start / agent_end / session_resume   → lifecycle
tool_call / tool_complete / tool_error     → tool activity
mcp_tool_call                              → MCP tools
llm_start / llm_chunk / llm_end           → generation
memory_save / skill_create / checkpoint_created → knowledge
error / crash / threat_blocked            → danger
cron_tick / background_proc / subagent_started → background
approval_request / dangerous_cmd          → attention
browser_navigate / file_edit / git_commit → navigation
mcp_connected / mcp_disconnected          → connections
context_pressure / token_usage / cost_update → metrics
compression_started / compression_ended   → compression
```

---

## Designing an Immersive Sound Palette

Good sound design in NeuroVision is about **contrast and restraint**:

1. **One sound per tier of importance.** Errors get bells. Normal activity gets flash.
   Momentous events (memory_save, agent_start) get TTS or audio files if on macOS.

2. **Use priority to prevent cacophony.** If an error fires simultaneously with a
   tool_call, the error (priority 10) should win. Set priority thoughtfully.

3. **Flash is underrated.** A brief screen flash on every tool_call is subtle and
   satisfying — it makes the screen feel alive without being annoying.

4. **say -v Whisper is ghostly and beautiful.** Short phrases like "online",
   "remembered", "done" add immense character to agent_start / memory_save events.
   Keep phrases under 2 words. Avoid anything that feels chatty.

5. **Cooldown math.** The 0.5s cooldown means llm_chunk (fired many times per
   generation) will only beep once per 0.5s. This is intentional — don't try to
   work around it.

6. **No sound for routine events.** llm_chunk, tool_complete, and similar
   high-frequency events should use flash at most — never bell or file.
   Reserve audible cues for state transitions.

### Recommended Minimal Palette

```python
def sound_cues(self):
    return {
        "error":       SoundCue("error",    "bell",  "",       volume=1.0, priority=10),
        "tool_call":   SoundCue("tick",     "flash", "",       volume=0.3, priority=3),
        "agent_start": SoundCue("wake",     "say",   "online", volume=0.7, priority=8),
        "memory_save": SoundCue("remember", "say",   "noted",  volume=0.5, priority=6),
    }
```

### Rich Sound Palette (macOS, with audio files)

```python
def sound_cues(self):
    base = "/path/to/hermes-neurovision/soundtrack/output/"
    return {
        "agent_start":    SoundCue("wake",     "say",  "online",    volume=0.8, priority=9),
        "agent_end":      SoundCue("sleep",    "say",  "offline",   volume=0.6, priority=9),
        "error":          SoundCue("error",    "bell", "",          volume=1.0, priority=10),
        "crash":          SoundCue("crash",    "bell", "",          volume=1.0, priority=10),
        "tool_call":      SoundCue("tick",     "flash","",          volume=0.2, priority=2),
        "memory_save":    SoundCue("bloom",    "say",  "remembered",volume=0.5, priority=6),
        "skill_create":   SoundCue("skill",    "say",  "learned",   volume=0.5, priority=6),
        "git_commit":     SoundCue("commit",   "say",  "committed", volume=0.4, priority=5),
        "approval_request": SoundCue("alert", "bell", "",           volume=0.8, priority=8),
    }
```
