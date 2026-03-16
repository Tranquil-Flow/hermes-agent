---
name: hermes-neurovision-theme-design
description: Design and implement new animated themes and screens for Hermes NeuroVision — the ASCII terminal visualizer. Full ThemePlugin API, complete state/tune/slider reference, sound system, postfx, emergent layers, reactive system, design principles for breathtaking screens, and ASCII video integration notes.
triggers:
  - "add a theme to hermes-neurovision"
  - "new screen for neurovision"
  - "build a theme plugin"
  - "hermes-neurovision theme"
  - "add a visualizer screen"
  - "neurovision screen"
  - "create a visualization"
---

# Hermes NeuroVision — Complete Screen Design Guide

Repo: /workspace/Projects/hermes-neurovision
Plugins: hermes_neurovision/theme_plugins/ (drop a .py here, auto-imported)
Registration: register(MyPlugin()) at module bottom
Test: hermes-neurovision --theme my-name

---

## Architecture Overview

Two rendering layers:

  GRAPH LAYER   — nodes (points), edges (lines), particles, packets,
                  pulse effects, streaks. Driven by plugin hook methods.
                  Skip entirely by returning [] from build_nodes().

  DRAW LAYER    — two hooks, called in order:
                    draw_background() — called BEFORE nodes/edges (hybrid backdrop)
                    draw_extras()     — called AFTER everything (foreground/overlay)

For pure generative screens: return [] from build_nodes(), put all rendering
in draw_extras(). This is the most common pattern and where all the magic lives.

For hybrid screens: use draw_background() for the field, draw_extras() for
foreground FX. The graph layer (nodes, edges, packets) renders in between.
Only use hybrid when the mix is genuinely compelling.

---

## Minimal Plugin Shell

```python
from hermes_neurovision.plugin import ThemePlugin, ReactiveElement, Reaction
from hermes_neurovision.theme_plugins import register
import math, curses, random

class MyPlugin(ThemePlugin):
    name = "my-theme"   # unique slug — used in --theme CLI flag

    def __init__(self):
        self._grid = None       # persistent state
        self._rng = random.Random(42)  # own rng for per-frame stochastic work

    def build_nodes(self, w, h, cx, cy, count, rng):
        return []               # [] = skip graph layer entirely

    def draw_extras(self, stdscr, state, color_pairs):
        pass                    # all rendering here

register(MyPlugin())
```

---

## State Object — Complete Field Reference

state is ThemeState, passed to draw_extras and all draw hooks.

  state.width               int   — terminal columns
  state.height              int   — terminal rows
  state.frame               int   — monotonically increasing frame counter (starts 0)
  state.intensity_multiplier float — 0.0..1.5+, your primary reactive scalar
                                    Use to scale counts, brightness, speed, size.
                                    Default resting value: 0.6
                                    Spikes on events (tool calls, LLM chunks, errors)
                                    Smoothly lerps toward target each frame
  state.rng                 random.Random — seeded rng, use for initialization only
  state.quiet               bool  — True in live daemon quiet mode (suppress passive spawns)
  state.flash_until         float — unix timestamp; if > now, a flash is active
  state.flash_color_key     str   — color key for active flash ("warning" etc.)
  state.tune                TuneSettings or None — ALL slider/toggle values (see below)
  state.config              ThemeConfig — theme config (name, accent_char, palette, rates...)
  state._last_event_time    float — unix timestamp of last event (for idle detection)

  Emergent simulation objects (None if plugin didn't configure them):
  state.automaton           CellularAutomaton or None
  state.physarum            PhysarumSim or None
  state.neural_field        NeuralField or None
  state.wave_field          WaveField or None
  state.boids               BoidsFlock or None
  state.reaction_diffusion  ReactionDiffusion or None

  Live animation state (usually don't read these directly):
  state.nodes               List[(float, float)]
  state.edges               List[(int, int)]
  state.particles           List[Particle]
  state.packets             List[Packet]
  state.pulses              List[(x, y, radius)]
  state.streaks             List[Streak]

Pattern:
  w, h, f = state.width, state.height, state.frame
  intensity = state.intensity_multiplier
  tune = state.tune  # may be None — always guard: if tune else 1.0

---

## TuneSettings — Sliders and Toggles

state.tune is a TuneSettings instance (or None if not in interactive mode).
Plugins can READ these to make their screens respect user controls.
ALWAYS guard: val = state.tune.animation_speed if state.tune else 1.0

Sliders (float, user-adjustable):
  burst_scale        0.0..3.0  step 0.1  — scale burst event size
  packet_rate_mult   0.0..3.0  step 0.1  — multiplier on packet spawn rate
  pulse_rate_mult    0.0..3.0  step 0.1  — multiplier on pulse spawn rate
  particle_density   0.0..3.0  step 0.1  — multiplier on particle spawn chance
  event_sensitivity  0.0..3.0  step 0.1  — how hard events hit intensity
  animation_speed    0.1..5.0  step 0.1  — time multiplier for your animations
  warp_strength      0.0..3.0  step 0.1  — postfx warp effect strength
  void_intensity     0.0..3.0  step 0.1  — postfx void/erase effect strength
  force_strength     0.0..3.0  step 0.1  — postfx force field strength
  decay_rate         0.0..3.0  step 0.1  — postfx cell decay multiplier
  emergent_speed     0.0..3.0  step 0.1  — emergent simulation speed (0=paused)
  emergent_opacity   0.0..1.0  step 0.05 — emergent layer visibility
  sound_volume       0.0..1.0  step 0.05 — sound engine volume

Toggles (bool):
  show_packets    show_particles   show_pulses    show_stars
  show_background show_nodes       show_flash     show_spawn_node
  show_streaks    show_specials    show_overlays  color_shifts
  mask_enabled    symmetry_enabled reactive_elements  sound_enabled

Using sliders in draw_extras:
  spd = (state.tune.animation_speed if state.tune else 1.0)
  t = f * 0.04 * spd
  steps = int(600 * (0.4 + intensity) * (state.tune.particle_density if state.tune else 1.0))

---

## Color Pairs — Complete Reference

color_pairs is passed to draw_extras. Keys always present:
  "bright"   — primary highlight (use with curses.A_BOLD for maximum punch)
  "accent"   — secondary / warm complement
  "soft"     — dim/muted, for mid-range density
  "base"     — near-background (use with A_DIM for texture fill)
  "warning"  — alert/danger (hot spots, errors, spikes)

Usage pattern:
  import curses
  bright  = curses.color_pair(color_pairs["bright"]) | curses.A_BOLD
  accent  = curses.color_pair(color_pairs["accent"])
  soft    = curses.color_pair(color_pairs["soft"])
  dim     = curses.color_pair(color_pairs["base"]) | curses.A_DIM
  danger  = curses.color_pair(color_pairs["warning"]) | curses.A_BOLD

Brightness tiers for value-mapped rendering:
  if   v > 0.75: attr = bright
  elif v > 0.50: attr = accent
  elif v > 0.25: attr = soft
  else:          attr = dim

Intensity-reactive threshold (makes screen "respond" to events):
  thresh_hi = max(0.45, 0.75 - 0.15 * intensity)
  thresh_md = max(0.20, 0.50 - 0.10 * intensity)

Always wrap addstr/addch in try/except curses.error — no exceptions.

---

## Coordinate Conventions

Origin: top-left (row=0, col=0).
stdscr.addstr(y, x, ch, attr)   — row first, col second.

Terminal aspect: characters are ~2x taller than wide.
For isotropic circles: dist = math.sqrt(dx*dx/2.0 + dy*dy)
For radii: rx ≈ 2 * ry to draw circles that look round.

Safe write region: rows 1..h-2, cols 0..w-2.
Never use range(h) or range(w) — always range(1, h-1) and range(0, w-1).

---

## The Three Screen Archetypes

---

### ARCHETYPE 1: Full-Field Math (per-pixel formula)

No state between frames. Each frame recomputes every cell.
Best for: plasma, interference, vortex tunnel, Lissajous, aurora bands,
          polar spirals, kaleidoscope fields, sdf shapes, hypnotic waves.

```python
def draw_extras(self, stdscr, state, color_pairs):
    import curses, math
    w, h, f = state.width, state.height, state.frame
    intensity = state.intensity_multiplier
    spd = state.tune.animation_speed if state.tune else 1.0

    bright = curses.color_pair(color_pairs["bright"]) | curses.A_BOLD
    accent = curses.color_pair(color_pairs["accent"])
    soft   = curses.color_pair(color_pairs["soft"])
    dim    = curses.color_pair(color_pairs["base"]) | curses.A_DIM

    chars = " ·.:;+=*#%@"
    n = len(chars) - 1
    cx2, cy2 = w / 2.0, h / 2.0
    t = f * 0.04 * spd

    for y in range(1, h - 1):
        for x in range(0, w - 1):
            dx, dy = x - cx2, y - cy2
            dist = math.sqrt(dx * dx / 2.0 + dy * dy)
            angle = math.atan2(dy, dx)
            # Compose 2–4 waves for richness
            v = (math.sin(dist * 0.5 - t)
               + math.sin(x * 0.1 + t * 0.7)
               + math.sin(angle * 3 + t)) / 3.0
            v = (v + 1.0) / 2.0 * intensity
            v = max(0.0, min(1.0, v))
            ch = chars[int(v * n)]
            thresh_hi = max(0.45, 0.75 - 0.15 * intensity)
            if   v > thresh_hi:  attr = bright
            elif v > 0.50:       attr = accent
            elif v > 0.25:       attr = soft
            else:                attr = dim
            try:
                stdscr.addstr(y, x, ch, attr)
            except curses.error:
                pass
```

Formula ingredient library:
  math.sin(x * freq + f * speed)          — scrolling wave
  math.sin(dist * freq - f * speed)       — expanding rings
  math.atan2(dy, dx)                      — polar angle
  math.sin(angle * N + f * speed)         — N-armed spiral/vortex
  1.0 / (dist * scale + eps)              — tunnel zoom
  math.sin(dx*freq) * math.sin(dy*freq)   — grid interference
  Combine 3–5 waves, average for plasma: v = (w1+w2+w3+w4+w5)/5

SDF shapes (signed distance fields — draw crisp geometry):
  circle:    d = dist - radius                          (d < 0 = inside)
  ring:      d = abs(dist - radius) - thickness
  box:       d = max(abs(dx/2) - bw, abs(dy) - bh)
  hexagon:   use dot product with 3 hex axes
  v = max(0.0, 1.0 - d * sharpness)                    (soft edge)
  v = 1.0 if d < 0 else 0.0                            (hard edge)

---

### ARCHETYPE 2: Density Accumulator (point plotting + decay)

Strange attractors, IFS fractals, chaos games, orbital traces.
Float grid accumulates hits; decays each frame → glowing trails.

```python
def __init__(self):
    self._grid = None
    self._px = 0.1
    self._py = 0.1
    self._w = self._h = 0
    self._rng = __import__('random').Random(12345)

def _init(self, w, h):
    self._grid = [[0.0] * w for _ in range(h)]
    self._px, self._py = 0.1, 0.1
    self._w, self._h = w, h

def draw_extras(self, stdscr, state, color_pairs):
    import curses, math
    w, h, f = state.width, state.height, state.frame
    intensity = state.intensity_multiplier
    spd = state.tune.animation_speed if state.tune else 1.0

    if self._grid is None or (w, h) != (self._w, self._h):
        self._init(w, h)

    bright = curses.color_pair(color_pairs["bright"]) | curses.A_BOLD
    accent = curses.color_pair(color_pairs["accent"])
    soft   = curses.color_pair(color_pairs["soft"])
    dim    = curses.color_pair(color_pairs["base"]) | curses.A_DIM
    grid   = self._grid

    steps = int(800 * (0.4 + intensity) * spd)
    px, py = self._px, self._py
    cx2, cy2 = w / 2.0, h / 2.0
    scale_x = (w - 2) / 6.0
    scale_y = (h - 2) / 6.0

    # Clifford attractor example — swap params for different characters
    A, B, C, D = -1.4, 1.6, 1.0, 0.7
    for _ in range(steps):
        nx = math.sin(A * py) + C * math.cos(A * px)
        ny = math.sin(B * px) + D * math.cos(B * py)
        px, py = nx, ny
        sx = int(cx2 + px * scale_x)
        sy = int(cy2 + py * scale_y)
        if 1 <= sy < h - 1 and 0 <= sx < w - 1:
            grid[sy][sx] = min(grid[sy][sx] + 0.06, 1.0)

    self._px, self._py = px, py

    decay = 0.975 - 0.01 * intensity
    chars = " ·.,:;=+*#▒▓█"
    nc = len(chars)

    for y in range(1, h - 1):
        row = grid[y]
        for x in range(0, w - 1):
            v = row[x] * decay
            row[x] = v
            ch = chars[max(0, min(nc - 1, int(v * (nc - 1))))]
            if   v > 0.75: attr = bright
            elif v > 0.40: attr = accent
            elif v > 0.15: attr = soft
            else:          attr = dim
            try:
                stdscr.addstr(y, x, ch, attr)
            except curses.error:
                pass
```

Decay tuning:
  0.977  slow (trail lasts ~40 frames)
  0.92   medium (~12 frames)
  0.88   fast (~8 frames, good when camera rotates)
  intensity-linked: decay = 0.975 - 0.01 * intensity

3D projection (for Lorenz, Rössler, Thomas, Halvorsen):
```python
def _project(self, px, py, pz, w, h, az, el, sx, sy):
    rx = px * math.cos(az) - py * math.sin(az)
    ry = px * math.sin(az) + py * math.cos(az)
    rz = ry * math.sin(el) + pz * math.cos(el)
    return int(w/2 + rx * sx), int(h/2 - rz * sy)
```
Rotate az += 0.006/frame. Nod el: el = 0.4 * math.sin(f * 0.0033).
Fast decay (~0.88) when camera is moving to prevent smear.

IFS chaos game (Barnsley, fractal trees):
  Transform: (a, b, c, d, e, f_off, prob). Choose by cumulative prob.
  Apply: nx = a*px + b*py + e; ny = c*px + d*py + f_off
  Burn in: discard first 50 iterations before plotting.

---

### ARCHETYPE 3: Simulation Grid (cellular automaton / reaction-diffusion)

State grid, stepped N times per frame. CA rules, Gray-Scott RD, wave equation.

```python
def __init__(self):
    self._grid = None
    self._w = self._h = 0

def _init_grid(self, sw, sh, rng):
    self._grid = bytearray(sw * sh)
    for i in range(sw * sh):
        if rng.random() < 0.30:
            self._grid[i] = 1
    self._sw, self._sh = sw, sh

def _step(self, sw, sh):
    g = self._grid
    new_g = bytearray(sw * sh)
    for y in range(sh):
        for x in range(sw):
            idx = y * sw + x
            # 8-neighbor count (Conway Life example)
            n = sum(
                g[((y + dy) % sh) * sw + (x + dx) % sw]
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dx, dy) != (0, 0)
            )
            alive = g[idx]
            new_g[idx] = 1 if (alive and n in (2,3)) or (not alive and n==3) else 0
    self._grid = new_g

def draw_extras(self, stdscr, state, color_pairs):
    import curses
    w, h = state.width, state.height
    sw, sh = max(10, w // 2), max(5, h // 2)  # sim smaller for heavy rules

    if self._grid is None or w != self._w or h != self._h:
        self._init_grid(sw, sh, state.rng)
        self._w, self._h = w, h

    self._step(sw, sh)

    bright = curses.color_pair(color_pairs["bright"])
    dim    = curses.color_pair(color_pairs["base"]) | curses.A_DIM
    g = self._grid

    for y in range(1, h - 1):
        for x in range(0, w - 1):
            sx = max(0, min(x * sw // max(w, 1), sw - 1))
            sy = max(0, min((y-1) * sh // max(h-2, 1), sh - 1))
            val = g[sy * sw + sx]
            try:
                stdscr.addstr(y, x, "█" if val else " ", bright if val else dim)
            except curses.error:
                pass
```

Gray-Scott RD parameters (uvv = u*v*v; Du=0.16, Dv=0.08):
  new_u = u + Du*lap_u - uvv + F*(1-u)
  new_v = v + Dv*lap_v + uvv - (F+k)*v
  F=0.037, k=0.060  → coral/fingerprints
  F=0.035, k=0.065  → spots
  F=0.014, k=0.054  → spirals
  F=0.029, k=0.057  → labyrinthine

---

## Sound System — Deep Reference

The sound engine is zero-dependency — works everywhere, richer on macOS.

### SoundCue — the atomic unit

```python
from hermes_neurovision.sound import SoundCue, SoundEngine

SoundCue(
    name="my_sound",        # unique name for cooldown tracking
    type="bell",            # 'bell' | 'flash' | 'say' | 'file'
    value="",               # text for 'say', file path for 'file'
    volume=0.5,             # 0.0-1.0 (file only, macOS only)
    priority=0,             # higher = override lower priority
)
```

Sound types:
  'bell'   — curses.beep() → terminal bell. Universal. Falls back to \a.
             Use for: agent_start, error, dangerous_cmd
  'flash'  — curses.flash() → screen flash. Universal.
             Use for: burst events, attention spikes
  'say'    — macOS only: afplay 'say -v Whisper <text>' fire-and-forget TTS.
             Use for: memorable moments, skill_create, milestone events
             voice = 'Whisper' (breathy/atmospheric, perfect for our aesthetic)
  'file'   — macOS only: afplay -v <vol> <path> fire-and-forget.
             Use for: custom WAV/AIFF/MP3 sound effects in soundtrack/

Cooldown: SoundEngine enforces 0.5s minimum between same sound name.
Engine is accessible via the Reaction.sound field — attach a sound cue
name to any Reaction and it fires automatically through the reactive engine.

### Attaching Sound to Reactions

```python
def react(self, event_kind, data):
    if event_kind == "agent_start":
        return Reaction(
            element=ReactiveElement.PULSE,
            intensity=0.9,
            origin=(0.5, 0.5),
            color_key="bright",
            duration=2.5,
            sound="wake_bell",          # ← plays this cue when reaction fires
        )
```

### Designing an Immersive Sound Palette

For each plugin, define a small palette of 3–5 sounds mapping event severity:

  LOW   (ambient): 'bell' on tool_call, llm_chunk — subtle, rhythmic
  MED   (events):  'flash' on memory_save, tool_complete — satisfying
  HIGH  (drama):   'bell' + draw flash overlay on error/crash
  RARE  (magical): 'say' on skill_create ("remembered"), checkpoint_created

Avoid sound fatigue: map only high-signal events. llm_chunk fires constantly —
give it only a 'flash' or skip it. Batch similar events with cooldowns.

macOS soundtrack files live in /workspace/Projects/hermes-neurovision/soundtrack/
  track1_emergence, track2_signal_fire, track3_neural_dawn, track4_void_signal

To play soundtrack clips from a theme:
  cue = SoundCue("emergence", "file", "/path/to/soundtrack/clip.wav", volume=0.3)

---

## React System — Full Event Reference

```python
from hermes_neurovision.plugin import ThemePlugin, ReactiveElement, Reaction

def react(self, event_kind: str, data) -> "Reaction | None":
    # return a Reaction or None
    pass
```

IMPORTANT: react() fires ONLY in live/daemon mode with real agent events.
In gallery mode, a synthetic pump fires VisualTrigger strings (not react calls):
  "wake", "ripple", "burst", "pulse", "packet", "cascade", "cool_down"
These trigger draw_overlay_effect() and special_effects(), not react().

Full event_kind → ReactiveElement mapping:

  PULSE (radial burst, one-shot dramatic):
    "agent_start", "agent_end", "session_resume"

  RIPPLE (concentric rings from point, one-shot):
    "tool_call", "tool_complete", "tool_error", "mcp_tool_call"

  STREAM (flowing particles in direction, sustained):
    "llm_start", "llm_chunk", "llm_end"

  BLOOM (organic growth, expands + holds + fades):
    "memory_save", "skill_create", "checkpoint_created"

  SHATTER (explosion of fragments scattering):
    "error", "crash", "threat_blocked"

  ORBIT (persistent rotating elements):
    "cron_tick", "background_proc", "subagent_started"

  GAUGE (fills/drains bar or arc, color at thresholds):
    "context_pressure", "token_usage", "cost_update"

  SPARK (bright flash + lingering afterglow):
    "approval_request", "dangerous_cmd"

  WAVE (horizontal sweep across screen):
    "compression_started", "compression_ended", "checkpoint_rollback"

  GLYPH (symbol/sigil that appears, slowly morphs):
    "personality_change", "reasoning_change"

  TRAIL (path/line tracing movement):
    "browser_navigate", "file_edit", "git_commit"

  CONSTELLATION (dots connecting/disconnecting):
    "mcp_connected", "mcp_disconnected", "provider_health"

### Reaction dataclass fields:

```python
Reaction(
    element=ReactiveElement.BLOOM,
    intensity=1.0,              # 0.0-1.0, how dramatic
    origin=(0.5, 0.5),          # where on screen (0-1 normalized coords)
    color_key="bright",         # "bright" | "accent" | "soft" | "base" | "warning"
    duration=2.5,               # seconds to run
    data={},                    # element-specific extra params (e.g. dx for STREAM)
    sound="my_cue_name",        # optional: fires SoundCue by name
)
```

### Minimal react() implementation (cover 6+ events):

```python
def react(self, event_kind, data):
    import random
    r = random.random
    if event_kind == "agent_start":
        return Reaction(ReactiveElement.PULSE, 0.9, (0.5, 0.5), "bright", 2.5, sound="wake")
    if event_kind in ("llm_start", "llm_chunk"):
        return Reaction(ReactiveElement.STREAM, 0.5, (0.0, r()), "accent", 0.8,
                        data={"dx": 1})
    if event_kind in ("tool_call", "mcp_tool_call"):
        return Reaction(ReactiveElement.RIPPLE, 0.7, (r(), r()), "accent", 1.8)
    if event_kind in ("memory_save", "skill_create"):
        return Reaction(ReactiveElement.BLOOM, 1.0, (0.5, 0.5), "bright", 2.5)
    if event_kind in ("error", "crash", "threat_blocked"):
        return Reaction(ReactiveElement.SHATTER, 1.0, (0.5, 0.5), "warning", 2.0,
                        sound="alert")
    if event_kind == "git_commit":
        return Reaction(ReactiveElement.TRAIL, 0.7, (0.0, 0.5), "soft", 1.5)
    if event_kind == "cron_tick":
        return Reaction(ReactiveElement.ORBIT, 0.5, (0.5, 0.5), "soft", 2.0)
    return None
```

---

## Emergent Layer Systems

One or more can run per plugin. They run as background/mid/foreground overlays.
Configure via config methods; the engine instantiates and runs them.

```python
def automaton_config(self):
    # Rule options: 'brians_brain'
    return {"rule": "brians_brain", "density": 0.08, "update_interval": 2}

def physarum_config(self):
    # Slime mold network formation — beautiful organic tubes
    return {"n_agents": 150, "sensor_dist": 4.0, "sensor_angle": 0.785,
            "deposit": 1.0, "decay": 0.95}

def neural_field_config(self):
    # Spreading activation across a grid — looks like firing neurons
    return {"threshold": 2, "fire_duration": 2, "refractory": 5}

def wave_config(self):
    # Wave propagation — ripples, interference, standing waves
    return {"speed": 0.3, "damping": 0.98}

def boids_config(self):
    # Flocking — murmurations, swarms, schools of fish
    return {"n_boids": 40, "sep_dist": 3.0, "align_dist": 8.0,
            "cohesion_dist": 12.0, "max_speed": 1.5}

def reaction_diffusion_config(self):
    return {"feed": 0.037, "kill": 0.060, "update_interval": 2}

def emergent_layer(self):
    return "background"  # "background" | "midground" | "foreground"
```

Emergent opacity/speed are controlled by state.tune.emergent_opacity and
state.tune.emergent_speed — the engine respects these automatically.

---

## PostFX — Full Reference

PostFX operates on the FrameBuffer AFTER all drawing. Each is opt-in via plugin methods.
ALL are modulated by state.tune sliders when enabled.

### Glow (glow_radius)
```python
def glow_radius(self): return 1  # 0=off, 1=subtle, 2=medium, 3=strong
```
Bright non-space cells bleed color as dim '·' dots into empty neighbors.
Radius 1 = 1-cell halo. Makes ASCII visuals feel luminous.

### Echo / Afterimage (echo_decay)
```python
def echo_decay(self): return 3  # 0=off; N=frames of ghost afterimage
```
Empty cells get filled with dimmed content from N frames ago.
Creates trails, ghost smears. Works beautifully with attractors.

### Cell Decay Sequence (decay_sequence)
```python
def decay_sequence(self): return "█▓▒░·. "  # cells age through this
```
Each non-space cell ages through the sequence over time.
Every 3 frames: char advances one step toward space. Creates beautiful
dissolution, crystallization, and erosion effects.

### Symmetry (symmetry)
```python
def symmetry(self): return "mirror_xy"  # or "mirror_x", "mirror_y", "rotate_4"
```
mirror_x: left half copies to right (bilateral)
mirror_y: top half copies to bottom
mirror_xy: all four quadrants (mandala-like)
rotate_4: four-fold rotational (kaleidoscope)
Can be toggled off by user with symmetry_enabled = False.

### Depth Layers (depth_layers)
```python
def depth_layers(self): return 3  # 1=flat; 2-4 = parallax depth
```
Simulates foreground/background parallax.

### Warp Field (warp_field)
```python
def warp_field(self, x, y, w, h, frame, strength):
    import math
    dx = math.sin(y * 0.1 + frame * 0.04) * strength * 2
    dy = math.cos(x * 0.1 + frame * 0.03) * strength
    return int(x + dx), int(y + dy)
```
Displacement mapping. Warps the rendered buffer.
strength comes from state.tune.warp_strength.

### Void Points (void_points)
```python
def void_points(self, w, h, frame, intensity):
    import math
    cx, cy = w // 2, h // 2
    r = int(5 + 3 * math.sin(frame * 0.05))
    return [(int(cx + math.cos(math.radians(a)) * r * 2),
             int(cy + math.sin(math.radians(a)) * r))
            for a in range(0, 360, 10)]
```
Erases cells at given positions. Good for carving holes, pulsing voids.

### Force Field (force_points)
```python
def force_points(self, w, h, frame, strength):
    # (x, y, force_strength, force_type) — 'radial' or 'vortex'
    return [(w//2, h//2, strength * 2.0, 'vortex')]
```
Physically displaces non-space cells.

### Render Mask (render_mask)
```python
def render_mask(self, w, h, frame, intensity):
    import math
    mask = [[True]*w for _ in range(h)]
    cx, cy = w/2, h/2
    r = min(w, h) * 0.4 * intensity
    for y in range(h):
        for x in range(w):
            if math.sqrt((x-cx)**2/2 + (y-cy)**2) > r:
                mask[y][x] = False
    return mask
```
Boolean stencil. Use for iris reveals, animated wipes, text cutouts.

---

## Additional Plugin Hooks

### draw_background() — Hybrid Backdrop
Called BEFORE nodes/edges. Only use when you want a field behind graph layer.
Recommended rarely — only when the combination is genuinely compelling.

### ambient_tick() — Idle Animation
```python
def ambient_tick(self, stdscr, state, color_pairs, idle_seconds):
    """idle_seconds grows while agent is quiet. Transition to calm state."""
    calm = min(1.0, idle_seconds / 10.0)
```

### intensity_curve() — Shape How Intensity Feels
```python
def intensity_curve(self, raw):
    import math
    return math.pow(raw, 1.5)   # exponential: dramatic peaks
```

### palette_shift() — Reactive Color Changes
```python
def palette_shift(self, trigger_effect, intensity, base_palette):
    import curses
    if trigger_effect == "burst":
        return (curses.COLOR_RED, curses.COLOR_YELLOW, curses.COLOR_WHITE, curses.COLOR_MAGENTA)
    return None
```

### special_effects() + draw_special() — Named One-Shot FX
```python
from hermes_neurovision.plugin import SpecialEffect

def special_effects(self):
    return [SpecialEffect("supernova", ["burst", "pulse"], min_intensity=0.7,
                          cooldown=8.0, duration=3.0)]

def draw_special(self, stdscr, state, color_pairs, special_name, progress, intensity):
    """progress: 0.0 → 1.0 over the effect's duration."""
    pass
```

### draw_overlay_effect() — Gallery-Mode Event Visuals
```python
def draw_overlay_effect(self, stdscr, state, color_pairs, trigger_effect, intensity, progress):
    """Gallery synthetic events: wake/ripple/burst/pulse/packet/cascade/cool_down."""
    pass
```

### Animated Particles (Particle.frames)
```python
from hermes_neurovision.scene import Particle

def spawn_particle(self, w, h, nodes, rng):
    return Particle(x=rng.uniform(0,w), y=rng.uniform(1,h-2),
                    vx=rng.uniform(-0.2,0.2), vy=rng.uniform(-0.1,0.1),
                    life=20, max_life=20, char="*",
                    frames=["·", ":", "*", "✦", "*", ":", "·"])
```
frames enables per-frame character animation on each particle.

---

## Existing Theme Names (DO NOT REUSE)

Originals: black-hole neural-sky storm-core moonwire rootsong stormglass spiral-galaxy
Nature: deep-abyss storm-sea dark-forest mountain-stars beach-lighthouse
Cosmic: aurora-borealis nebula-nursery binary-rain wormhole
Industrial: liquid-metal factory-floor pipe-hell oil-slick
Whimsical: campfire aquarium circuit-board lava-lamp firefly-field
Hostile: noxious-fumes maze-runner
Exotic: neon-rain volcanic crystal-cave spider-web snow-globe
Mechanical: clockwork coral-reef ant-colony satellite-orbit starfall
Cosmic New: quasar supernova sol terra binary-star
ASCII Fields: synaptic-plasma oracle cellular-cortex reaction-field stellar-weave
             life-colony aurora-bands waveform-scope lissajous-mind pulse-matrix
Extreme: fractal-engine n-body standing-waves
Experimental: clifford-attractor barnsley-fern flow-field
Emergent Showcase: mycelium-network swarm-mind neural-cascade tide-pool turing-garden
Hybrid: plasma-grid deep-signal
Emergent V2: dna-helix pendulum-waves kaleidoscope electric-storm coral-growth
Advanced: dna-strand pendulum-array mandala-scope ghost-echo magnetic-field
Strange Attractors: lorenz-butterfly rossler-ribbon halvorsen-star aizawa-torus thomas-labyrinth
Spectacular: hypnotic-tunnel plasma-rainbow fractal-zoom particle-vortex chladni-sand
Quantum: quantum-foam
New Original: ascii-rain sand-automaton ascii-rorschach wireframe-cube hypercube-fold
Also: lorenz-attractor duffing electric-mycelium cathedral-circuit hybrid
      julia-morph harmonograph fourier-epicycles spirograph rorschach sand-cascade
      dla-crystal fourier-epicycles harmonograph magnetic-field

---

## File Placement and Registration

hermes_neurovision/theme_plugins/__init__.py auto-imports all .py files.
Just create a new .py file — it will be discovered automatically.
register(MyPlugin()) at module level is required.

To test immediately:
  cd /workspace/Projects/hermes-neurovision
  source .venv/bin/activate
  hermes-neurovision --theme my-name

To add to the official THEMES tuple:
  Edit hermes_neurovision/themes.py — add to THEMES tuple and add a
  ThemeConfig entry in build_theme_config().

---

## ThemeConfig Reference

```python
ThemeConfig(
    name=name,
    title="Display Name",
    accent_char="·",
    background_density=0.025,  # 0.01=sparse → 0.05=dense stars
    star_drift=0.04,
    node_jitter=0.20,
    packet_rate=0.28,
    packet_speed=(0.04, 0.08),
    pulse_rate=0.09,
    edge_bias=0.50,
    cluster_count=3,
    palette=(
        curses.COLOR_CYAN, curses.COLOR_MAGENTA,
        curses.COLOR_WHITE, curses.COLOR_BLUE,
    ),
)
```

---

## Intensity Coupling Patterns — The Full Toolkit

intensity = state.intensity_multiplier  (default resting ~0.6, spikes to 1.5+)

Particle count:    target_n = int(60 + 140 * intensity)
Iteration steps:   steps = int(800 * (0.4 + intensity))
Decay:             decay = 0.975 - 0.01 * intensity
Speed:             t = f * (0.02 + 0.01 * intensity)
Brightness thresh: thresh = max(0.45, 0.75 - 0.15 * intensity)
Band thickness:    thick = 1.5 + 0.5 * intensity
Step speed:        spd = state.tune.animation_speed if state.tune else 1.0
Color saturation:  go bright+bold when intensity > 1.0

Calm vs active state:
  idle = (time.time() - state._last_event_time)
  calm_factor = min(1.0, idle / 10.0)  # 0=active, 1=fully calm after 10s

---

## Graph Layer Hooks (optional)

Override only if you want nodes + edges in your screen.

  build_nodes(w, h, cx, cy, count, rng)       → list[(x,y)] or [] to disable
  build_edges_extra(nodes, edges_set)          → add (i,j) tuples for extra edges
  step_nodes(nodes, frame, w, h)              → mutate nodes in-place per frame
  step_star(star, frame, w, h, rng)           → bool (True=handled, False=default)
  step_star_post(star, frame, w, h, rng)      → post-drift tweak
  spawn_particle(w, h, nodes, rng)            → Particle or None
  particle_base_chance()                      → float (default 0.028)
  particle_life_range()                       → (min, max) int tuple
  node_glyph(idx, intensity, total)           → str
  node_color_key(idx, intensity, total)       → "bright"|"accent"|"soft"|"base"
  edge_glyph(dx, dy)                         → str or None
  edge_color_key(step, idx_a, frame)          → color key str
  packet_color_key()                         → color key str
  packet_budget()                            → int (max concurrent packets)
  particle_color_key(age_ratio)              → color key str
  pulse_style()                              → "ring"|"rays"|"spoked"|"ripple"|"cloud"|"diamond"
  pulse_params()                             → (growth_rate, limit_ratio)
  pulse_color_key()                          → color key str
  streak_color_key()                         → color key str

---

## ASCII Video Integration Notes

When designing screens that should also export as ASCII video (demo_video.py):

1. Every frame of draw_extras IS a video frame.
2. Design with t = frame * 0.04, let animation_speed slider tune it.
3. Terminal color is limited — video exporter applies its own color remapping.
4. High-density chars (█▓▒░) read better than sparse (·:.) in video output.
5. Pure generative screens (Archetype 1) export best to video.
6. For dedicated ASCII video production, see the ascii-video skill.

---

## Design Philosophy: Breathtaking Screens

The standard is: unique, interesting, fun, vibrant, dynamic, breathtaking,
mesmerising. Every screen should feel like nothing else in the gallery.

### Make it MOVE in surprising ways
- Counter-rotating dual layers at different speeds
- Waves that collide and interfere (not just travel)
- Patterns that breathe — expand and contract with time
- Camera rotation on attractors (never static viewpoint)
- Emergence: simple rules → complex beauty (physarum, RD, flocking)
- Oscillations that drift: sin(f*0.03) * sin(f*0.0073) — beating frequencies

### Make it RESPOND to the agent
- Intensity scales at least 3 things: count, speed, brightness threshold
- react() should feel thematically appropriate — SHATTER for crashes is right
- ambient_tick() creates a calm→active transition narrative
- Sound design should match the visual metaphor

### Character palette selection
- Dense: ██▓▒░▤▥▦ (fills, solid fields)
- Organic: ·.:;oO (soft, plasma/nebula)
- Technical: =+-|/ (circuits, structured)
- Crystal: ◆◇◈◉●○◦ (gems, high contrast)
- Cosmic: ✦✧★☆⊹⊕⊗ (stars, signals)
- Flow: ~≈∿∾ (water, smoke, energy)
- Runes/symbols: ⟁⟐⟟⟠⟡⊛⊞ (mysterious, glyphs)
- Braille: use for ultra-dense pixel simulation (⠿⣿)
- Unicode blocks: ▀▄▌▐░▒▓█ (half-block art, depth)

### The PostFX stack as artistic tool
- glow_radius=1 on almost any screen — makes everything feel luminous
- echo_decay=3-5 on attractors — gorgeous trail afterimages
- decay_sequence="█▓▒░·. " + slow sim = crystalline erosion
- symmetry="mirror_xy" on plasma = instant mandala
- warp_field + plasma = breathing, living field
- force_points "vortex" at center = everything spirals

### Hybrid screens (use rarely, only when genuinely interesting)
- draw_background() for the generative field
- Keep graph layer minimal (few nodes) as focal points
- draw_extras() for foreground glyphs / reactive overlays

### The 5-layer stack mental model
  1. Background: emergent system / full-field math (deepest)
  2. Midground: density accumulator / particle cloud
  3. Graph layer: nodes + edges (only if truly needed)
  4. Reactive elements: events overlay (automated)
  5. Foreground: draw_extras() HUD / glyphs / extra FX

Most breathtaking screens use layers 1 + 4 + 5.
Attractors shine with just layer 2 + echo_decay postfx.

---

## Critical Pitfalls

1. ALWAYS wrap addstr/addch in try/except curses.error.
2. Grid init guard: if self._grid is None or (w, h) != (self._w, self._h):
3. Terminal aspect: dist = sqrt(dx*dx/2 + dy*dy). Circles: rx ≈ 2*ry.
4. Row/col bounds: range(1, h-1) and range(0, w-1). Never range(h)/range(w).
5. Heavy sim rules: use sw=w//2, sh=h//2 and upscale on render.
6. state.rng — init only. Per-frame: use self._rng = random.Random() in __init__.
7. state.color_theme does NOT exist. Use color_pairs dict argument.
8. Density grid + rotating camera = smear. Set decay~0.88 when projecting.
9. IFS chaos game: burn in first 50 iterations before plotting.
10. build_nodes must return a list ([] for nodeless). None = AttributeError.
11. Duplicate method definitions: last one wins. Don't define twice.
12. state.tune may be None. Always guard: spd = state.tune.animation_speed if state.tune else 1.0

---

## Quick New Screen Checklist

[ ] Pick archetype: full-field math / density accumulator / simulation grid
[ ] class MyPlugin(ThemePlugin): name = "my-name"  (unique slug)
[ ] Check existing names list above — pick something fresh
[ ] def build_nodes: return []  (unless you want graph layer)
[ ] def __init__: grid=None, _rng=random.Random(), persistent state
[ ] def draw_extras: resize guard, read intensity + tune, render loop
[ ] Wrap ALL addstr in try/except curses.error
[ ] Aspect ratio compensation for circles (dx/2, rx≈2*ry)
[ ] Safe bounds: range(1, h-1) and range(0, w-1)
[ ] Add glow_radius() returning 1 or 2
[ ] Add react() with at least 6 event handlers
[ ] Add ambient_tick() for idle breathing / slow morphing
[ ] Add ONE emergent config appropriate to theme (optional but elevates)
[ ] intensity_curve() if theme benefits from non-linear response
[ ] register(MyPlugin()) at module bottom
[ ] Add ThemeConfig in themes.py (if adding to gallery)
[ ] Test: hermes-neurovision --theme my-name
[ ] Add sound: at least bell on agent_start, shatter on error
