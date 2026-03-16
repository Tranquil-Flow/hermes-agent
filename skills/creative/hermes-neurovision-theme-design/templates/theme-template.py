"""
HERMES NEUROVISION — THEME TEMPLATE v2.0
=========================================
Copy this file, give it a unique name, drop it in:
  hermes_neurovision/theme_plugins/

Replace all TODO markers. The plugin is auto-discovered.
Test: hermes-neurovision --theme my-theme-slug
"""
import math
import random
import curses
from hermes_neurovision.plugin import ThemePlugin, ReactiveElement, Reaction, SpecialEffect
from hermes_neurovision.theme_plugins import register


class MyTheme(ThemePlugin):
    # ── Identity ─────────────────────────────────────────────────────────────
    name = "my-theme"   # REQUIRED: unique slug, used with --theme CLI flag
                        # Check existing names in skill — do not reuse

    # ── Constructor — init all persistent state here ─────────────────────────
    def __init__(self):
        self._grid = None          # 2D grid for density accumulator / simulation
        self._px = 0.1             # attractor x position
        self._py = 0.1             # attractor y position
        self._w = self._h = 0      # last known terminal size (resize guard)
        # IMPORTANT: use your own rng for per-frame stochastic work
        # state.rng is for initialization only
        self._rng = random.Random(42)

    # ── Build nodes ──────────────────────────────────────────────────────────
    def build_nodes(self, w, h, cx, cy, count, rng):
        # Return [] to skip the graph layer (most pure-generative screens)
        # Return a list of (x, y) float tuples to enable nodes/edges/packets
        return []

    # ── Main drawing — called every frame after all other layers ─────────────
    def draw_extras(self, stdscr, state, color_pairs):
        w, h, f = state.width, state.height, state.frame
        intensity = state.intensity_multiplier   # resting ~0.6, spikes to 1.5+
        spd = state.tune.animation_speed if state.tune else 1.0

        # Color setup
        bright = curses.color_pair(color_pairs["bright"]) | curses.A_BOLD
        accent = curses.color_pair(color_pairs["accent"])
        soft   = curses.color_pair(color_pairs["soft"])
        dim    = curses.color_pair(color_pairs["base"]) | curses.A_DIM
        danger = curses.color_pair(color_pairs["warning"]) | curses.A_BOLD

        # Resize guard (REQUIRED if you maintain persistent state)
        if self._grid is None or (w, h) != (self._w, self._h):
            self._init(w, h)

        # ── TODO: Replace with your rendering logic ───────────────────────────
        #
        # ARCHETYPE 1 — Full-field math (per-pixel formula):
        #   t = f * 0.04 * spd
        #   for y in range(1, h-1):
        #       for x in range(0, w-1):
        #           dx, dy = x - w/2, y - h/2
        #           dist = math.sqrt(dx*dx/2.0 + dy*dy)   # compensate aspect ratio
        #           v = (math.sin(dist*0.5 - t) + 1.0) / 2.0 * intensity
        #           v = max(0.0, min(1.0, v))
        #           ch = " ·.:+=*#@"[int(v * 8)]
        #           thresh = max(0.45, 0.75 - 0.15 * intensity)
        #           attr = bright if v > thresh else accent if v > 0.5 else soft if v > 0.25 else dim
        #           try: stdscr.addstr(y, x, ch, attr)
        #           except curses.error: pass
        #
        # ARCHETYPE 2 — Density accumulator (attractor/fractal):
        #   steps = int(800 * (0.4 + intensity) * spd)
        #   px, py = self._px, self._py
        #   for _ in range(steps):
        #       ... iterate attractor ...
        #       if in bounds: self._grid[sy][sx] += 0.06
        #   self._px, self._py = px, py
        #   decay = 0.975 - 0.01 * intensity
        #   for y/x: v = grid[y][x] * decay; grid[y][x] = v
        #            ch = chars[int(v * (nc-1))]
        #            try: stdscr.addstr(y, x, ch, ...)
        #            except curses.error: pass
        #
        # ARCHETYPE 3 — Simulation grid (CA / RD):
        #   self._step(w//2, h//2)
        #   render grid upscaled to terminal
        # ──────────────────────────────────────────────────────────────────────

        pass  # REPLACE THIS

    def _init(self, w, h):
        """Initialize or reinitialize persistent state for new terminal size."""
        self._grid = [[0.0] * w for _ in range(h)]
        self._px, self._py = 0.1, 0.1
        self._w, self._h = w, h

    # ── Optional: Hybrid backdrop (renders BEFORE graph layer) ───────────────
    # def draw_background(self, stdscr, state, color_pairs):
    #     """Only use if you want a field BEHIND nodes/edges (hybrid screen)."""
    #     pass

    # ── PostFX — opt-in, all return defaults that disable them ───────────────
    def glow_radius(self):
        return 1   # 0=off, 1=subtle halo, 2=medium, 3=strong color bleed

    def echo_decay(self):
        return 0   # 0=off, N=frames of ghost afterimage (great for attractors)

    def decay_sequence(self):
        return None  # or "█▓▒░·. " — cells age through this sequence

    def symmetry(self):
        return None  # "mirror_x" | "mirror_y" | "mirror_xy" | "rotate_4" | None

    def depth_layers(self):
        return 1   # 1=flat, 2-4=parallax depth simulation

    # def warp_field(self, x, y, w, h, frame, strength):
    #     """Displacement mapping — return (src_x, src_y) to sample."""
    #     return (x, y)  # identity (no warp)

    # def void_points(self, w, h, frame, intensity):
    #     """Return list of (x, y) positions to erase."""
    #     return []

    # def force_points(self, w, h, frame, strength):
    #     """Return list of (x, y, force_strength, force_type) — 'radial' or 'vortex'."""
    #     return []

    # def render_mask(self, w, h, frame, intensity):
    #     """Return 2D list[list[bool]] stencil, or None."""
    #     return None

    # ── Emergent systems — comment in ONE that fits your theme ───────────────
    # def physarum_config(self):
    #     return {"n_agents": 150, "sensor_dist": 4.0, "sensor_angle": 0.785,
    #             "deposit": 1.0, "decay": 0.95}

    # def boids_config(self):
    #     return {"n_boids": 40, "sep_dist": 3.0, "align_dist": 8.0,
    #             "cohesion_dist": 12.0, "max_speed": 1.5}

    # def reaction_diffusion_config(self):
    #     return {"feed": 0.037, "kill": 0.060, "update_interval": 2}

    # def automaton_config(self):
    #     return {"rule": "brians_brain", "density": 0.08, "update_interval": 2}

    # def neural_field_config(self):
    #     return {"threshold": 2, "fire_duration": 2, "refractory": 5}

    # def wave_config(self):
    #     return {"speed": 0.3, "damping": 0.98}

    # def emergent_layer(self):
    #     return "background"  # "background" | "midground" | "foreground"

    # ── Reactive element system — fires in live/daemon mode ──────────────────
    def react(self, event_kind: str, data) -> "Reaction | None":
        r = self._rng.random
        if event_kind == "agent_start":
            return Reaction(ReactiveElement.PULSE, 0.9, (0.5, 0.5), "bright", 2.5,
                            sound="wake")
        if event_kind in ("llm_start", "llm_chunk"):
            return Reaction(ReactiveElement.STREAM, 0.5, (0.0, r()), "accent", 0.8,
                            data={"dx": 1})
        if event_kind in ("tool_call", "tool_complete", "mcp_tool_call"):
            return Reaction(ReactiveElement.RIPPLE, 0.7, (r(), r()), "accent", 1.8)
        if event_kind in ("memory_save", "skill_create", "checkpoint_created"):
            return Reaction(ReactiveElement.BLOOM, 1.0, (0.5, 0.5), "bright", 2.5)
        if event_kind in ("error", "crash", "threat_blocked"):
            return Reaction(ReactiveElement.SHATTER, 1.0, (0.5, 0.5), "warning", 2.0,
                            sound="alert")
        if event_kind in ("approval_request", "dangerous_cmd"):
            return Reaction(ReactiveElement.SPARK, 1.0, (0.5, 0.5), "warning", 1.5)
        if event_kind == "git_commit":
            return Reaction(ReactiveElement.TRAIL, 0.7, (0.0, 0.5), "soft", 1.5)
        if event_kind == "cron_tick":
            return Reaction(ReactiveElement.ORBIT, 0.5, (0.5, 0.5), "soft", 2.0)
        return None

    # ── Idle / ambient animation ─────────────────────────────────────────────
    # def ambient_tick(self, stdscr, state, color_pairs, idle_seconds):
    #     """Called every frame when agent is quiet. idle_seconds grows over time.
    #     Use to transition to calm/breathing state."""
    #     calm = min(1.0, idle_seconds / 10.0)
    #     # e.g. draw a slow breathing ring when calm > 0.5
    #     pass

    # ── Intensity curve ──────────────────────────────────────────────────────
    # def intensity_curve(self, raw):
    #     """Shape how raw intensity (0..1.5) feels for this theme."""
    #     return math.pow(raw, 1.5)   # exponential: dramatic peaks


# ── REQUIRED: register the plugin ────────────────────────────────────────────
register(MyTheme())


# ── ThemeConfig for themes.py (add this when listing in gallery) ──────────────
#
# In hermes_neurovision/themes.py, add to THEMES tuple:
#   "my-theme",
#
# Add to build_theme_config() configs dict:
#   "my-theme": ThemeConfig(name, "My Theme Display Name", "·",
#       background_density=0.025,  # 0.01=sparse → 0.05=dense stars
#       star_drift=0.04,
#       node_jitter=0.20,
#       packet_rate=0.28,
#       packet_speed=(0.04, 0.08),
#       pulse_rate=0.09,
#       edge_bias=0.50,
#       cluster_count=3,
#       palette=(curses.COLOR_CYAN, curses.COLOR_MAGENTA,
#                curses.COLOR_WHITE, curses.COLOR_BLUE)),
