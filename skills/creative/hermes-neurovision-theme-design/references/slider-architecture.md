# Slider Architecture — TuneSettings Full Reference

The tuner panel (opened with `t` at runtime) exposes a TuneSettings object
that is available on `state.tune` inside every draw_extras / draw_background call.
This is the primary way screens respond to user input in real time.

---

## Accessing TuneSettings

```python
def draw_extras(self, stdscr, state, color_pairs):
    tune = state.tune          # TuneSettings | None (always set in live/gallery)
    if tune is None:           # defensive guard, should never fire
        return
    speed  = tune.animation_speed    # 0.1 – 5.0, default 1.0
    dense  = tune.particle_density   # 0.0 – 3.0, default 1.0
    # ... etc.
```

---

## Slider Fields (float, all user-adjustable in the Tuner panel)

| Attribute          | Range    | Default | What to drive with it                          |
|--------------------|----------|---------|------------------------------------------------|
| burst_scale        | 0 – 3    | 1.0     | particle burst count on events                 |
| packet_rate_mult   | 0 – 3    | 1.0     | data-packet spawn frequency on edges           |
| pulse_rate_mult    | 0 – 3    | 1.0     | pulse ring spawn frequency                     |
| particle_density   | 0 – 3    | 1.0     | ambient particle count; multiply your target_n |
| event_sensitivity  | 0 – 3    | 1.0     | threshold for triggering reactions             |
| animation_speed    | 0.1 – 5  | 1.0     | **master speed** — scale all time constants    |
| warp_strength      | 0 – 3    | 1.0     | distortion magnitude in warp_field()           |
| void_intensity     | 0 – 3    | 1.0     | size / count of void erasure points            |
| force_strength     | 0 – 3    | 1.0     | magnitude of force_points() push/pull          |
| decay_rate         | 0 – 3    | 1.0     | density grid decay speed (higher = faster fade)|
| emergent_speed     | 0 – 3    | 1.0     | emergent system step rate (0 = paused)         |
| emergent_opacity   | 0 – 1    | 1.0     | emergent layer alpha blend                     |
| sound_volume       | 0 – 1    | 0.5     | master volume for SoundEngine                  |

### Canonical Coupling Patterns

```python
# Animation speed — scale the frame counter's effective increment
t = state.frame * 0.05 * (state.tune.animation_speed if state.tune else 1.0)

# Particle target — scale from a baseline count
density = state.tune.particle_density if state.tune else 1.0
target_n = int(60 * density)
while len(self._particles) < target_n:
    self._particles.append(self._spawn())

# Density grid decay — faster at higher decay_rate
dr = state.tune.decay_rate if state.tune else 1.0
decay = max(0.80, 0.977 - 0.015 * dr)

# Speed + intensity together — natural for attractors
spd = state.tune.animation_speed if state.tune else 1.0
steps = int(600 * (0.5 + state.intensity_multiplier) * spd)

# Warp amplitude
ws = state.tune.warp_strength if state.tune else 1.0
warp_amp = 3.0 * ws * state.intensity_multiplier
```

---

## Toggle Fields (bool)

| Attribute         | Default | Controls                                           |
|-------------------|---------|----------------------------------------------------|
| show_packets      | True    | data-packet travel on graph edges                  |
| show_particles    | True    | ambient particle system                            |
| show_pulses       | True    | pulse rings from event reactions                   |
| show_stars        | True    | background star field                              |
| show_background   | True    | draw_background() call (hybrid themes)             |
| show_nodes        | True    | graph node rendering                               |
| show_flash        | True    | screen flash on events                             |
| show_spawn_node   | True    | spawn-point node visibility                        |
| show_streaks      | True    | streak/comet trails                                |
| show_specials     | True    | special effect overlays                            |
| show_overlays     | True    | overlay_effects layer                              |
| color_shifts      | True    | palette_shift() reactions                          |
| mask_enabled      | True    | render_mask() clipping                             |
| symmetry_enabled  | True    | symmetry() mirroring / rotation                    |
| reactive_elements | True    | master gate for ReactiveElement layer              |
| sound_enabled     | True    | master gate for all sound output                   |

### Respecting Toggles in draw_extras

The graph layer respects its toggles automatically. For things you draw
yourself in draw_extras, check the relevant flags:

```python
def draw_extras(self, stdscr, state, color_pairs):
    tune = state.tune
    if tune and not tune.show_particles:
        self._particles.clear()
        return

    spd = tune.animation_speed if tune else 1.0
    t = state.frame * 0.04 * spd
    density = tune.particle_density if tune else 1.0
    target_n = int(80 * density)
```

---

## Post-Processing Fields (int, consumed by renderer)

These are owned by the renderer, not draw_extras. Plugins seed default values
via hook methods; the tuner lets users override them at runtime.

| Attribute      | Type | Default | Plugin hook that seeds it     |
|----------------|------|---------|-------------------------------|
| echo_frames    | int  | 0       | echo_decay() → int            |
| glow_radius    | int  | 0       | glow_radius() → int           |
| parallax_depth | int  | 1       | depth_layers() → int          |

When echo_frames > 0, the renderer blends recent frames — your draw_extras
code needs no changes to benefit from it.

---

## TuneSettings Lifecycle

1. App creates one TuneSettings() at startup.
2. Each render frame: `state.tune = self.tune` before calling renderer.
3. Renderer passes state to all plugin hooks — state.tune is always live.
4. User opens tuner with `t`, adjusts values interactively.
5. Values persist for the session (not saved to disk).

No polling or callbacks needed. Read state.tune every frame.
