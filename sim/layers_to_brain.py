"""
Manim (ManimCE, https://manim.community) scene built on top of fig1.svg (imported via
SVGMobject): every static element -- skull outline, cortex silhouette, legend, left
caption -- comes straight from the paper figure. Pieces that move or reveal later are
picked out by submobject index (found by inspecting each submobject's fill color and
bounding box in SVGMobject(fig1.svg).submobjects) rather than being hand-redrawn.

Two SVG quirks drive most of the helper functions below:
  - manim ignores stroke-dasharray, so "dashed" skip bars/ellipses/legend-swatch are
    rebuilt as (solid fill) + DashedVMobject(stroke-only outline) to actually dash.
  - the cortex-ellipse and legend-swatch paths are hand-drawn "sketchy" wobbly curves
    (meant for a real SVG renderer to dash evenly by arc length); DashedVMobject on
    the raw path inherits that wobble, so styled_ellipse()/styled_legend_swatch() fit
    a clean Ellipse/Rectangle to each path first and dash that instead.

Narrative (one-way, no mirrored reverse -- ends with a crossfade back to the start so
the clip loops cleanly):
    1. frozen transformer, every layer kept (all bars pink).
    2. router fades in, then assigns skip/keep/repeat one layer at a time,
       bottom-to-top -- each step fires that layer's fan line together with its
       recolor. The two repeat layers turn green together as one step (a single
       repeated block, not two decisions).
    3. "in" connector fades in and a bottom-to-top wave along the inter-layer arrows
       stands in for the prompt feeding through the now-programmed stack (detouring
       back down through the repeat block before continuing up), then the response
       box + "out" connector fade in.
    4. cleanup fades out everything stage-2/3-only (fan lines, triangle, init tag,
       router connector, repeat panel, 8 of the 9 arrows), then the program morphs
       into its thalamo-cortical reading in one motion: bars -> cortex ellipses
       (order fixed by the ellipses' own angular position around the thalamus, see
       ELLIPSE_FOR_SLOT), surviving arrow -> sweep arrow, "..." dots -> cortex dots,
       self-loop icon -> small loop, in/out connectors -> brain's in/out connectors,
       router pill/label -> thalamus core/label, thalamus band + "Cortex" label fade
       in alongside. The right-hand caption fades in a beat later, on its own.
    5. hold on the brain, then the whole left side reappears so both halves of fig1
       are visible together (matching the static original), hold, then crossfade
       back to the stage-1 look to reset the loop.

Appearing pieces are built fresh from `svg_ref` (an untouched copy of the parsed SVG)
rather than faded from an already-hidden live mobject, since FadeIn on a mobject
already zeroed via set_opacity(0) fades 0 -> 0. Mobjects that later serve as
ReplacementTransform sources are extracted as independent top-level mobjects up
front, since Transform's cleanup doesn't reach into a parent group to detach one.

Setup (once, from inside sim/): pycairo (a manim dependency) needs pkg-config to see
zlib/expat, keg-only on Apple Silicon Homebrew -- keep the venv's own bin *ahead* of
Homebrew's on PATH (the reverse order silently runs `pip install` against the global
site-packages instead of the venv, reporting everything "already satisfied"):
    brew install zlib expat pkgconf
    python3.11 -m venv .venv && source .venv/bin/activate
    export PKG_CONFIG_PATH="/opt/homebrew/opt/zlib/lib/pkgconfig:/opt/homebrew/opt/expat/lib/pkgconfig:$PKG_CONFIG_PATH"
    export PATH="$VIRTUAL_ENV/bin:/opt/homebrew/bin:$PATH"
    pip install -r requirements.txt

Render (from inside sim/ -- SVG_PATH above is relative to this file's directory):
    manim -qh layers_to_brain.py LayersToBrain     # 1080p60
    manim -ql layers_to_brain.py LayersToBrain     # quick draft
"""
import numpy as np
from manim import (
    Scene, SVGMobject, VGroup, Ellipse, Rectangle, DashedVMobject, AnimationGroup, LaggedStart, Succession, Wait,
    ReplacementTransform, FadeIn, FadeOut, Indicate, ORIGIN, WHITE, YELLOW,
)

SVG_PATH = "fig1.svg"
SVG_WIDTH = 13.2
BAR_RECOLOR_RUNTIME = 0.45

# --- submobject indices in SVGMobject(fig1.svg) --------------------------------
ROUTER_PILL = 42                       # flies over and becomes the thalamus core
ROUTER_LABEL = 43
ROUTER_CONNECTOR = 7                   # prompt-to-router arrow
TRIANGLE = 65                          # pale-yellow fan backdrop
INIT_TAG = [76, 77]                    # "initialize program" box + text
SELF_LOOP_ICON = 64                    # loop-back icon next to the repeat layers
SELF_LOOP_PANEL = 44                    # gray background behind the repeat bars
SWEEP_ARROW = 98                       # arrow sweeping across the cortex
LOOP_ARROW = 99                        # small loop between the repeat-colored areas
LEFT_ARROWS = [45, 46, 47, 48, 49, 50, 51, 52, 55]   # the 9 inter-layer up-arrows
WAVE_ORDER = [55, 50, 51, 49, 45, 48, 47, 46, 52]    # LEFT_ARROWS, bottom-to-top
DOTS_ARROW_ID = 45     # this one gap is fig1's "..." ellipsis marker, not a real
                        # arrow (width 0.012 vs. the real arrows' 0.027, and twice
                        # the curve count) -- it must never be picked to stand in
                        # for the sweep arrow, only ever fade out like the rest
LEFT_IN = [6, 38]                       # in-arrow + "in" label: present from the start
LEFT_OUT = [5, 40]                      # out-arrow + "out" label: only once there's a response
RIGHT_IN = [101, 39]
RIGHT_OUT = [100, 41]
PROMPT_GROUP = list(range(15, 30))      # box + text/equation glyphs
RESPONSE_GROUP = list(range(30, 38))    # box + text glyphs
LEGEND_SKIP_SWATCH = 9
THALAMUS_CORE = 81
THALAMUS_BAND = 80
CORTEX_LABEL = 86
THALAMUS_LABEL = 85
CORTEX_DOTS = 97                        # "..." marking the other, unlabeled cortical areas
RIGHT_CAPTION = [1, 2]

# left stack bars, top-to-bottom, with their (kind, svg index)
BAR_SLOTS = [
    ("keep", 53), ("skip", 59), ("skip", 63), ("keep", 54), ("skip", 60),
    ("skip", 62), ("repeat", 57), ("repeat", 58), ("skip", 61), ("keep", 56),
]
KEEP_TEMPLATE = 53
# matching cortex-area ellipse, and matching fan line, for each slot above.
# ELLIPSE_FOR_SLOT is not a free choice: tracing the cortex ellipses' angular
# position around the thalamus gives a spatial color sequence (keep, skip,
# repeat, repeat, skip, skip, keep, skip, skip, keep) that matches the bars'
# *temporal* order (bottom-to-top, i.e. reversed(BAR_SLOTS)) exactly -- the
# original artist clearly intended a 1:1 correspondence, arc position for
# processing order. An earlier version of this mapping matched colors without
# checking that ordering, which scrambled it within each color group (e.g. the
# first/entry layer landed at the same spot the last/exit layer should have).
ELLIPSE_FOR_SLOT = [93, 92, 94, 95, 96, 91, 90, 89, 87, 88]
FAN_LINE_FOR_SLOT = [75, 72, 71, 67, 70, 69, 74, 73, 68, 66]

HIDDEN_AT_START = set(
    [ROUTER_PILL, ROUTER_LABEL, ROUTER_CONNECTOR, TRIANGLE, SELF_LOOP_ICON, SELF_LOOP_PANEL, SWEEP_ARROW, LOOP_ARROW,
     THALAMUS_CORE, THALAMUS_BAND, CORTEX_LABEL, THALAMUS_LABEL, CORTEX_DOTS, LEGEND_SKIP_SWATCH]
    + INIT_TAG + LEFT_IN + LEFT_OUT + RIGHT_IN + RIGHT_OUT + RIGHT_CAPTION + RESPONSE_GROUP
    + FAN_LINE_FOR_SLOT + [idx for _, idx in BAR_SLOTS] + ELLIPSE_FOR_SLOT
)
EXTRACTED_VISIBLE = set(LEFT_ARROWS)


class LayersToBrain(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        svg = SVGMobject(SVG_PATH).set(width=SVG_WIDTH).move_to(ORIGIN)
        svg_ref = svg.copy()  # pristine style/position reference, never added to the scene

        slot_centers = [svg_ref.submobjects[idx].get_center() for _, idx in BAR_SLOTS]

        to_remove = [svg.submobjects[i] for i in HIDDEN_AT_START | EXTRACTED_VISIBLE]
        svg.remove(*to_remove)
        self.add(svg)

        def fresh(idx):
            return svg_ref[idx].copy()

        def fresh_triangle():
            # must render behind every fan line and the router pill, always -- z_index
            # makes that true regardless of add order. Without it, whichever fan line
            # happens to share a self.play() call with the triangle's own FadeIn (or,
            # in a VGroup, whichever ones are listed before it) renders *behind* the
            # triangle instead of in front, looking like that one line's color is off.
            return fresh(TRIANGLE).set_z_index(-1)

        def fresh_group(indices):
            return VGroup(*[fresh(i) for i in indices])

        def dashed(mobj, num_dashes):
            fill_part = mobj.copy().set_stroke(width=0)
            outline = DashedVMobject(mobj.copy().set_fill(opacity=0), num_dashes=num_dashes, dashed_ratio=0.5)
            return VGroup(fill_part, outline)

        def styled_bar(idx, kind):
            return dashed(fresh(idx), num_dashes=44) if kind == "skip" else fresh(idx)

        # The cortex-column shapes in fig1.svg aren't clean ellipses: they're
        # hand-drawn "sketchy" wobbly paths (~137 bezier curves for one small oval)
        # meant to be dashed by a real SVG renderer's stroke-dasharray, which
        # dashes evenly by rendered arc length and hides the wobble. Manim ignores
        # stroke-dasharray and DashedVMobject on that wobbly path inherits its
        # irregular curvature, so the dashes come out uneven and jagged. Fix: fit a
        # clean Ellipse (center/size/rotation via PCA on the path's own points,
        # since these ovals are pre-rotated to point radially) and dash that
        # instead, at the same visual density the original SVG's "4 4"
        # dasharray would give once scaled into this scene's units.
        def fit_ellipse(mobj):
            pts = np.array(mobj.get_points())[:, :2]
            center = pts.mean(axis=0)
            cov = np.cov((pts - center).T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            width, height = 2 * np.sqrt(2 * eigvals)
            angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
            return center, width, height, angle

        def styled_ellipse(idx, kind):
            src = svg_ref[idx]
            if kind != "skip":
                return fresh(idx)
            center, width, height, angle = fit_ellipse(src)
            a, b = width / 2, height / 2
            perimeter = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
            dash_cycle = 8 * (SVG_WIDTH / 1733)  # matches the SVG's "4 4" dasharray, scaled into this scene
            num_dashes = max(4, round(perimeter / dash_cycle))

            def clean_ellipse():
                e = Ellipse(width=width, height=height).rotate(angle)
                e.move_to([center[0], center[1], 0])
                return e

            fill_part = clean_ellipse().set_fill(src.fill_color, opacity=src.fill_opacity).set_stroke(width=0)
            outline = DashedVMobject(
                clean_ellipse().set_stroke(src.stroke_color, width=src.stroke_width).set_fill(opacity=0),
                num_dashes=num_dashes, dashed_ratio=0.5,
            )
            return VGroup(fill_part, outline)

        # same wobbly-path problem as the ellipses above, but for the small skip
        # legend swatch (a hand-drawn wobbly square, not rotated, so no PCA needed --
        # its own axis-aligned width/height already are its true size).
        def styled_legend_swatch():
            src = svg_ref[LEGEND_SKIP_SWATCH]
            w, h = src.width, src.height
            center = src.get_center()
            perimeter = 2 * (w + h)
            dash_cycle = 12 * (SVG_WIDTH / 1733)  # matches the SVG's "6 6" dasharray, scaled into this scene
            num_dashes = max(6, round(perimeter / dash_cycle))

            def clean_rect():
                return Rectangle(width=w, height=h).move_to(center)

            fill_part = clean_rect().set_fill(src.fill_color, opacity=src.fill_opacity).set_stroke(width=0)
            outline = DashedVMobject(
                clean_rect().set_stroke(src.stroke_color, width=src.stroke_width).set_fill(opacity=0),
                num_dashes=num_dashes, dashed_ratio=0.5,
            )
            return VGroup(fill_part, outline)

        # ---- stage 1: plain frozen transformer, every layer kept ------------
        # the prompt sits there already, but its "in" connector only arrives once the
        # router has finished and the forward pass is about to start -- see stage 3.
        pink_bars = [fresh(KEEP_TEMPLATE).move_to(c) for c in slot_centers]
        left_arrows = [fresh(i) for i in LEFT_ARROWS]
        legend_skip_swatch = styled_legend_swatch()
        self.add(VGroup(*pink_bars), *left_arrows, legend_skip_swatch)
        self.wait(1.3)  # enough to read the caption/prompt, not so long it looks stalled

        # ---- stage 2: the router appears, then decides one ray at a time ----
        router_pill = fresh(ROUTER_PILL)
        router_label = fresh(ROUTER_LABEL)
        router_connector = fresh(ROUTER_CONNECTOR)
        self.play(FadeIn(VGroup(router_pill, router_label, router_connector)), run_time=1.6)

        # bottom-to-top, matching the "in" side and the forward-pass wave below. The
        # two repeat slots are adjacent and turn together as one step -- the router
        # visits them as a single repeated block, not two independent decisions --
        # and the panel gets its own beat fully *before* that step: fading the panel
        # in in the same self.play() as the bars' ReplacementTransform caused them to
        # flicker (the transform's interpolated proxy mobject didn't consistently
        # honor the panel's z_index while both were animating at once).
        recolor_order = list(reversed(range(len(BAR_SLOTS))))
        repeat_pair = tuple(i for i, (kind, _) in enumerate(BAR_SLOTS) if kind == "repeat")
        steps, seen_repeat = [], False
        for slot_i in recolor_order:
            if slot_i in repeat_pair:
                if not seen_repeat:
                    steps.append(repeat_pair)
                    seen_repeat = True
            else:
                steps.append((slot_i,))

        triangle = fresh_triangle()
        init_tag = fresh(INIT_TAG[0])
        init_tag_label = fresh(INIT_TAG[1])
        self_loop_icon = None
        real_bars = [None] * len(BAR_SLOTS)
        fan_lines = [None] * len(BAR_SLOTS)
        for step_i, slots in enumerate(steps):
            if slots == repeat_pair:
                self_loop_icon = fresh(SELF_LOOP_ICON)
                self_loop_panel = fresh(SELF_LOOP_PANEL).set_z_index(-1)  # must sit behind the repeat bars
                self.play(FadeIn(VGroup(self_loop_icon, self_loop_panel)), run_time=BAR_RECOLOR_RUNTIME)
            anims = []
            for slot_i in slots:
                kind, idx = BAR_SLOTS[slot_i]
                real_bar = styled_bar(idx, kind)
                fan_line = fresh(FAN_LINE_FOR_SLOT[slot_i])
                anims += [ReplacementTransform(pink_bars[slot_i], real_bar), FadeIn(fan_line)]
                real_bars[slot_i] = real_bar
                fan_lines[slot_i] = fan_line
            if step_i == 0:
                anims.append(FadeIn(VGroup(triangle, init_tag, init_tag_label)))
            self.play(*anims, run_time=BAR_RECOLOR_RUNTIME)
        self.wait(1.4)  # a clear break: initialization is done, the forward pass hasn't started yet

        # ---- stage 3: the prompt feeds through the now-programmed stack -----
        # the "in" connector is itself part of the forward pass, so it only shows up
        # once the router is done and the pass is about to begin.
        # entry order bottom-to-top is recolor_order (same order the router used in
        # stage 2); entry_slots[k+1] is the bar the wave reaches after crossing
        # wave_arrows[k], since there are 10 bars and 9 gaps between them.
        entry_slots = recolor_order
        arrival_slots = entry_slots[1:]

        def compute_pulse(slot_i):
            # only keep/repeat bars actually run -- skip bars are dashed/hollow
            # already, so leaving them untouched here (no flash, unlike the solid
            # bars) is what visually shows the forward pass skipping over them
            # rather than computing them.
            kind, _ = BAR_SLOTS[slot_i]
            if kind == "skip":
                return None
            return Indicate(real_bars[slot_i], scale_factor=1.35, color=YELLOW)

        def arrival(arrow, slot_i):
            arrow_anim = Indicate(arrow, scale_factor=1.6, color=YELLOW)
            pulse = compute_pulse(slot_i)
            return AnimationGroup(arrow_anim, pulse) if pulse else arrow_anim

        left_in = [fresh(i) for i in LEFT_IN]
        first_pulse = compute_pulse(entry_slots[0])
        in_fade = FadeIn(VGroup(*left_in))
        # same beat-then-flash spacing as every later arrow-then-bar arrival below --
        # without it, the "in" connector and the very first bar landed in the same
        # instant, unlike every other step in the wave.
        if first_pulse:
            self.play(LaggedStart(in_fade, first_pulse, lag_ratio=0.7), run_time=0.8)
        else:
            self.play(in_fade, run_time=0.5)

        # the wave climbs to the top of the repeat block (3 gaps: into the block,
        # across it, to its top), the self-loop fires to show the block actually
        # running again, which sends the pass back *down* to re-cross the same
        # internal gap -- not straight on to the gap above the block -- before it
        # finally continues upward from there.
        wave_arrows = [left_arrows[LEFT_ARROWS.index(i)] for i in WAVE_ORDER]
        self.play(
            LaggedStart(*[arrival(wave_arrows[i], arrival_slots[i]) for i in range(3)], lag_ratio=0.7),
            run_time=1.2,
        )
        self.play(Indicate(self_loop_icon, scale_factor=1.3, color=YELLOW), run_time=0.7)
        # the block re-runs its two bars in the same order they were first entered
        # (bottom-to-top), one after the other -- not both at once, since they're
        # still two separate layers computing in sequence, just a repeated pair. The
        # arrow bump belongs *between* them (data crossing up from the lower bar to
        # the upper one after the lower one recomputes), not before the lower bar
        # has even fired.
        lower_slot, upper_slot = (i for i in recolor_order if i in repeat_pair)
        self.play(
            LaggedStart(
                Indicate(real_bars[lower_slot], scale_factor=1.3, color=YELLOW),
                Indicate(wave_arrows[2], scale_factor=1.6, color=YELLOW),
                Indicate(real_bars[upper_slot], scale_factor=1.3, color=YELLOW),
                lag_ratio=0.6,
            ),
            run_time=0.9,
        )
        self.play(
            LaggedStart(*[arrival(wave_arrows[i], arrival_slots[i]) for i in range(3, 9)], lag_ratio=0.7),
            run_time=1.8,
        )
        response_group = fresh_group(RESPONSE_GROUP)
        left_out = [fresh(i) for i in LEFT_OUT]
        self.play(FadeIn(response_group), FadeIn(VGroup(*left_out)), run_time=0.6)
        self.wait(2.0)

        # ---- stage 4: the program travels over into its cortical reading ----
        # only one middle inter-layer arrow actually becomes the sweep arrow, and
        # the "..." ellipsis dots become the brain's own "..." dots next to
        # "Cortex" (dots into dots, not dots into an arrow); the other 7 real
        # arrows simply fade out early, in this cleanup step, rather than
        # lingering through the whole 2.4s transform below. (An earlier version had
        # all 9 arrows converge into one shared ReplacementTransform target, which
        # does produce a real many-to-one merge rather than 9 overlapping copies --
        # but visually it read as "every arrow becomes the big one", not as one
        # arrow standing in for the sweep; and before that, the naive "pick the
        # middle of the 9" selection landed on the dots marker itself, since it
        # occupies the visually-middle gap in the stack.)
        real_wave_order = [a for a in WAVE_ORDER if a != DOTS_ARROW_ID]
        middle_arrow_id = real_wave_order[len(real_wave_order) // 2]
        middle_arrow = left_arrows[LEFT_ARROWS.index(middle_arrow_id)]
        dots_arrow = left_arrows[LEFT_ARROWS.index(DOTS_ARROW_ID)]
        other_arrows = [a for a in left_arrows if a is not middle_arrow and a is not dots_arrow]
        # triangle survives this cleanup -- unlike the rest, it isn't discarded, it
        # becomes the thalamus band below.
        self.play(
            FadeOut(VGroup(
                *fan_lines, init_tag, init_tag_label, router_connector, self_loop_panel, *other_arrows,
            )),
            run_time=0.5,
        )
        cortex_ellipses = [styled_ellipse(ELLIPSE_FOR_SLOT[i], kind) for i, (kind, _) in enumerate(BAR_SLOTS)]
        sweep_target = fresh(SWEEP_ARROW)
        loop_target = fresh(LOOP_ARROW)
        right_targets = [fresh(i) for i in RIGHT_IN + RIGHT_OUT]
        thalamus_core = fresh(THALAMUS_CORE)
        thalamus_band = fresh(THALAMUS_BAND)
        thalamus_label = fresh(THALAMUS_LABEL)
        cortex_label = fresh(CORTEX_LABEL)
        cortex_dots = fresh(CORTEX_DOTS)

        # the left stack empties out bottom-to-top (recolor_order again, same as
        # stages 2/3), each bar mostly finishing before the next starts (lag_ratio
        # close to 1). The dots gap is woven into that same sequence right after the
        # real bar physically below it (slot 5) rather than moving with the other
        # connectors -- it's still just another gap in this same stack, so it should
        # empty out in its actual stack position, not off on its own later.
        DOTS_AFTER_SLOT = 5
        bar_and_dots = []
        for slot in recolor_order:
            bar_and_dots.append(ReplacementTransform(real_bars[slot], cortex_ellipses[slot]))
            if slot == DOTS_AFTER_SLOT:
                bar_and_dots.append(ReplacementTransform(dots_arrow, cortex_dots))
        left_wave = LaggedStart(*bar_and_dots, lag_ratio=0.85, run_time=2.9)

        # the remaining connectors and the router/triangle settle into the thalamus
        # together, as one beat, starting once the wave reaches its third-to-last bar
        # rather than waiting for the whole stack to empty first -- so the right side
        # is already assembling while the last couple of bars are still landing,
        # instead of trailing behind and leaving the cortex looking unfinished.
        arrows_and_thalamus = AnimationGroup(
            ReplacementTransform(middle_arrow, sweep_target),
            ReplacementTransform(self_loop_icon, loop_target),
            *[ReplacementTransform(a, t) for a, t in zip(left_in + left_out, right_targets)],
            ReplacementTransform(router_pill, thalamus_core),
            ReplacementTransform(router_label, thalamus_label),
            ReplacementTransform(triangle, thalamus_band),
            FadeIn(cortex_label),
            run_time=1.4,
        )
        self.play(left_wave, Succession(Wait(2.05), arrows_and_thalamus))
        # the caption is the last thing to settle, a beat after everything else lands
        self.wait(1.5)
        right_caption = fresh_group(RIGHT_CAPTION)
        self.play(FadeIn(right_caption), run_time=0.6)
        self.wait(3.9)  # long enough to actually read the caption and take in the analogy

        # ---- final reveal: the whole left side simply reappears (no shuffling back
        # and forth) so the complete fig1 pairing is visible in one frame at once --
        final_bars = [styled_bar(idx, kind).move_to(c) for (kind, idx), c in zip(BAR_SLOTS, slot_centers)]
        final_arrows = [fresh(i) for i in LEFT_ARROWS]
        final_fan_lines = [fresh(i) for i in FAN_LINE_FOR_SLOT]
        final_triangle = fresh_triangle()
        final_init_tag = fresh_group(INIT_TAG)
        final_router = VGroup(fresh(ROUTER_PILL), fresh(ROUTER_LABEL), fresh(ROUTER_CONNECTOR))
        final_self_loop = VGroup(fresh(SELF_LOOP_ICON), fresh(SELF_LOOP_PANEL).set_z_index(-1))
        final_left_inout = [fresh(i) for i in LEFT_IN + LEFT_OUT]
        full_left = VGroup(
            *final_bars, *final_arrows, *final_fan_lines, final_triangle, final_init_tag,
            final_router, final_self_loop, *final_left_inout,
        )
        self.play(FadeIn(full_left), run_time=1.0)
        self.wait(4.0)

        # ---- reset: crossfade back to the stage-1 look, no reverse playback --
        brain_side = VGroup(
            *cortex_ellipses, sweep_target, loop_target, *right_targets,
            thalamus_core, thalamus_band, cortex_label, thalamus_label, cortex_dots, right_caption, response_group,
        )
        reset_bars = [fresh(KEEP_TEMPLATE).move_to(c) for c in slot_centers]
        reset_arrows = [fresh(i) for i in LEFT_ARROWS]
        self.play(
            FadeOut(brain_side),
            FadeOut(full_left),
            FadeIn(VGroup(*reset_bars, *reset_arrows)),
            run_time=1.0,
        )
        self.wait(0.6)
