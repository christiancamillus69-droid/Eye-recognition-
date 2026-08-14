# Orbit Lock

A one-input arcade game. You orbit a core while rings collapse inward — each has a gap.
Your only control reverses your spin. Thread every gap.

Open `index.html` in a browser. No build step, no dependencies, no network calls.

## Controls

Tap, click, or press **Space** to reverse your direction of spin. That's the whole input.

## How it scores

| Event | Reward |
| --- | --- |
| Thread a gap | 10 × multiplier |
| Thread it *close to the edge* — a **perfect** | 2× that, plus 25, plus charge |
| Collect a drifting marker | 75 × multiplier, plus charge |
| Ring burned during overdrive | 30 × multiplier |

The multiplier climbs one step per 5-ring chain and resets when you breach. Perfects fill
the charge meter; when it fills, the core enters **overdrive** for 4.5 seconds and burns
rings out of the sky instead of dying to them.

## Difficulty

Level rises every 6 rings. Ring speed, spin rate, and gap tightness all scale with it, and
two-gap rings (drawn as a dashed ring, so they read as a different material) appear from
level 3.

Cadence is defined in *seconds between arrivals* rather than pixels of spacing, with a floor
of 0.95s — the player needs time to physically travel between two gaps, so past that floor
difficulty comes from speed, gap size, and spin instead of raw density. Each new gap is also
anchored within the arc the player can actually cover before the ring lands, so a run is
never lost to an unreachable spawn.

## Notes

Single file, ~700 lines. Canvas 2D for the scope, WebAudio for procedurally generated sound
(no audio assets), `localStorage` for best score, runs, and top chain. Honors
`prefers-reduced-motion` by dropping screen shake and the radar sweep.
