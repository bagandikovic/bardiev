# Agar Quyosh birdan yoʻqolsa? — what if the Sun suddenly disappeared

A 90-second video (`quyosh_yoqolsa.mp4`, 1920×1080, 30 fps, captions in Uzbek) of a
Newtonian N-body simulation of the Solar System after the Sun is removed.

## Physics

* **Bodies:** the 8 planets, Pluto, Ceres, Vesta and Halley's comet (12 bodies), with real masses.
* **Initial state:** real orbital elements (JPL approximate Keplerian elements, Standish) propagated
  to 2026-10-25, when the Sun vanishes. That gives real distances and real orbital velocities.
  Vesta's orbital phase is approximate.
* **After t = 0:** the Sun's gravity is switched off instantly, as Newtonian gravity requires.
  The bodies then move only under their mutual gravity: full N-body, kick-drift-kick leapfrog
  with an adaptive step (about 76 000 steps to 10 000 years). Halving the step changes the results
  by less than 0.01 m/s.
* **Collisions:** every step checks each pair's closest approach along the segment. If two bodies
  touch, they merge (mass and momentum are conserved), sparks are drawn and a crackle is played.
  **With real distances and velocities no collision happens.** The closest approach is
  Mercury–Earth at 12.6 million km on day 44. Every body leaves on a nearly straight line,
  because the planets' relative speeds (km/s) are far larger than their mutual escape speeds (m/s).
  The planets bend Earth's velocity by only 12.5 m/s in 10 000 years. The largest deflection is
  Mars's: about 300 m/s after it passes 55 million km from Jupiter at 1.27 years.
* **Moons** stay bound to their planets. The Earth–Moon orbital energy is −0.53 km²/s² < 0, so the
  pair flies off together.

## Video timeline

| Time | Content |
|------|---------|
| 0–6 s | Normal Solar System (Keplerian orbits); the Sun vanishes at 6 s |
| 6–40 s | Early phase: from minutes to 2 years (8 min 20 s light delay, temperature, orbit crossings, near-misses, the Earth–Moon pair) |
| 40–55 s | Sped up: 2 → 10 000 years; bodies crossing the heliopause (120 AU) are listed |
| 55–90 s | Final stage at 10 000 years: log-distance map, where Earth is (≈63 000 AU ≈ 1 light-year), a table of all objects that left the system, and conclusions |

Earth's temperature figures (≈ −18 °C after a week, ≈ −73 °C after a year) are commonly quoted
estimates and are labelled *taxminiy* (approximate) in the video.

The soundtrack is synthesised in `audio.py` (no external samples). It has a warm pad while the Sun
is there, a boom when it vanishes, a dark drone, a rising tone during the speed-up and a cold pad
in the final stage. Chimes mark captions and pings mark heliopause crossings.

## Build

```sh
./make_video.sh      # python3 + numpy + pillow, ffmpeg (libx264), Inter font
```

`sim.py` → `frames.npz`, `events.json` · `render.py` → `video_silent.mp4`, `cues.json` ·
`audio.py` → `soundtrack.wav` · ffmpeg mux → `quyosh_yoqolsa.mp4`.
`python3 render.py --stills 12 50 72` renders single frames (seconds) for checking.
