"""Synthesised soundtrack (no external samples), timed from cues.json.

0-6 s warm pad with the Sun · 6 s implosion boom · 6-40 s dark drone with a slow pulse ·
40-55 s rising tension for the speed-up · 55-90 s calm, cold ambience.
Soft chimes mark on-screen captions, pings mark bodies crossing the heliopause,
crackling sparks mark collisions (if the simulation produces any).
"""
import json
import wave

import numpy as np

from timeline import DURATION, T_VANISH, T_FAST, T_FINALSTAGE

SR = 44_100
N = int(SR * DURATION)
t = np.arange(N) / SR
rng = np.random.default_rng(3)


def env(t0, t1, attack, release):
    e = np.clip((t - t0) / attack, 0, 1) * np.clip((t1 - t) / release, 0, 1)
    return e * e * (3 - 2 * e)


def filtered(x, lo=0.0, hi=SR / 2):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / (1 + (f / hi) ** 4)
    if lo > 0:
        X *= 1 / (1 + (lo / np.maximum(f, 1e-3)) ** 4)
    return np.fft.irfft(X, len(x))


def pad(freqs, e, amp, detune=0.25):
    out = np.zeros((N, 2))
    for k, fr in enumerate(freqs):
        lfo = 0.75 + 0.25 * np.sin(2 * np.pi * (0.07 + 0.03 * k) * t + k)
        for ch, dt in enumerate((-detune, detune)):
            ph = 2 * np.pi * (fr + dt) * t
            out[:, ch] += (np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)) * lfo
    return out * (amp / len(freqs)) * e[:, None]


def one_shot(sig, at, out, gain=1.0, pan=0.0):
    i = int(at * SR)
    if i >= N:
        return
    sig = sig[: N - i]
    out[i:i + len(sig), 0] += sig * gain * (1 - pan) / 1
    out[i:i + len(sig), 1] += sig * gain * (1 + pan) / 1


def bell(freq, dur=2.2):
    tt = np.arange(int(dur * SR)) / SR
    return (np.sin(2 * np.pi * freq * tt) + 0.35 * np.sin(2 * np.pi * freq * 2.76 * tt) * np.exp(-tt * 3)
            + 0.15 * np.sin(2 * np.pi * freq * 5.4 * tt) * np.exp(-tt * 6)) * np.exp(-tt * 2.2) \
        * np.clip(tt / 0.005, 0, 1)


def boom():
    tt = np.arange(int(4.5 * SR)) / SR
    f = 28 + 70 * np.exp(-tt * 2.5)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 0.9)
    noise = filtered(rng.standard_normal(len(tt)), hi=900) * np.exp(-tt * 3.5) * 0.6
    crack = filtered(rng.standard_normal(len(tt)), lo=1500) * np.exp(-tt * 18) * 0.5
    return (body + noise + crack) * np.clip(tt / 0.004, 0, 1)


def thump():
    tt = np.arange(int(0.5 * SR)) / SR
    return np.sin(2 * np.pi * (45 + 40 * np.exp(-tt * 25)) * tt) * np.exp(-tt * 9)


def sparks():
    tt = np.arange(int(1.5 * SR)) / SR
    imp = (rng.random(len(tt)) < 0.004 * np.exp(-tt * 2)) * rng.uniform(-1, 1, len(tt))
    return filtered(imp, lo=2000) * 6 * np.exp(-tt * 2)


def reverb(x, seconds=2.8, mix=0.28):
    n = int(seconds * SR)
    out = np.empty_like(x)
    for ch in range(2):
        ir = rng.standard_normal(n) * np.exp(-np.arange(n) / SR * 3.2 / seconds * 3)
        ir = filtered(ir, hi=5000)
        ir /= np.sqrt((ir ** 2).sum())
        L = len(x) + n
        wet = np.fft.irfft(np.fft.rfft(x[:, ch], L) * np.fft.rfft(ir, L), L)[: len(x)]
        out[:, ch] = x[:, ch] * (1 - mix) + wet * mix * 2.5
    return out


def main():
    with open("cues.json") as fh:
        cues = json.load(fh)
    mix = np.zeros((N, 2))

    # intro: warm major pad, the Sun's hum
    mix += pad([110, 138.59, 164.81, 220], env(0, T_VANISH, 1.2, 0.08), 0.55)
    mix += pad([55], env(0, T_VANISH, 1.0, 0.08), 0.25)
    # reverse swell into the vanishing
    sw = filtered(rng.standard_normal(N), lo=200, hi=4000) * np.clip((t - (T_VANISH - 1.4)) / 1.4, 0, 1) ** 3
    mix += (sw * (t < T_VANISH) * 0.35)[:, None]
    one_shot(boom(), T_VANISH, mix, 0.9)

    # early phase: dark drone, space wind, slow heartbeat pulse
    e1 = env(T_VANISH + 0.5, T_FAST + 1.0, 3.0, 1.5)
    mix += pad([55, 82.41, 130.81, 174.61], e1, 0.5, detune=0.4)
    wind = filtered(rng.standard_normal(N), lo=150, hi=900) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.11 * t))
    mix += (wind * e1 * 0.10)[:, None]
    for s in np.arange(T_VANISH + 4, T_FAST, 1.25):
        one_shot(thump(), s, mix, 0.22)

    # speed-up: rising tone + opening noise filter
    u = np.clip((t - T_FAST) / (T_FINALSTAGE - T_FAST), 0, 1)
    e2 = env(T_FAST - 0.5, T_FINALSTAGE + 0.4, 1.0, 0.4)
    rise = np.zeros((N, 2))
    for base in (110, 165, 220):
        f = base * 2 ** (2 * u)
        ph = 2 * np.pi * np.cumsum(f) / SR
        rise[:, 0] += np.sin(ph) + 0.3 * np.sin(2 * ph)
        rise[:, 1] += np.sin(ph * 1.003) + 0.3 * np.sin(2 * ph * 1.003)
    trem = 0.7 + 0.3 * np.sin(2 * np.pi * (2 + 8 * u) * t)
    mix += rise * (e2 * (0.10 + 0.25 * u ** 1.5) * trem)[:, None]
    noise = rng.standard_normal(N)
    lo_band, hi_band = filtered(noise, hi=500), filtered(noise, lo=500, hi=6000)
    mix += ((lo_band * (1 - u) + hi_band * u) * e2 * (0.05 + 0.18 * u ** 2))[:, None]
    mix += pad([55, 82.41], e2, 0.35)

    # final stage: soft whoosh, then a cold, calm A-minor(add9) pad
    wh = filtered(rng.standard_normal(int(2.5 * SR)), lo=300, hi=3000)
    tt = np.arange(len(wh)) / SR
    one_shot(wh * np.exp(-tt * 1.8) * np.clip(tt / 0.05, 0, 1), T_FINALSTAGE, mix, 0.35)
    one_shot(thump(), T_FINALSTAGE, mix, 0.5)
    e3 = env(T_FINALSTAGE, DURATION, 2.5, 3.0)
    mix += pad([110, 164.81, 220, 261.63, 329.63, 493.88], e3, 0.55, detune=0.6)
    mix += pad([55], e3, 0.2)

    # cues
    for k, s in enumerate(cues["chimes"]):
        if s > T_VANISH + 0.1:
            one_shot(bell([659.25, 880.0, 783.99, 987.77][k % 4]), s, mix, 0.10, pan=0.3 * ((-1) ** k))
    for k, s in enumerate(cues["pings"]):
        one_shot(bell(1318.5 + 60 * k, 0.9), s, mix, 0.07, pan=-0.5 + k / max(1, len(cues["pings"])))
    for s in cues["collisions"]:
        one_shot(sparks(), s, mix, 0.6)

    mix = reverb(mix)
    mix *= np.clip(t / 0.5, 0, 1)[:, None] * np.clip((DURATION - t) / 1.5, 0, 1)[:, None]
    mix = np.tanh(mix * 0.9 / np.percentile(np.abs(mix), 99.7))    # soft limiter so the boom doesn't dwarf the rest
    mix *= 0.89 / np.abs(mix).max()
    pcm = (mix * 32767).astype(np.int16)
    with wave.open("soundtrack.wav", "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print("soundtrack.wav written")


if __name__ == "__main__":
    main()
