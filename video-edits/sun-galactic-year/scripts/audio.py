# Original synthesized space score + SFX (royalty-free, generated from scratch)
import numpy as np, sys, wave
SR = 48000; DUR = 51.0; N = int(SR*DUR)
rng = np.random.default_rng(3)
t = np.arange(N)/SR
CUTS  = [5.27, 11.82, 18.4, 24.9, 27.8, 37.6, 40.8, 44.18, 48.6]
PUNCH = [8.4, 21.8, 30.6, 36.4, 46.0, 49.0]

def hz(m): return 440*2**((m-69)/12)
def lp(x, fc):            # one-pole lowpass (vectorised via cumulative filter)
    from scipy.signal import lfilter
    a = np.exp(-2*np.pi*fc/SR); return lfilter([1-a], [1, -a], x, axis=0)
def env(n, a, r):
    e = np.ones(n); A = int(a*SR); R = int(r*SR)
    if A: e[:A] = np.linspace(0, 1, A)
    if R: e[-R:] *= np.linspace(1, 0, R)
    return e
def put(buf, start, sig, gain=1.0):
    i = int(start*SR)
    if i < 0: sig = sig[-i:]; i = 0
    j = min(len(buf), i+len(sig)); buf[i:j] += gain*sig[:j-i]

def reverb(x, secs=3.5, mix=0.35):
    from scipy.signal import fftconvolve
    n = int(secs*SR); ir = rng.normal(0, 1, (n, 2))*np.exp(-np.linspace(0, 7, n))[:, None]
    ir = lp(ir, 6000); ir /= np.abs(ir).sum(0)*0.02
    wet = np.stack([fftconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], 1)
    return x*(1-mix) + wet*mix

music = np.zeros((N, 2)); sfx = np.zeros((N, 2))

# --- pad: D minor journey  Dm - Bb - F - C  (and a lift to Gm/A at the reveal)
prog = [(0.0, [50, 57, 62, 65]), (5.27, [46, 53, 58, 62]), (11.82, [41, 53, 57, 60]), (18.4, [48, 55, 60, 64]),
        (24.9, [50, 57, 62, 65]), (27.8, [46, 53, 58, 65]), (37.6, [43, 50, 55, 58]), (44.18, [45, 52, 57, 61]),
        (48.6, [50, 57, 62, 66])]
for k, (st, notes) in enumerate(prog):
    en = prog[k+1][0] if k+1 < len(prog) else DUR
    n = int((en-st+1.5)*SR); tt = np.arange(n)/SR
    ch = np.zeros((n, 2))
    for m in notes:
        for d, pan in ((-0.07, 0.2), (0.07, 0.8)):
            f = hz(m)*2**(d/12/10)
            saw = 2*((tt*f) % 1) - 1
            s = 0.6*np.sin(2*np.pi*f*tt) + 0.25*saw
            ch[:, 0] += s*(1-pan); ch[:, 1] += s*pan
    ch = lp(ch, 900 + 500*np.sin(2*np.pi*0.08*tt)[:, None].mean())
    ch *= env(n, 1.2, 1.5)[:, None]*0.06
    put(music, st-0.3, ch)
# sub drone
music += (0.10*np.sin(2*np.pi*hz(26)*t)*(0.6+0.4*np.sin(2*np.pi*0.05*t)))[:, None]

# --- sparkle arpeggio (bells), sparse at the start, denser in the middle, delay-ish via reverb
bpm = 100; step = 60/bpm/2
for k, (st, notes) in enumerate(prog):
    en = prog[k+1][0] if k+1 < len(prog) else DUR - 1
    tt0 = st; i = 0
    while tt0 < en:
        m = notes[i % 4] + 24; i += 1
        n = int(1.2*SR); tt = np.arange(n)/SR
        b = (np.sin(2*np.pi*hz(m)*tt) + 0.3*np.sin(2*np.pi*hz(m)*2.01*tt))*np.exp(-tt*4)
        pan = 0.5 + 0.35*np.sin(i)
        dens = 0.5 if st < 18 else 1.0
        if rng.random() < dens: put(music, tt0, np.c_[b*(1-pan), b*pan], 0.035)
        tt0 += step

# --- soft pulse (kick + sub) driving the "speed" section and the finale
def kick():
    n = int(0.5*SR); tt = np.arange(n)/SR
    f = 45 + 90*np.exp(-tt*30); ph = 2*np.pi*np.cumsum(f)/SR
    k = np.sin(ph)*np.exp(-tt*7); return np.c_[k, k]
beat = 60/bpm
for a, b, g in ((18.4, 27.6, 0.20), (27.8, 37.4, 0.13), (44.18, 50.0, 0.22)):
    x = a
    while x < b: put(music, x, kick(), g); x += beat
# hi-hat ticks in the speed section
x = 18.4 + beat/2
while x < 27.4:
    n = int(0.06*SR); hh = rng.normal(0, 1, n)*np.exp(-np.arange(n)/SR*80); hh -= lp(hh, 7000)
    put(music, x, np.c_[hh, hh], 0.05); x += beat/2

music = reverb(music, 3.5, 0.4)

# --- SFX
def whoosh(dur=0.9, lo=300, hi=4000):
    n = int(dur*SR); x = rng.normal(0, 1, (n, 2))
    from scipy.signal import butter, sosfilt
    out = np.zeros_like(x); seg = 512
    for s in range(0, n, seg):
        p = s/n; fc = lo*(hi/lo)**(np.sin(np.pi*p))
        sos = butter(2, [max(40, fc*0.5), min(SR/2-100, fc*1.5)], 'bandpass', fs=SR, output='sos')
        out[s:s+seg] = sosfilt(sos, x[max(0, s-2048):s+seg], axis=0)[-len(x[s:s+seg]):]
    e = np.sin(np.pi*np.linspace(0, 1, n))**2
    pan = np.linspace(0.15, 0.85, n)
    return np.c_[out[:, 0]*e*(1-pan), out[:, 1]*e*pan]*2
def boom(dur=2.2):
    n = int(dur*SR); tt = np.arange(n)/SR
    f = 32 + 70*np.exp(-tt*9); ph = 2*np.pi*np.cumsum(f)/SR
    b = np.sin(ph)*np.exp(-tt*2.2) + 0.25*lp(rng.normal(0, 1, n), 400)*np.exp(-tt*6)
    return np.c_[b, b]
def riser(dur=3.0):
    n = int(dur*SR); tt = np.arange(n)/SR; p = tt/dur
    f = 200*2**(3*p**2); ph = 2*np.pi*np.cumsum(f)/SR
    x = 0.4*np.sin(ph) + 0.3*np.sin(ph*1.5) + 0.5*(lambda nz: lp(nz, 1500)*(1-p) + lp(nz, 10000)*p)(rng.normal(0, 1, n))
    x *= p**2
    return np.c_[x, np.roll(x, 300)]
def flyby(dur=2.0):
    n = int(dur*SR); tt = np.arange(n)/SR; p = tt/dur
    f = 420*(1.25 - 0.5*np.tanh((p-0.5)*8)); ph = 2*np.pi*np.cumsum(f)/SR
    x = (0.5*np.sin(ph) + 0.6*lp(rng.normal(0, 1, n), 3000))*np.exp(-((p-0.5)/0.18)**2)
    pan = p
    return np.c_[x*(1-pan), x*pan]
def chime(dur=3.0):
    n = int(dur*SR); tt = np.arange(n)/SR; x = np.zeros(n)
    for m, d in ((86, 0), (90, 0.08), (93, 0.16), (98, 0.24)):
        k = int(d*SR); x[k:] += np.sin(2*np.pi*hz(m)*tt[:n-k])*np.exp(-tt[:n-k]*2.5)
    return np.c_[x, np.roll(x, 600)]

put(sfx, 0.0, boom(), 0.9); put(sfx, 0.05, chime(), 0.10)
for c in CUTS: put(sfx, c-0.55, whoosh(), 0.55)
for p in PUNCH: put(sfx, p, boom(1.6), 0.55)
put(sfx, 21.8-3.0, riser(3.0), 0.35)          # build into "720,000 km/h"
put(sfx, 19.2, flyby(), 0.6); put(sfx, 23.0, flyby(1.6), 0.5); put(sfx, 25.6, flyby(), 0.55)
put(sfx, 36.4, chime(), 0.12)                 # "galactic year"
put(sfx, 48.6-2.5, riser(2.5), 0.35)          # build into the punchline
put(sfx, 49.0, boom(2.0), 0.4)
sfx = reverb(sfx, 1.8, 0.2)

def norm(x, peak): return x/np.abs(x).max()*peak
music = norm(music, 0.45); sfx = norm(sfx, 0.6)
fade = np.ones(N); fade[-int(1.2*SR):] = np.linspace(1, 0, int(1.2*SR)); fade[:int(0.02*SR)] = np.linspace(0, 1, int(0.02*SR))
mix = (music + sfx)*fade[:, None]
mix = np.tanh(mix*1.3)/np.tanh(1.3)*0.89
def wr(path, x):
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1)*32767).astype('<i2').tobytes())
out = sys.argv[1]
wr(f"{out}/mix.wav", mix); wr(f"{out}/music_only.wav", music*fade[:, None]); wr(f"{out}/sfx_only.wav", sfx*fade[:, None])
print("ok")
