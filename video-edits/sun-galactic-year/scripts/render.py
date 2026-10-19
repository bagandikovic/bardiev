import cv2, numpy as np, subprocess, sys, math
sys.path.insert(0, sys.argv[3]); from desub import clean
SRC, OUT = sys.argv[1], sys.argv[2]
W, H, FPS = 1080, 1920, 60
CUTS   = [5.27, 11.82, 18.4, 24.9, 27.8, 37.6, 40.8, 44.18, 48.6]       # scene changes -> whip zoom + flash
PUNCH  = [0.0, 8.4, 21.8, 30.6, 36.4, 46.0, 49.0]                       # key words/numbers -> punch-in
SHAKE  = [(19.0, 27.6, 1.0)]                                            # "races through space ... incredibly fast"
WARP   = (18.6, 25.6)                                                   # warp-speed star streaks
SEGS   = [0.0] + CUTS + [51.0]

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
r = np.sqrt(((xx - W/2)/(W/2))**2 + ((yy - H/2)/(H/2))**2)
VIG = np.clip(1.0 - 0.38*np.clip(r - 0.45, 0, None)**1.6, 0, 1)[..., None].astype(np.float32)
rng = np.random.default_rng(7)
GRAIN = [rng.normal(0, 3.2, (H//2, W//2, 1)).astype(np.float32) for _ in range(6)]

# warp starfield particles (3D, flying toward camera)
NP = 260
P = np.c_[rng.uniform(-1, 1, NP), rng.uniform(-1.8, 1.8, NP), rng.uniform(0.1, 1, NP)]

def env_decay(t, events, amp, tau):
    return sum(amp*math.exp(-(t-e)/tau) for e in events if 0 <= t-e < 6*tau)

def ease(x): return 0.5 - 0.5*math.cos(math.pi*min(max(x, 0), 1))

def window(t, a, b, fade=0.5):
    return ease((t-a)/fade) * ease((b-t)/fade)

def warp_overlay(t, k):
    ov = np.zeros((H//2, W//2, 3), np.float32)
    if k <= 0: return None
    speed = 0.9*k
    for i in range(NP):
        x, y, z = P[i]
        z2 = z - speed/FPS
        if z2 <= 0.05:
            P[i] = [rng.uniform(-1, 1), rng.uniform(-1.8, 1.8), 1.0]; continue
        p1 = (int(W/4 + x/z*W/4), int(H/4 + y/z*W/4))
        p0 = (int(W/4 + x/z2*W/4), int(H/4 + y/z2*W/4))
        P[i, 2] = z2
        b = min(1.0, 0.25/z2) * k
        cv2.line(ov, p1, p0, (255*b, 235*b, 220*b), 1, cv2.LINE_AA)
    return cv2.GaussianBlur(ov, (0, 0), 0.8)

def transform(t):
    seg = max(i for i, s in enumerate(SEGS[:-1]) if t >= s)
    a, b = SEGS[seg], SEGS[seg+1]
    s = 1.04 + 0.05*ease((t-a)/(b-a))                     # slow Ken Burns push-in per shot
    s += env_decay(t, PUNCH, 0.075, 0.22)                  # punch-in on key beats
    s += env_decay(t, CUTS, 0.14, 0.12)                    # whip-zoom into each new shot
    s += sum(0.10*ease((t-(c-0.25))/0.25) for c in CUTS if c-0.25 <= t < c)   # zoom-out into the cut
    dx = dy = rot = 0.0
    for (sa, sb, amp) in SHAKE:
        k = window(t, sa, sb)*amp
        if k:
            dx += k*(9*math.sin(t*37.0) + 6*math.sin(t*61.3+1))
            dy += k*(9*math.sin(t*43.1+2) + 6*math.sin(t*71.7))
            rot += k*0.35*math.sin(t*23.0)
    dx += env_decay(t, CUTS, 1, 0.1)*14*math.sin(t*90)    # tiny jolt on cuts
    M = cv2.getRotationMatrix2D((W/2, H/2), rot, s); M[0, 2] += dx; M[1, 2] += dy
    return M

ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
    "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
    "-profile:v", "high", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
cap = cv2.VideoCapture(SRC); n = 0
while True:
    ok, f = cap.read()
    if not ok: break
    t = n/FPS
    f = clean(f)
    f = cv2.warpAffine(f, transform(t), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    x = f.astype(np.float32)
    # chromatic split while the speed shake is strongest
    ca = window(t, 21.6, 27.2)*4 + env_decay(t, CUTS, 6, 0.08)
    if ca > 0.5:
        sh = int(round(ca)); x[..., 2] = np.roll(x[..., 2], sh, 1); x[..., 0] = np.roll(x[..., 0], -sh, 1)
    # grade: contrast + saturation pop
    g = x.mean(2, keepdims=True)
    x = g + (x - g)*1.18
    x = (x - 128)*1.07 + 128
    # warp-speed streaks
    k = window(*(t,) + WARP, fade=0.8) * 0.9
    ov = warp_overlay(t, k)
    if ov is not None: x += cv2.resize(ov, (W, H))
    # flash on cuts + soft light-in at the start
    fl = env_decay(t, CUTS, 110, 0.07) + (90*math.exp(-t/0.25))
    x += fl
    x *= VIG
    x += cv2.resize(GRAIN[n % 6], (W, H))[..., None]
    x *= min(1.0, (51.0 - t)/0.6)                          # fade out
    ff.stdin.write(np.clip(x, 0, 255).astype(np.uint8).tobytes())
    n += 1
    if n % 300 == 0: print(n, flush=True)
ff.stdin.close(); ff.wait(); print("done", n)
