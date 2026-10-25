"""Render the simulation (frames.npz + events.json) to a silent 1920x1080 video.

Also writes cues.json: video times of sound cues, used by audio.py.
"""
import json
import math
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim import AU_KM, KMS, PLANET_ELEMENTS, SMALL_ELEMENTS, JD_VANISH, heliocentric_state
from timeline import FPS, N_FRAMES, T_VANISH, T_FAST, T_FINALSTAGE, sim_time

W, H = 1920, 1080
CX, CY = 820, 565            # map centre (old Sun position)
MAP_R = 470                  # px from centre to top of map
LCX, LCY, LOG_R = 690, 575, 430   # log-radial map used in the final stage
LY_AU = 63_241.08
YEAR = 365.25
BG = (6, 8, 16)

FONT_DIR = "/usr/share/fonts/opentype/inter/"
_fonts = {}


def font(size, weight="Medium"):
    key = (size, weight)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(f"{FONT_DIR}Inter-{weight}.otf", size)
    return _fonts[key]


def num(x, dec=0):
    """Uzbek number format: space thousands separator, comma decimal."""
    s = f"{x:,.{dec}f}".replace(",", " ").replace(".", ",")
    return s.replace("-", "−")


def fmt_elapsed(d):
    if d < 1 / 24:
        m = d * 1440
        return f"{int(m)} daqiqa {int((m % 1) * 60):02d} soniya"
    if d < 1:
        h = d * 24
        return f"{int(h)} soat {int((h % 1) * 60):02d} daqiqa"
    if d < YEAR:
        return f"{int(d)} kun {int((d % 1) * 24):02d} soat"
    y = d / YEAR
    if y < 100:
        return f"{int(y)} yil {int((y % 1) * YEAR)} kun"
    return f"{num(round(y))} yil"


def fmt_rate(days_per_s):
    if days_per_s < 1 / 1440:
        x = days_per_s * 86400
        return f"{num(x, 1 if x >= 1 else 3)} soniya"
    if days_per_s < 1 / 24:
        return f"{num(days_per_s * 1440, 1)} daqiqa"
    if days_per_s < 1:
        return f"{num(days_per_s * 24, 1)} soat"
    if days_per_s < YEAR:
        return f"{num(days_per_s)} kun"
    return f"{num(days_per_s / YEAR)} yil"


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def fade(sec, t0, t1, ramp=0.4):
    """Opacity of an element shown between t0 and t1."""
    if sec < t0 or sec > t1:
        return 0.0
    return min(1.0, (sec - t0) / ramp, (t1 - sec) / ramp)


def text(d, xy, s, size=24, color=(230, 235, 245), alpha=1.0, weight="Medium", anchor="la", shadow=True):
    if alpha <= 0:
        return
    f = font(size, weight)
    a = int(255 * alpha)
    if shadow:
        d.text((xy[0] + 2, xy[1] + 2), s, font=f, fill=(0, 0, 0, int(a * 0.8)), anchor=anchor)
    d.text(xy, s, font=f, fill=(*color, a), anchor=anchor)


def panel(d, box, alpha=1.0, border=(90, 110, 150)):
    if alpha <= 0:
        return
    d.rounded_rectangle(box, radius=14, fill=(10, 14, 28, int(232 * alpha)),
                        outline=(*border, int(160 * alpha)), width=2)


def glow_sprite(color, r):
    size = int(r * 7) | 1
    c = size // 2
    y, x = np.mgrid[:size, :size]
    dist = np.hypot(x - c, y - c)
    a = np.exp(-(dist / (r * 1.4)) ** 2) * 110
    a[dist <= r] = 255
    rgb = np.zeros((size, size, 3))
    rgb[:] = color
    inner = np.clip(1 - dist / max(r, 1), 0, 1)[..., None]
    rgb = np.clip(rgb + inner * 60, 0, 255)         # slightly brighter centre
    img = np.dstack([rgb, np.clip(a, 0, 255)]).astype(np.uint8)
    return Image.fromarray(img, "RGBA")


def orbit_path(name):
    """Sampled original Keplerian orbit (AU, ecliptic x/y)."""
    if name in PLANET_ELEMENTS:
        a = PLANET_ELEMENTS[name][0][0]
    else:
        a = SMALL_ELEMENTS[name][0]
    period = 2 * math.pi * math.sqrt(a ** 3) / 0.01720209895
    pts = [heliocentric_state(name, JD_VANISH + period * k / 2000)[0][:2] for k in range(2001)]
    return np.array(pts)


class Renderer:
    def __init__(self):
        data = np.load("frames.npz")
        self.times, self.P, self.V = data["times"], data["P"], data["V"]
        with open("events.json") as fh:
            self.ev = json.load(fh)
        self.names = self.ev["names"]
        self.colors = [tuple(c) for c in self.ev["colors"]]
        self.n = len(self.names)
        self.ie = self.names.index("Yer")
        rj = 69911
        self.rpx = [2.2 + 11.5 * (r / rj) ** 0.6 for r in self.ev["radius_km"]]
        self.sprites = [glow_sprite(c, r) for c, r in zip(self.colors, self.rpx)]
        self.sun = glow_sprite((255, 210, 90), 20)
        self.orbits = [orbit_path(nm) for nm in self.names]
        self.f_vanish = int(round(T_VANISH * FPS))
        rng = np.random.default_rng(7)
        bg = np.zeros((H, W, 3), np.uint8)
        bg[:] = BG
        img = Image.fromarray(bg)
        d = ImageDraw.Draw(img)
        for _ in range(900):
            x, y = rng.uniform(0, W), rng.uniform(0, H)
            b = int(rng.uniform(40, 170) * rng.uniform(0.3, 1) ** 2) + 25
            d.point((x, y), fill=(b, b, min(255, b + 20)))
        self.bg = img
        self.cues = {"vanish": T_VANISH, "chimes": [], "pings": [], "collisions": []}
        self.captions = self.build_captions()
        self.cues["chimes"] = [c[0] for c in self.captions]
        self.crossings = [c for c in self.ev["crossings"] if c["what"] == "Geliopauza"]
        for c in self.crossings:
            c["sec"] = self.sec_of(c["t"])
            self.cues["pings"].append(c["sec"])
        for c in self.ev["collisions"]:
            c["sec"] = self.sec_of(c["t"])
            self.cues["collisions"].append(c["sec"])

    # ---------- helpers -------------------------------------------------
    def sec_of(self, t):
        return float(np.searchsorted(self.times, t)) / FPS

    def pair(self, a, b):
        for p in self.ev["pairs"]:
            if {p["a"], p["b"]} == {a, b}:
                return p

    def crossing(self, body, what):
        for c in self.ev["crossings"]:
            if c["body"] == body and c["what"] == what:
                return c

    def build_captions(self):
        ev = self.ev
        v_e = ev["speed0_kms"][self.ie]
        me = self.pair("Merkuriy", "Yer")
        mj = self.pair("Mars", "Yupiter")
        mars = self.crossing("Yer", "Mars orbitasi")
        jup = self.crossing("Yer", "Yupiter orbitasi")
        oort = self.crossing("Yer", "Oort bulutining ichki chegarasi")
        dv_mars = ev["dv_ms"][self.names.index("Mars")]
        km = AU_KM / 1e6
        caps = [
            (T_VANISH, "Quyosh yoʻqoldi! Nyuton fizikasida uning tortishish kuchi shu zahoti yoʻqoladi."),
            (10.8, "8 daqiqa 20 soniya: Quyoshning soʻnggi nuri Yerga yetdi. Yerda abadiy tun boshlandi."),
            (14.4, "Sayyoralar endi orbita boʻylab emas — urinma boʻylab toʻgʻri chiziqda uchmoqda."),
            (17.4, f"1 kun: Yer {num(v_e * 86400 / 1e6, 1)} mln km yoʻl bosdi. Tezligi oʻzgarmadi: {num(v_e, 1)} km/s."),
            (21.7, "1 hafta: Yer yuzasining oʻrtacha harorati ≈ −18 °C gacha tushadi (taxminiy)."),
            (self.sec_of(me["t"]) - 0.2, f"{int(me['t'])}-kun: Merkuriy Yerdan {num(me['sep_au'] * km, 1)} mln km "
                                         f"masofadan oʻtdi — toʻqnashuv yoʻq."),
            (30.6, f"{int(mars['t'])}-kun: Yer Mars orbitasini kesib oʻtdi."),
            (33.2, f"{int(jup['t'])}-kun: Yer Yupiter orbitasini kesib oʻtdi."),
            (36.3, "1 yil: Yer harorati ≈ −73 °C. Okeanlar yuzasi muzlaydi."),
            (38.4, f"Mars Yupiter yonidan {num(mj['sep_au'] * km)} mln km da oʻtdi: tezligi "
                   f"{num(dv_mars)} m/s ga oʻzgardi."),
            (T_FAST, "Vaqt tezlashtirildi! Endi 1 soniyada yuzlab yillar oʻtadi."),
            (43.6, "Jismlar birin-ketin sobiq Quyosh tizimi chegarasini — geliopauzani (120 a.b.) — kesib oʻtmoqda."),
            (self.sec_of(oort["t"]), f"≈ {num(round(oort['t'] / YEAR, -1))} yil: Yer Oort bulutiga "
                                     f"(2 000 a.b.) kirdi."),
            (51.6, "Hech bir sayyora ortga qaytmaydi: tezliklari oʻzaro tortishish uchun juda katta."),
        ]
        caps.sort()
        out = []
        for k, (t0, s) in enumerate(caps):
            t1 = caps[k + 1][0] - 0.15 if k + 1 < len(caps) else T_FINALSTAGE - 0.2
            out.append((t0, min(t1, t0 + 5.0), s))
        return out

    # ---------- projection ---------------------------------------------
    def half_height(self, f, sec):
        if sec < T_VANISH:
            if sec < 1.5:
                return 33.0
            return math.exp(math.log(33) + (math.log(2) - math.log(33)) * smooth((sec - 1.5) / 3.5))
        f = min(f, int(T_FINALSTAGE * FPS) - 1)
        return max(2.0, 1.7 * float(np.linalg.norm(self.P[f, self.ie, :2])))

    def project(self, xy, sec, Hau):
        """xy: (...,2|3) AU -> screen px. Linear map, blended into a log-radial map in the final stage."""
        r3 = np.linalg.norm(xy, axis=-1)
        xy = xy[..., :2]
        k = MAP_R / Hau
        lin = np.stack([CX + xy[..., 0] * k, CY - xy[..., 1] * k], -1)
        b = smooth((sec - T_FINALSTAGE) / 2.5)
        if b <= 0:
            return lin
        r = np.linalg.norm(xy, axis=-1)
        rr = np.where(r > 0, r, 1)
        rho = LOG_R * np.clip(np.log10(np.maximum(r3, 1)) / LOGMAX, 0, 1)
        log = np.stack([LCX + xy[..., 0] / rr * rho, LCY - xy[..., 1] / rr * rho], -1)
        return lin * (1 - b) + log * b

    # ---------- drawing ------------------------------------------------
    def draw_rings(self, d, sec, Hau):
        b = smooth((sec - T_FINALSTAGE) / 2.5)
        rings = [(120, "Geliopauza · 120 a.b."), (2000, "Oort buluti boshlanishi · 2 000 a.b."),
                 (100_000, "Oort buluti chekkasi · 100 000 a.b.")]
        if b < 1:
            for R, label in rings:
                px = R * MAP_R / Hau
                a = (1 - b) * min(1, max(0, (px - 40) / 60)) * (1 if px < 1400 else 0)
                if a > 0:
                    d.ellipse((CX - px, CY - px, CX + px, CY + px), outline=(120, 140, 200, int(70 * a)), width=2)
                    text(d, (CX, CY + px + 6), label, 18, (140, 160, 210), a * 0.85, anchor="ma")
        if b > 0:
            rho = lambda R: LOG_R * math.log10(R) / LOGMAX
            r0, r1 = rho(2000), rho(100_000)
            for rr in np.linspace(r0, r1, 14):
                d.ellipse((LCX - rr, LCY - rr, LCX + rr, LCY + rr), outline=(70, 90, 150, int(14 * b)), width=int((r1 - r0) / 13) + 1)
            for R, lab in [(1, "1 a.b."), (10, "10"), (100, "100"), (1000, "1 000"), (10_000, "10 000"),
                           (100_000, "100 000 a.b.")]:
                px = rho(R)
                d.ellipse((LCX - px, LCY - px, LCX + px, LCY + px), outline=(110, 130, 180, int(60 * b)), width=1)
                text(d, (LCX + px * 0.707 + 4, LCY + px * 0.707 + 2), lab, 16, (130, 150, 200), b * 0.8)
            px = rho(268_000)
            d.ellipse((LCX - px, LCY - px, LCX + px, LCY + px), outline=(255, 170, 120, int(90 * b)), width=2)
            text(d, (LCX, LCY + px + 8), "Proksima Sentavr masofasi · 4,24 yorugʻlik yili", 18, (255, 185, 140), b * 0.9,
                 anchor="ma")
            px = rho(120)
            d.ellipse((LCX - px, LCY - px, LCX + px, LCY + px), outline=(150, 170, 230, int(90 * b)), width=2)
            text(d, (LCX, LCY - px - 4), "Geliopauza", 16, (150, 170, 230), b * 0.8, anchor="md")
            text(d, (LCX, LCY + rho(20000)), "OORT BULUTI", 20, (120, 140, 200), b * 0.55, "SemiBold", "mm")

    def draw_orbits(self, d, sec, Hau):
        a = 1 - smooth((sec - T_FAST) / 4)
        if a <= 0:
            return
        for i, path in enumerate(self.orbits):
            pts = self.project(path, sec, Hau)
            if np.abs(pts - (CX, CY)).max() < 6:
                continue
            col = tuple(int(c * 0.55) for c in self.colors[i])
            d.line([tuple(p) for p in pts], fill=(*col, int(110 * a)), width=1)

    def draw_trails(self, d, f, sec, Hau):
        start = max(0, f - 60) if f < self.f_vanish else max(0, self.f_vanish - 45)
        fe = min(f, int(T_FINALSTAGE * FPS))
        idx = np.unique(np.linspace(start, fe, min(fe - start + 1, 260)).astype(int))
        if len(idx) < 2:
            return
        a = 1 - smooth((sec - T_FINALSTAGE) / 1.2)
        if a <= 0:
            return
        pts = self.project(self.P[idx], sec, Hau)
        for i in range(self.n):
            col = tuple(int(c * 0.7) for c in self.colors[i])
            d.line([tuple(p) for p in pts[:, i]], fill=(*col, int(150 * a)), width=2)

    def draw_bodies(self, img, d, f, sec, Hau):
        pos = self.project(self.P[f], sec, Hau)
        final = sec >= T_FINALSTAGE
        labels = []
        for i in range(self.n):
            x, y = pos[i]
            if not (-50 < x < W + 50 and -50 < y < H + 50):
                continue
            sp = self.sprites[i]
            img.paste(sp, (int(x - sp.width / 2), int(y - sp.height / 2)), sp)
            r = self.rpx[i]
            if self.names[i] == "Saturn":
                d.ellipse((x - 2.3 * r, y - 0.75 * r, x + 2.3 * r, y + 0.75 * r), outline=(240, 220, 170, 200), width=2)
            lab = self.names[i]
            if final:
                lab = f"{lab} · {num(np.linalg.norm(self.P[f, i]))} a.b."
            labels.append((x, y, r, lab, i))
        self.place_labels(d, labels)
        return pos

    def place_labels(self, d, labels):
        """Greedy label placement: try several spots around each body, keep the first free one."""
        boxes = [(x - r * (2.3 if self.names[i] == "Saturn" else 1) - 2, y - r - 2,
                  x + r * (2.3 if self.names[i] == "Saturn" else 1) + 2, y + r + 2) for x, y, r, _, i in labels]
        prio = sorted(labels, key=lambda l: (self.names[l[4]] != "Yer", -self.rpx[l[4]]))
        for x, y, r, lab, i in prio:
            big = self.names[i] == "Yer"
            size = 21 if big else 18
            w = font(size, "SemiBold").getlength(lab)
            h = size + 4
            spots = [(x + r + 6, y - h / 2), (x - r - 6 - w, y - h / 2), (x - w / 2, y - r - h - 4),
                     (x - w / 2, y + r + 4), (x + r + 6, y - h - 8), (x + r + 6, y + 8),
                     (x - r - 6 - w, y - h - 8), (x - r - 6 - w, y + 8)]
            for lx, ly in spots + [spots[0]]:
                box = (lx - 2, ly - 2, lx + w + 2, ly + h + 2)
                if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in boxes):
                    break
            boxes.append(box)
            text(d, (lx, ly), lab, size, (140, 195, 255) if big else self.colors[i], 0.95,
                 "SemiBold" if big else "Medium")

    def draw_sun(self, img, d, sec):
        if sec < T_VANISH:
            pulse = 1 + 0.04 * math.sin(sec * 5)
            sp = self.sun.resize((int(self.sun.width * pulse),) * 2)
            img.paste(sp, (CX - sp.width // 2, CY - sp.height // 2), sp)
            return
        dt = sec - T_VANISH
        if dt < 1.6:
            r = 20 + dt * 420
            a = max(0, 1 - dt / 1.6)
            d.ellipse((CX - r, CY - r, CX + r, CY + r), outline=(255, 230, 160, int(200 * a)), width=int(6 * a) + 1)
            if dt < 0.35:
                flash = 1 - dt / 0.35
                d.rectangle((0, 0, W, H), fill=(255, 245, 220, int(150 * flash)))
                rr = 20 * (1 - dt / 0.35)
                d.ellipse((CX - rr, CY - rr, CX + rr, CY + rr), fill=(255, 250, 230, 255))
        a = min(1, dt / 1.5) * (1 - smooth((sec - T_FINALSTAGE) / 2))
        la = a * (1 - smooth((sec - T_FAST) / 1.5))
        if a > 0:
            c = (255, 190, 90, int(150 * a))
            d.line((CX - 9, CY, CX + 9, CY), fill=c, width=2)
            d.line((CX, CY - 9, CX, CY + 9), fill=c, width=2)
            text(d, (CX - 12, CY + 8), "Quyoshning sobiq oʻrni", 15, (255, 190, 110), la * 0.8, anchor="ra")

    def draw_sparks(self, d, f, sec, pos):
        for c in self.ev["collisions"]:
            dt = sec - c["sec"]
            if 0 <= dt < 1.5:
                rng = np.random.default_rng(int(c["t"] * 1000) % 2 ** 31)
                x, y = self.project(np.array(c["pos"][:2]), sec, self.half_height(f, sec))
                for _ in range(40):
                    ang, sp = rng.uniform(0, 2 * math.pi), rng.uniform(40, 220)
                    rr = sp * dt
                    a = 1 - dt / 1.5
                    px, py = x + rr * math.cos(ang), y + rr * math.sin(ang)
                    d.line((px, py, px - 6 * math.cos(ang), py - 6 * math.sin(ang)),
                           fill=(255, int(200 * a) + 50, 80, int(255 * a)), width=2)

    def draw_hud(self, d, f, sec, Hau):
        t = self.times[f]
        # title
        text(d, (40, 30), "Agar Quyosh birdan yoʻqolsa?", 46, (255, 225, 150), 1, "Bold")
        text(d, (42, 88), "Nyuton fizikasi · N-jism simulyatsiyasi · haqiqiy masofa va tezliklar", 21, (170, 185, 210))
        if sec < T_VANISH:
            phase = "Kirish: Quyosh tizimi odatdagidek · 2026-yil 25-oktabr"
        elif sec < T_FAST:
            phase = "1-bosqich · Dastlabki davr"
        elif sec < T_FINALSTAGE:
            phase = "2-bosqich · Vaqt tezlashtirildi »»"
        else:
            phase = "3-bosqich · Yakuniy holat"
        d.rounded_rectangle((40, 124, 52 + font(22, "SemiBold").getlength(phase) + 12, 160), 8,
                            fill=(40, 60, 110, 200))
        text(d, (52, 129), phase, 22, (225, 235, 255), 1, "SemiBold", shadow=False)

        # clock
        panel(d, (1460, 30, 1890, 176))
        if sec < T_VANISH:
            text(d, (1480, 44), "Quyosh yoʻqolishiga:", 20, (170, 185, 210))
            text(d, (1480, 72), f"{int(-t) + 1} kun", 40, (255, 225, 150), 1, "Bold")
        else:
            text(d, (1480, 44), "Quyosh yoʻqolganidan beri:", 20, (170, 185, 210))
            text(d, (1480, 72), fmt_elapsed(t), 36 if t < 100 * YEAR else 42, (255, 255, 255), 1, "Bold")
        rate = (sim_time(sec + 0.02) - sim_time(sec)) / 0.02
        if rate <= 0:
            rate_s = "vaqt toʻxtatildi"
        elif rate * 86400 < 1:
            rate_s = "Sekinlashtirilgan koʻrinish"
        else:
            rate_s = f"Video 1 soniya ≈ {fmt_rate(rate)}"
        text(d, (1480, 132), rate_s, 20, (150, 200, 255))

        # scale bar (linear stage)
        a = 1 - smooth((sec - T_FINALSTAGE) / 1.5)
        if a > 0:
            target = 220 * Hau / MAP_R
            p10 = 10 ** math.floor(math.log10(target))
            L = max(m * p10 for m in (1, 2, 5) if m * p10 <= target)
            px = L * MAP_R / Hau
            y = H - 50
            d.line((40, y, 40 + px, y), fill=(200, 210, 230, int(220 * a)), width=3)
            d.line((40, y - 7, 40, y + 7), fill=(200, 210, 230, int(220 * a)), width=3)
            d.line((40 + px, y - 7, 40 + px, y + 7), fill=(200, 210, 230, int(220 * a)), width=3)
            text(d, (40, y - 34), f"{num(L, 1 if L < 1 else 0)} a.b.  (1 a.b. = 149,6 mln km)", 18, (200, 210, 230), a)

    def draw_earth_panel(self, d, f, sec):
        a = fade(sec, T_VANISH + 0.5, T_FINALSTAGE + 0.6)
        if a <= 0:
            return
        t = self.times[f]
        p, v = self.P[f, self.ie], self.V[f, self.ie]
        r = float(np.linalg.norm(p))
        panel(d, (1460, 200, 1890, 480 if sec < T_FAST + 0.5 else 430), a, (70, 130, 220))
        text(d, (1480, 214), "YER", 26, (140, 195, 255), a, "Bold")
        text(d, (1480, 254), "Sobiq Quyoshdan masofa:", 18, (170, 185, 210), a)
        dist = f"{num(r, 2 if r < 100 else 0)} a.b."
        text(d, (1480, 278), dist, 30, (255, 255, 255), a, "SemiBold")
        sub = f"{num(r * AU_KM / 1e6)} mln km" if r < 1000 else f"{num(r / LY_AU, 3)} yorugʻlik yili"
        text(d, (1480, 316), sub, 18, (170, 185, 210), a)
        text(d, (1480, 348), f"Tezlik: {num(np.linalg.norm(v) * KMS, 2)} km/s", 22, (230, 235, 245), a)
        if sec < T_FAST + 0.5:
            temp = np.interp(math.log10(1 + max(t, 0)), [0, math.log10(8), math.log10(366)], [15, -18, -73])
            lab = f"Harorat: ≈ {num(temp)} °C" + (" va pasaymoqda" if t > 366 else "")
            text(d, (1480, 384), lab, 22, (160, 220, 255), a)
            text(d, (1480, 414), "(taxminiy, yuza oʻrtachasi)", 16, (140, 150, 170), a)
            text(d, (1480, 440), "Yorugʻlik: yoʻq" if t > 0.00579 else "Yorugʻlik: hali yetib kelmoqda",
                 18, (200, 200, 200), a)
        else:
            text(d, (1480, 384), "Oy: hamon Yer atrofida", 22, (220, 220, 220), a)

    def draw_moon_inset(self, d, sec):
        a = fade(sec, T_VANISH + 1.0, T_FAST + 0.3)
        if a <= 0:
            return
        t = sim_time(sec)
        box = (1460, 505, 1890, 815)
        panel(d, box, a, (110, 110, 140))
        text(d, (1480, 518), "YER – OY JUFTI", 22, (220, 225, 235), a, "Bold")
        cx, cy, R = 1675, 668, 95
        d.ellipse((cx - R, cy - R, cx + R, cy + R), outline=(150, 150, 170, int(90 * a)), width=1)
        d.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), fill=(70, 145, 255, int(255 * a)))
        rate = (sim_time(sec + 1 / FPS) - t) / 27.32 * 360
        if rate > 25:   # too fast to show as a dot: draw it as a blurred ring
            d.ellipse((cx - R, cy - R, cx + R, cy + R), outline=(200, 200, 200, int(150 * a)), width=4)
        else:
            ang = 2 * math.pi * t / 27.32 + 0.8
            mx, my = cx + R * math.cos(ang), cy - R * math.sin(ang)
            d.ellipse((mx - 5, my - 5, mx + 5, my + 5), fill=(215, 215, 215, int(255 * a)))
        e = self.ev["moon_energy"]
        text(d, (1480, 772), f"Oy bogʻlangan: energiya {num(e, 2)} km²/s² < 0", 17, (190, 200, 215), a)
        text(d, (1480, 792), "Quyoshsiz ham Oy Yerni tark etmaydi", 17, (190, 200, 215), a)

    def draw_captions(self, d, sec):
        for t0, t1, s in self.captions:
            a = fade(sec, t0, t1, 0.3)
            if a > 0:
                f = font(30, "SemiBold")
                wdt = f.getlength(s)
                x0 = max(30, CX - wdt / 2 - 24)
                d.rounded_rectangle((x0, H - 132, x0 + wdt + 48, H - 76), 12, fill=(0, 0, 0, int(170 * a)))
                text(d, (x0 + 24, H - 104), s, 30, (255, 240, 200), a, "SemiBold", "lm")

    def draw_exit_list(self, d, sec):
        a = fade(sec, T_FAST + 2.5, T_FINALSTAGE + 0.6)
        if a <= 0:
            return
        shown = [c for c in self.crossings if c["sec"] <= sec]
        panel(d, (40, 190, 420, 236 + 30 * len(self.crossings)), a * 0.9)
        text(d, (58, 202), "Geliopauzadan chiqqanlar:", 20, (200, 215, 240), a, "SemiBold")
        for k, c in enumerate(shown):
            i = self.names.index(c["body"])
            flash = 1 + max(0, 1 - (sec - c["sec"]) * 2)
            col = tuple(min(255, int(x * flash)) for x in self.colors[i])
            text(d, (58, 236 + 30 * k), c["body"], 20, col, a)
            text(d, (400, 236 + 30 * k), f"{num(c['t'] / YEAR)} yil", 20, (220, 225, 235), a, anchor="ra")

    def draw_final(self, d, f, sec, pos):
        ev = self.ev
        x0, x1 = 1310, 1890
        # 1) Where is Earth?
        a = fade(sec, T_FINALSTAGE + 2.5, 67.0)
        if a > 0:
            p = self.P[f, self.ie]
            r = float(np.linalg.norm(p))
            pulse = 18 + 10 * (0.5 + 0.5 * math.sin(sec * 4))
            ex, ey = pos[self.ie]
            d.ellipse((ex - pulse, ey - pulse, ex + pulse, ey + pulse), outline=(120, 190, 255, int(220 * a)), width=3)
            panel(d, (x0, 470, x1, 870), a, (70, 130, 220))
            y = 486
            text(d, (x0 + 22, y), "YER QAYERDA?", 30, (140, 195, 255), a, "Bold")
            rows = [
                (f"Sobiq Quyosh oʻrnidan: {num(r)} a.b.", 24, (255, 255, 255)),
                (f"= {num(r * AU_KM / 1e12, 2)} trln km ≈ {num(r / LY_AU, 2)} yorugʻlik yili", 21, (200, 210, 230)),
                (f"Tezlik: {num(np.linalg.norm(self.V[f, self.ie]) * KMS, 2)} km/s", 22, (230, 235, 245)),
                (f"Sayyoralar tortishishi uni atigi {num(ev['dv_ms'][self.ie], 1)} m/s ga oʻzgartirdi", 19,
                 (200, 210, 230)),
                ("Hamrohi: Oy — hamon Yer atrofida aylanmoqda", 21, (220, 220, 220)),
                ("Holati: muzlagan, qorongʻi «daydi sayyora»", 21, (160, 220, 255)),
                ("Hayot faqat okean tubidagi issiq manbalarda", 19, (200, 210, 230)),
                ("saqlanib qolishi mumkin", 19, (200, 210, 230)),
            ]
            y += 50
            for s, sz, c in rows:
                text(d, (x0 + 22, y), s, sz, c, a)
                y += sz + 16
        # 2) Table of objects that left the system
        a = fade(sec, 67.2, 79.4)
        if a > 0:
            panel(d, (x0, 200, x1, 960), a, (200, 150, 90))
            text(d, (x0 + 20, 214), f"TIZIMNI TARK ETGAN JISMLAR: {sum(ev['unbound'].values())} / {self.n}", 25,
                 (255, 205, 140), a, "Bold")
            text(d, (x0 + 20, 250), "Barchasi gravitatsion bogʻlanmagan (umumiy energiya > 0)", 17,
                 (200, 200, 210), a)
            hx = [x0 + 20, x0 + 285, x0 + 415, x1 - 22]
            y = 286
            for s, xx, anc in zip(["Jism", "Masofa, a.b.", "Tezlik", "Geliopauza"], hx, ["la", "ra", "ra", "ra"]):
                text(d, (xx, y), s, 18, (170, 185, 210), a, "SemiBold", anc)
            hp = {c["body"]: c["t"] for c in self.crossings}
            order = np.argsort(-np.linalg.norm(self.P[f, :, :2], axis=1))
            for k, i in enumerate(order):
                y = 320 + 46 * k
                nm = self.names[i]
                r = float(np.linalg.norm(self.P[f, i]))
                vis = fade(sec, 67.6 + 0.25 * k, 79.4)
                text(d, (hx[0], y), nm, 21, self.colors[i], vis, "SemiBold")
                text(d, (hx[1], y), num(r), 21, (240, 240, 245), vis, anchor="ra")
                text(d, (hx[2], y), f"{num(np.linalg.norm(self.V[f, i]) * KMS, 1)} km/s", 21,
                     (240, 240, 245), vis, anchor="ra")
                text(d, (hx[3], y), f"{num(hp[nm] / YEAR)}-yil" if nm in hp else "—", 21, (255, 205, 140), vis,
                     anchor="ra")
        # 3) Conclusions
        a = fade(sec, 79.6, 90.5)
        if a > 0:
            cl = ev["closest"]
            panel(d, (x0, 260, x1, 900), a, (150, 120, 220))
            y = 280
            text(d, (x0 + 20, y), "XULOSA", 30, (210, 190, 255), a, "Bold")
            n_col = len(ev["collisions"])
            rows = [
                (f"Toʻqnashuvlar soni: {n_col}", 24, (255, 255, 255), "SemiBold"),
                (f"Eng yaqin yaqinlashuv: {cl['a']} va {cl['b']}", 20, (220, 225, 235), "Medium"),
                (f"{num(cl['sep_au'] * AU_KM / 1e6, 1)} mln km, {int(cl['t'])}-kunda", 20, (220, 225, 235), "Medium"),
                ("", 8, None, None),
                ("Nega yagona katta jism hosil boʻlmadi?", 22, (255, 220, 150), "SemiBold"),
                ("Sayyoralar bir-biridan km/s tezlikda uzoqlashadi,", 19, (220, 225, 235), "Medium"),
                ("oʻzaro tortishish esa ularni bor-yoʻgʻi bir necha", 19, (220, 225, 235), "Medium"),
                ("m/s ga burishi mumkin. Ularni Quyosh ushlab turardi.", 19, (220, 225, 235), "Medium"),
                ("", 8, None, None),
                ("Yupiter va Saturn yoʻldoshlari hamda halqalarini", 19, (220, 225, 235), "Medium"),
                ("oʻzi bilan olib ketdi — ular ham bogʻlanganicha qoladi.", 19, (220, 225, 235), "Medium"),
                ("", 8, None, None),
                ("Quyoshsiz Quyosh tizimi — yulduzlararo fazoda", 22, (160, 220, 255), "SemiBold"),
                ("tarqalib ketayotgan qorongʻi, muzlagan olamlar.", 22, (160, 220, 255), "SemiBold"),
            ]
            y += 54
            for s, sz, c, wgt in rows:
                if s:
                    text(d, (x0 + 20, y), s, sz, c, a, wgt)
                y += sz + 14
        a = fade(sec, 82, 91)
        text(d, (1890, H - 36), f"Hisob: Nyuton tortishish qonuni · {self.n} jism · {num(ev['steps'])} integrallash "
                                f"qadami · leapfrog", 16, (130, 140, 160), a, anchor="ra")

    def frame(self, f):
        sec = f / FPS
        img = self.bg.copy()
        d = ImageDraw.Draw(img, "RGBA")
        Hau = self.half_height(f, sec)
        self.draw_rings(d, sec, Hau)
        self.draw_orbits(d, sec, Hau)
        self.draw_trails(d, f, sec, Hau)
        self.draw_sun(img, d, sec)
        pos = self.draw_bodies(img, d, f, sec, Hau)
        self.draw_sparks(d, f, sec, pos)
        self.draw_hud(d, f, sec, Hau)
        self.draw_earth_panel(d, f, sec)
        self.draw_moon_inset(d, sec)
        self.draw_exit_list(d, sec)
        self.draw_captions(d, sec)
        if sec >= T_FINALSTAGE:
            self.draw_final(d, f, sec, pos)
            banner = fade(sec, T_FINALSTAGE, T_FINALSTAGE + 3.0)
            if banner > 0:
                s = "10 000 yildan keyin: Yer va boshqa sayyoralar qayerda?"
                d.rounded_rectangle((CX - 470, 190, CX + 470, 250), 12, fill=(0, 0, 0, int(170 * banner)))
                text(d, (CX, 220), s, 30, (255, 225, 150), banner, "SemiBold", "mm")
            text(d, (40, 190), "Logarifmik masofa shkalasi", 18, (150, 165, 200), smooth((sec - 56) / 2) * 0.9)
        # global fade in / out
        g = min(1.0, sec / 0.6, (90 - sec) / 1.5)
        if g < 1:
            d.rectangle((0, 0, W, H), fill=(0, 0, 0, int(255 * (1 - max(g, 0)))))
        return img


LOGMAX = math.log10(300_000)


def main():
    r = Renderer()
    with open("cues.json", "w") as fh:
        json.dump(r.cues, fh, indent=1)
    if len(sys.argv) > 1 and sys.argv[1] == "--stills":
        for s in sys.argv[2:]:
            r.frame(int(float(s) * FPS)).save(f"still_{s}.png")
        return
    out = sys.argv[1] if len(sys.argv) > 1 else "video_silent.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "21", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for f in range(N_FRAMES):
        ff.stdin.write(r.frame(f).tobytes())
        if f % 300 == 0:
            print(f"frame {f}/{N_FRAMES}", flush=True)
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    main()
