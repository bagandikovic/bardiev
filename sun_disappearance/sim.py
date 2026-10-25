"""Newtonian N-body simulation: what happens to the Solar System if the Sun vanishes.

Units: AU, days, GM in AU^3/day^2.  Frame: heliocentric ecliptic J2000
(the Sun's own barycentric speed, ~13 m/s, is negligible next to km/s orbital speeds).

Initial state: real orbital elements (JPL approximate Keplerian elements, Standish)
propagated to the date the Sun disappears.  Before t=0 the bodies move on their
Keplerian orbits around the Sun; at t=0 the Sun is removed and the bodies evolve
only under their mutual gravity (full N-body, adaptive leapfrog).

Output: frames.npz (positions/velocities for every video frame) and events.json.
"""
import json
import numpy as np

from timeline import FPS, frame_times, T_FINAL

K = 0.01720209895                 # Gaussian gravitational constant
GM_SUN = K * K                    # AU^3/day^2
AU_KM = 149_597_870.7
DAY_S = 86_400.0
KMS = AU_KM / DAY_S               # 1 AU/day in km/s
JD_VANISH = 2461338.5             # 2026-10-25 00:00 TT
JD_J2000 = 2451545.0

# name, GM / GM_sun, radius [km], colour
BODIES = [
    ("Merkuriy", 1.6601e-7, 2440, (175, 170, 165)),
    ("Venera", 2.4478e-6, 6052, (235, 205, 140)),
    ("Yer", 3.0404e-6, 6371, (70, 145, 255)),          # Earth+Moon barycentre mass
    ("Mars", 3.2272e-7, 3390, (225, 105, 65)),
    ("Yupiter", 9.5479e-4, 69911, (225, 185, 135)),
    ("Saturn", 2.8589e-4, 58232, (235, 215, 160)),
    ("Uran", 4.3662e-5, 25362, (150, 225, 235)),
    ("Neptun", 5.1514e-5, 24622, (85, 125, 245)),
    ("Pluton", 7.40e-9, 1188, (205, 185, 165)),
    ("Serera", 4.72e-10, 470, (165, 165, 155)),
    ("Vesta", 1.30e-10, 263, (190, 175, 150)),
    ("Galley kometasi", 1.1e-16, 5.5, (200, 235, 255)),
]

# JPL approximate elements (1800-2050): a, e, I, L, long.peri, long.node  + rates per century
PLANET_ELEMENTS = {
    "Merkuriy": ([0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593],
                 [0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081]),
    "Venera": ([0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255],
               [0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418]),
    "Yer": ([1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0],
            [0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0]),
    "Mars": ([1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891],
             [0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343]),
    "Yupiter": ([5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909],
                [-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106]),
    "Saturn": ([9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448],
               [-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794]),
    "Uran": ([19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503],
             [-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589]),
    "Neptun": ([30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574],
               [0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664]),
    "Pluton": ([39.48211675, 0.24882730, 17.14001206, 238.92903833, 224.06891629, 110.30393684],
               [-0.00031596, 0.00005170, 0.00004818, 145.20780515, -0.04062942, -0.01183482]),
}
# small bodies: a, e, i, node, arg.peri, mean anomaly at epoch, epoch JD
SMALL_ELEMENTS = {
    "Serera": (2.7691, 0.0760, 10.594, 80.305, 73.597, 77.372, 2459000.5),
    "Vesta": (2.3618, 0.0887, 7.142, 103.851, 150.728, 205.0, 2459000.5),       # phase approximate
    "Galley kometasi": (17.834, 0.96714, 162.26, 58.42, 111.33, 0.0, 2446470.5),  # perihelion 1986-02-09
}

MOON_DIST_KM = 384_400.0
MOON_V_KMS = 1.022
GM_EARTH_MOON_KM = 403_503.0      # km^3/s^2


def kepler_state(a, e, inc, node, argp, M, gm=GM_SUN):
    """Position/velocity from Keplerian elements (angles in degrees)."""
    inc, node, argp, M = np.radians([inc, node, argp, M])
    E = M if e < 0.8 else np.pi
    for _ in range(100):
        E -= (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    n = np.sqrt(gm / a ** 3)
    x, y = a * (np.cos(E) - e), a * np.sqrt(1 - e * e) * np.sin(E)
    edot = n / (1 - e * np.cos(E))
    vx, vy = -a * np.sin(E) * edot, a * np.sqrt(1 - e * e) * np.cos(E) * edot
    cO, sO, cw, sw, ci, si = np.cos(node), np.sin(node), np.cos(argp), np.sin(argp), np.cos(inc), np.sin(inc)
    R = np.array([[cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci],
                  [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci],
                  [sw * si, cw * si]])
    return R @ [x, y], R @ [vx, vy]


def heliocentric_state(name, jd):
    if name in PLANET_ELEMENTS:
        el, rate = PLANET_ELEMENTS[name]
        T = (jd - JD_J2000) / 36525.0
        a, e, inc, L, wbar, node = np.array(el) + T * np.array(rate)
        return kepler_state(a, e, inc, node, wbar - node, (L - wbar) % 360)
    a, e, inc, node, argp, M0, ep = SMALL_ELEMENTS[name]
    M = M0 + np.degrees(np.sqrt(GM_SUN / a ** 3) * (jd - ep))
    return kepler_state(a, e, inc, node, argp, M % 360)


def accelerations(pos, gm):
    d = pos[None, :, :] - pos[:, None, :]
    r2 = (d * d).sum(-1)
    np.fill_diagonal(r2, np.inf)
    return (gm[None, :, None] * d / r2[..., None] ** 1.5).sum(1)


def main():
    names = [b[0] for b in BODIES]
    gm = np.array([b[1] for b in BODIES]) * GM_SUN
    radius = np.array([b[2] for b in BODIES]) / AU_KM
    n = len(BODIES)
    times = frame_times()          # days relative to the vanishing, one per frame
    nf = len(times)
    P = np.zeros((nf, n, 3))
    V = np.zeros((nf, n, 3))

    # before t=0: Keplerian motion around the Sun
    pre = np.where(times < 0)[0]
    for f in pre:
        for i, nm in enumerate(names):
            P[f, i], V[f, i] = heliocentric_state(nm, JD_VANISH + times[f])

    pos = np.array([heliocentric_state(nm, JD_VANISH)[0] for nm in names])
    vel = np.array([heliocentric_state(nm, JD_VANISH)[1] for nm in names])
    v0 = vel.copy()
    alive = np.ones(n, bool)
    t = 0.0
    acc = accelerations(pos, gm)

    iu, ju = np.triu_indices(n, 1)
    min_sep = np.full(len(iu), np.inf)
    min_sep_t = np.zeros(len(iu))
    collisions = []
    cross_radii = {"Mars orbitasi": 1.524, "Yupiter orbitasi": 5.203, "Saturn orbitasi": 9.537,
                   "Neptun orbitasi": 30.07, "Geliopauza": 120.0, "Oort bulutining ichki chegarasi": 2000.0}
    crossings = []
    crossed = set()
    steps = 0

    def record(p_old, p_new, t_old, h):
        nonlocal pos, vel, gm, alive
        # closest approach of every pair within the step, assuming straight-line motion
        dp = p_old[ju] - p_old[iu]
        dv = (p_new[ju] - p_new[iu]) - dp
        s = np.clip(-(dp * dv).sum(1) / np.maximum((dv * dv).sum(1), 1e-300), 0, 1)
        sep = np.linalg.norm(dp + s[:, None] * dv, axis=1)
        live = alive[iu] & alive[ju]
        better = (sep < min_sep) & live
        min_sep[better] = sep[better]
        min_sep_t[better] = t_old + s[better] * h
        for k in np.where(live & (sep < radius[iu] + radius[ju]))[0]:
            i, j = iu[k], ju[k]
            m = gm[i] + gm[j]
            pos[i] = (gm[i] * pos[i] + gm[j] * pos[j]) / m
            vel[i] = (gm[i] * vel[i] + gm[j] * vel[j]) / m
            radius[i] = (radius[i] ** 3 + radius[j] ** 3) ** (1 / 3)
            gm[i], gm[j] = m, 0.0
            alive[j] = False
            collisions.append({"t": t_old + s[k] * h, "a": names[i], "b": names[j], "pos": pos[i].tolist()})
        r_old, r_new = np.linalg.norm(p_old, axis=1), np.linalg.norm(p_new, axis=1)
        for label, R in cross_radii.items():
            for i in np.where((r_old < R) & (r_new >= R) & alive)[0]:
                if (i, label) not in crossed and not label.startswith(names[i]):
                    crossed.add((i, label))
                    tc = t_old + h * (R - r_old[i]) / (r_new[i] - r_old[i])
                    crossings.append({"t": tc, "body": names[i], "what": label, "r": R})

    for f in range(nf):
        tgt = times[f]
        if tgt < 0:
            continue
        while t < tgt - 1e-12:
            d = pos[None] - pos[:, None]
            w = vel[None] - vel[:, None]
            r = np.linalg.norm(d, axis=-1)
            u = np.linalg.norm(w, axis=-1) + 1e-30
            mask = alive[None] & alive[:, None]
            np.fill_diagonal(mask, False)
            h = np.clip(0.01 * (r / u)[mask].min(), 0.02, 50.0)
            h = min(h, tgt - t)
            p_old = pos.copy()
            vel += 0.5 * h * acc
            pos += h * vel
            acc = accelerations(pos, gm)
            vel += 0.5 * h * acc
            record(p_old, pos.copy(), t, h)
            t += h
            steps += 1
        P[f], V[f] = pos, vel

    # gravitational binding: energy of each body w.r.t. the centre of mass of all the others
    unbound = {}
    for i in range(n):
        others = [j for j in range(n) if j != i and alive[j]]
        M = gm[others].sum()
        cm = (gm[others, None] * pos[others]).sum(0) / M
        cv = (gm[others, None] * vel[others]).sum(0) / M
        e = 0.5 * np.sum((vel[i] - cv) ** 2) - M / np.linalg.norm(pos[i] - cm)
        unbound[names[i]] = bool(e > 0)

    k = int(np.argmin(min_sep))
    # Earth-Moon: unaffected by the Sun's disappearance at first order
    e_moon = 0.5 * MOON_V_KMS ** 2 - GM_EARTH_MOON_KM / MOON_DIST_KM
    events = {
        "names": names,
        "colors": [b[3] for b in BODIES],
        "radius_km": [b[2] for b in BODIES],
        "fps": FPS,
        "t_final_days": T_FINAL,
        "steps": steps,
        "collisions": collisions,
        "crossings": sorted(crossings, key=lambda c: c["t"]),
        "closest": {"a": names[iu[k]], "b": names[ju[k]], "sep_au": float(min_sep[k]), "t": float(min_sep_t[k])},
        "pairs": [{"a": names[i], "b": names[j], "sep_au": float(s), "t": float(ts)}
                  for i, j, s, ts in zip(iu, ju, min_sep, min_sep_t)],
        "unbound": unbound,
        "speed0_kms": (np.linalg.norm(v0, axis=1) * KMS).tolist(),
        "dv_ms": (np.linalg.norm(vel - v0, axis=1) * KMS * 1000).tolist(),
        "moon_energy": e_moon,
        "moon_bound": bool(e_moon < 0),
        "alive": alive.tolist(),
    }
    np.savez_compressed("frames.npz", times=times, P=P, V=V)
    with open("events.json", "w") as fh:
        json.dump(events, fh, indent=1, ensure_ascii=False)
    print(f"steps={steps} collisions={len(collisions)} closest={events['closest']}")
    for c in events["crossings"]:
        print(f"  {c['t'] / 365.25:10.3f} yr  {c['body']:16s} {c['what']}")
    for i, nm in enumerate(names):
        print(f"  {nm:16s} r={np.linalg.norm(pos[i]):10.0f} AU  v0={events['speed0_kms'][i]:6.2f} km/s "
              f"dv={events['dv_ms'][i]:8.3f} m/s unbound={unbound[nm]}")


if __name__ == "__main__":
    main()
