"""Video timeline: maps video frames to simulation time (days since the Sun vanished)."""
import numpy as np

FPS = 30
DURATION = 90.0
N_FRAMES = int(FPS * DURATION)

T_VANISH = 6.0        # s   intro with the Sun still present (Keplerian orbits)
T_FAST = 40.0         # s   end of the early phase, start of the speed-up
T_FINALSTAGE = 55.0   # s   end of the speed-up, final stage until 90 s

PRE_DAYS = 120.0                  # intro shows the last 120 days of normal orbits
T_EARLY = 2 * 365.25              # early phase covers the first 2 years
T_FINAL = 10_000 * 365.25         # the speed-up reaches 10 000 years
EARLY_POWER = 6.0                 # t ~ u^6: first seconds show minutes, last seconds show months


def sim_time(sec):
    if sec < T_VANISH:
        return -PRE_DAYS * (1 - sec / T_VANISH)
    if sec < T_FAST:
        u = (sec - T_VANISH) / (T_FAST - T_VANISH)
        return T_EARLY * u ** EARLY_POWER
    if sec < T_FINALSTAGE:
        u = (sec - T_FAST) / (T_FINALSTAGE - T_FAST)
        return T_EARLY * (T_FINAL / T_EARLY) ** u
    return T_FINAL


def frame_times():
    return np.array([sim_time(f / FPS) for f in range(N_FRAMES)])
