#!/bin/sh
# Builds quyosh_yoqolsa.mp4 (90 s, 1920x1080, 30 fps) from scratch.
# Requires: python3 with numpy + pillow, ffmpeg with libx264, Inter font.
set -e
cd "$(dirname "$0")"
python3 sim.py                       # N-body physics -> frames.npz, events.json
python3 render.py video_silent.mp4   # frames -> silent video, cues.json
python3 audio.py                     # cues -> soundtrack.wav
ffmpeg -y -loglevel error -i video_silent.mp4 -i soundtrack.wav \
       -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart quyosh_yoqolsa.mp4
echo "done: quyosh_yoqolsa.mp4"
