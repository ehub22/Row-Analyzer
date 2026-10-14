"""Generate SYNTHETIC H.264 test clips. Never real footage, never a benchmark.

The analysis pipeline needs a decoded-video fixture: WebCodecs only accepts a
real container, and the repository ships no media (see .gitignore) because
athlete imagery is never committed. This writes a small clip of moving colour
blocks that is enough to exercise decode → detect → pose → track end to end
with the constant ONNX doubles from make_test_models.py.

    python -m venv .venv && .venv/bin/pip install av numpy
    .venv/bin/python scripts/make_test_clip.py --frames 40 --out private/test-models/short40.mp4

Requires PyAV (bundled FFmpeg, libx264). Output lands under private/, which is
gitignored, so nothing here reaches the repository or the Pages build.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import av
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--people", type=int, default=8, help="drifting blobs")
    parser.add_argument("--out", type=Path, default=Path("private/test-models/short.mp4"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(args.out), "w")
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width, stream.height, stream.pix_fmt = args.width, args.height, "yuv420p"
    stream.options.update(
        {
            "preset": "veryfast",
            "crf": "30",
            "movflags": "+faststart",
            # A keyframe every ~0.5 s keeps frame-accurate stepping meaningful.
            "x264-params": f"keyint={max(1, args.fps // 2)}:min-keyint=1:scenecut=0",
        }
    )

    ys, xs = np.mgrid[0 : args.height, 0 : args.width]
    for i in range(args.frames):
        image = np.full((args.height, args.width, 3), 24, np.uint8)
        for p in range(args.people):
            cx = 40 + p * 70 + int(12 * np.sin(i / 6 + p))
            cy = args.height // 2 + int(6 * np.cos(i / 5 + p))
            image += (np.abs(xs - cx) < 12)[..., None].astype(np.uint8) * np.array([200, 60, 40], np.uint8)
            image += (np.abs(ys - cy) < 26)[..., None].astype(np.uint8) * np.array([40, 90, 190], np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes, {args.frames} frames)")


if __name__ == "__main__":
    main()
