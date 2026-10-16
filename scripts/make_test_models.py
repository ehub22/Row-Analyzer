"""Generate CONSTANT TEST DOUBLES, never real pose estimators. Ignored private path.

The doubles let the whole pipeline run without weights: a YOLOX-grid detector
output and a 17- or 133-joint SimCC pair, both constant graphs. They prove plumbing
(decode → detect → crop → pose → tracker → metrics), never accuracy.

    python -m venv .venv && .venv/bin/pip install onnx numpy
    .venv/bin/python scripts/make_test_models.py                 # one person
    .venv/bin/python scripts/make_test_models.py --people 8      # eight seats
    .venv/bin/python scripts/make_test_models.py --people 4 --joints 17  # body-only

`--people N` places N well-separated detections so the pose-worker pool and the
seat tracker can be exercised at full width (an eight is 8 crops × 4 TTA passes
per frame), which is what scripts/worker-check.mjs and the timing checks use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path("private/test-models")


def write(root: Path, name: str, shape: list[int], outputs: dict[str, np.ndarray]) -> None:
    nodes, infos = [], []
    for key, value in outputs.items():
        nodes.append(helper.make_node("Constant", [], [key], value=numpy_helper.from_array(value)))
        infos.append(helper.make_tensor_value_info(key, TensorProto.FLOAT, list(value.shape)))
    graph = helper.make_graph(
        nodes,
        "TEST_DOUBLE_NOT_TRAINED",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        infos,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, root / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people", type=int, default=1, help="number of detections (1-8)")
    parser.add_argument("--joints", type=int, choices=(17, 133), default=133)
    parser.add_argument("--format", choices=("simcc", "heatmap"), default="simcc")
    parser.add_argument("--detector-size", type=int, choices=(416, 640), default=640)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()
    root: Path = args.out
    root.mkdir(parents=True, exist_ok=True)
    width = args.width or (192 if args.joints == 17 else 288)
    height = args.height or (256 if args.joints == 17 else 384)

    if not 1 <= args.people <= 8:
        raise SystemExit("--people must be between 1 and 8 (a boat cannot exceed 8 seats)")

    size = args.detector_size
    grid = size // 8
    count = sum((size // stride) ** 2 for stride in (8, 16, 32))
    detector = np.zeros((1, count, 85), dtype=np.float32)
    step = 9 if size == 640 else max(4, (grid - 12) // max(1, args.people - 1))
    for i in range(args.people):
        # Stride-8 cell offsets: cx/cy within the cell, log(w/h) relative to it.
        cell_x, cell_y = 6 + i * step, min(30 if size == 640 else 18, grid - 1)
        offset = cell_y * grid + cell_x
        detector[0, offset, :6] = [0.5, 0.5, np.log(96 / 8), np.log(280 / 8), 0.99, 0.99]
    detector_name = "TEST-ONLY-detector.onnx" if args.people == 1 else f"TEST-ONLY-detector{args.people}.onnx"
    if size != 640:
        detector_name = detector_name.replace(".onnx", f"-{size}.onnx")
    write(root, detector_name, [1, 3, size, size], {"output": detector})

    pose_name = "TEST-ONLY-pose17.onnx" if args.joints == 17 else "TEST-ONLY-pose.onnx"
    if args.format == "simcc":
        x = np.zeros((1, args.joints, width * 2), dtype=np.float32)
        y = np.zeros((1, args.joints, height * 2), dtype=np.float32)
        for j in range(args.joints):
            x[0, j, min(width * 2 - 1, 250 + j % 20)] = 0.95
            y[0, j, min(height * 2 - 1, 300 + j % 30)] = 0.95
        outputs = {"simcc_x": x, "simcc_y": y}
    else:
        heat = np.zeros((1, args.joints, height // 4, width // 4), dtype=np.float32)
        for j in range(args.joints):
            heat[0, j, height // 8, min(width // 4 - 1, width // 8 + j % 4)] = 0.95
        outputs = {"heatmap": heat}
        pose_name = pose_name.replace(".onnx", "-heatmap.onnx")
    write(root, pose_name, [1, 3, height, width], outputs)
    print("Generated test doubles in", root)


if __name__ == "__main__":
    main()
