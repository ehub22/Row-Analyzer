"""Generate CONSTANT TEST DOUBLES, never real pose estimators. Ignored private path.

The doubles let the whole pipeline run without weights: a YOLOX-grid detector
output and a 133-joint SimCC pair, both constant graphs. They prove plumbing
(decode → detect → crop → pose → tracker → metrics), never accuracy.

    python -m venv .venv && .venv/bin/pip install onnx numpy
    .venv/bin/python scripts/make_test_models.py                 # one person
    .venv/bin/python scripts/make_test_models.py --people 8      # eight seats

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
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()
    root: Path = args.out
    root.mkdir(parents=True, exist_ok=True)

    if not 1 <= args.people <= 8:
        raise SystemExit("--people must be between 1 and 8 (a boat cannot exceed 8 seats)")

    detector = np.zeros((1, 8400, 85), dtype=np.float32)
    for i in range(args.people):
        # Stride-8 cell offsets: cx/cy within the cell, log(w/h) relative to it.
        cell_x, cell_y = 6 + i * 9, 30
        offset = cell_y * 80 + cell_x
        detector[0, offset, :6] = [0.5, 0.5, np.log(96 / 8), np.log(280 / 8), 0.99, 0.99]
    detector_name = "TEST-ONLY-detector.onnx" if args.people == 1 else f"TEST-ONLY-detector{args.people}.onnx"
    write(root, detector_name, [1, 3, 640, 640], {"output": detector})

    x = np.zeros((1, 133, 576), dtype=np.float32)
    y = np.zeros((1, 133, 768), dtype=np.float32)
    for j in range(133):
        x[0, j, 250 + j % 20] = 0.95
        y[0, j, 300 + j % 30] = 0.95
    write(root, "TEST-ONLY-pose.onnx", [1, 3, 384, 288], {"simcc_x": x, "simcc_y": y})
    print("Generated test doubles in", root)


if __name__ == "__main__":
    main()
