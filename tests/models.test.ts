import { describe, it, expect } from "vitest";
import { validateInput, validateOutputs } from "../src/core/modelContract";
import { defaultModels } from "../src/core/types";
import { verifyModel } from "../src/core/models";
import { decodeYolox } from "../src/core/detection";
describe("model contracts fail before video decoding", () => {
  it("accepts the exact reference contracts", () => {
    expect(
      validateInput(
        [{ isTensor: true, type: "float32", shape: [1, 3, 640, 640] }],
        "detector",
        defaultModels,
      ),
    ).toBe(640);
    validateOutputs(
      [
        { dims: [1, 133, 576], data: new Float32Array(133 * 576) },
        { dims: [1, 133, 768], data: new Float32Array(133 * 768) },
      ],
      "pose",
      defaultModels,
    );
  });
  it("rejects wrong channel layout/type/batch/size", () => {
    for (const shape of [
      [1, 640, 640, 3],
      [2, 3, 640, 640],
      [1, 3, 640, 320],
    ])
      expect(() =>
        validateInput(
          [{ isTensor: true, type: "float32", shape }],
          "detector",
          defaultModels,
        ),
      ).toThrow();
    expect(() =>
      validateInput(
        [{ isTensor: true, type: "float16", shape: [1, 3, 384, 288] }],
        "pose",
        defaultModels,
      ),
    ).toThrow();
    expect(() =>
      validateInput(
        [{ isTensor: true, type: "float32", shape: [1, 3, 256, 192] }],
        "pose",
        defaultModels,
      ),
    ).toThrow();
  });
  it("rejects body-only, swapped axes and nonfinite tensors", () => {
    for (const dims of [
      [1, 17, 576],
      [1, 133, 768],
    ])
      expect(() =>
        validateOutputs(
          [{ dims, data: new Float32Array(2) }],
          "pose",
          defaultModels,
        ),
      ).toThrow();
    expect(() =>
      validateOutputs(
        [{ dims: [1, 8400, 85], data: new Float32Array([NaN]) }],
        "detector",
        defaultModels,
      ),
    ).toThrow();
    expect(() =>
      decodeYolox(new Float32Array(85), [1, 1, 85], 640, 1),
    ).toThrow();
  });
  it("rejects HTML, LFS pointers, truncation and wrong checksums", async () => {
    await expect(
      verifyModel(new Blob(["<html>not a model</html>"])),
    ).rejects.toThrow("HTML");
    await expect(
      verifyModel(new Blob(["version https://git-lfs.github.com/spec/v1"])),
    ).rejects.toThrow("pointer");
    await expect(
      verifyModel(new Blob(["12345678"]), {
        id: "pose",
        name: "x.onnx",
        bytes: 8,
        sha256: "bad",
      }),
    ).rejects.toThrow("mismatch");
    await expect(verifyModel(new Blob(["12345678"]))).resolves.toMatch(
      /^[a-f0-9]{64}$/,
    );
  });
});
