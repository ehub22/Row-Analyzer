import type { ModelSettings } from "./types";
type Meta = {
  isTensor: boolean;
  shape?: readonly (number | string)[];
  type?: string;
};
export function validateInput(
  meta: readonly Meta[],
  kind: "detector" | "pose",
  settings: ModelSettings,
) {
  const m = meta[0],
    shape = m?.shape;
  if (
    meta.length !== 1 ||
    !m.isTensor ||
    m.type !== "float32" ||
    !shape ||
    shape.length !== 4
  )
    throw Error(
      `${kind}: expected one float32 NCHW input. External-data and body-only exports are not supported.`,
    );
  const size = typeof shape[2] === "number" ? shape[2] : 640;
  const expected =
    kind === "detector"
      ? [1, 3, size, size]
      : [1, 3, settings.height, settings.width];
  if (
    shape.some((d, i) => typeof d === "number" && d !== expected[i]) ||
    size < 32 ||
    size > 2048 ||
    (kind === "detector" && size % 32 !== 0)
  )
    throw Error(
      `${kind}: input [${shape}] does not match [${expected}]. Check model and processing settings.`,
    );
  return kind === "detector" ? size : settings.width;
}
export function validateOutputs(
  outputs: readonly { dims: readonly number[]; data: unknown }[],
  kind: "detector" | "pose",
  settings: ModelSettings,
  size = 640,
) {
  const expected =
    kind === "detector"
      ? settings.detector === "yolox"
        ? [[1, [8, 16, 32].reduce((n, s) => n + (size / s) ** 2, 0), 85]]
        : [[1, 84, -1]]
      : settings.format === "simcc"
        ? [
            [1, 133, settings.width * 2],
            [1, 133, settings.height * 2],
          ]
        : [[1, 133, -1, -1]];
  if (
    outputs.length !== expected.length ||
    outputs.some(
      (o, i) =>
        o.dims.length !== expected[i].length ||
        o.dims.some(
          (d, j) =>
            !Number.isInteger(d) ||
            d <= 0 ||
            (expected[i][j] !== -1 && d !== expected[i][j]),
        ) ||
        !(o.data instanceof Float32Array) ||
        o.data.length !== o.dims.reduce((a, b) => a * b, 1) ||
        o.data.some((v) => !Number.isFinite(v)),
    )
  )
    throw Error(
      `${kind}: incompatible or non-finite output. Expected ${JSON.stringify(expected)} (133 whole-body joints; x then y for SimCC).`,
    );
}
