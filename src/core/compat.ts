// Shape-level compatibility checks run right after an ONNX session is created.
// They fail fast with an actionable message instead of a mid-video tensor error.
// These checks cannot prove accuracy; they only confirm the documented I/O contract.

export type Shape = readonly (number | string | null | undefined)[];

const isNum = (v: unknown): v is number => typeof v === "number";

function fail(model: string, detail: string): never {
  throw new Error(`${model} is incompatible: ${detail}`);
}

/** YOLOX raw-grid [1,N,85] or YOLOv8/11 [1,84,N] with square NCHW input. */
export function checkDetectorContract(
  input: Shape,
  output: Shape,
  mode: "yolox" | "yolo",
): void {
  const model = "Detector";
  if (input.length !== 4 || (isNum(input[1]) && input[1] !== 3))
    fail(model, `expected an NCHW RGB input, got [${input.join(",")}].`);
  if (isNum(input[2]) && isNum(input[3]) && input[2] !== input[3])
    fail(model, `input must be square, got ${input[2]}×${input[3]}.`);
  if (output.length !== 3)
    fail(model, `expected a 3-D output, got [${output.join(",")}].`);
  if (mode === "yolox") {
    if (isNum(output[2]) && output[2] !== 85)
      fail(
        model,
        `YOLOX needs a [1,N,85] raw-grid output, got [${output.join(",")}]. Decoded-coordinate graphs are not supported.`,
      );
  } else if (isNum(output[1]) && output[1] !== 84) {
    fail(
      model,
      `YOLOv8/11 needs a [1,84,N] output, got [${output.join(",")}]. End-to-end [1,N,6] exports are not supported.`,
    );
  }
}

/** Shape-level check for a selected COCO-17 or COCO-WholeBody-133 export. */
export function checkPoseContract(
  input: Shape,
  outputs: readonly Shape[],
  format: "simcc" | "heatmap",
  width: number,
  height: number,
  joints: 17 | 133 = 133,
): void {
  const model = "Pose model";
  if (
    input.length !== 4 ||
    (isNum(input[0]) && input[0] !== 1) ||
    (isNum(input[1]) && input[1] !== 3)
  )
    fail(model, `expected [1,3,H,W] input, got [${input.join(",")}].`);
  if (isNum(input[2]) && input[2] !== height)
    fail(model, `input height is ${input[2]} but settings say ${height}.`);
  if (isNum(input[3]) && input[3] !== width)
    fail(model, `input width is ${input[3]} but settings say ${width}.`);
  const expected = format === "simcc" ? 2 : 1;
  const rank = format === "simcc" ? 3 : 4;
  if (outputs.length !== expected)
    fail(
      model,
      `${format} needs ${expected} output${expected === 1 ? "" : "s"}; found ${outputs.length}.`,
    );
  for (const output of outputs)
    if (
      output.length !== rank ||
      (isNum(output[0]) && output[0] !== 1) ||
      (isNum(output[1]) && output[1] !== joints)
    )
      fail(
        model,
        `selected ${joints}-joint ${format} layout, got [${output.join(",")}].`,
      );
}

/** Metadata may be symbolic; the first real output shapes must still agree. */
export function checkPoseOutputDims(
  outputs: readonly (readonly number[])[],
  format: "simcc" | "heatmap",
  width: number,
  height: number,
  joints: 17 | 133,
): void {
  checkPoseContract(
    [1, 3, height, width],
    outputs,
    format,
    width,
    height,
    joints,
  );
  if (outputs.some((dims) => dims.some((n) => !Number.isInteger(n) || n <= 0)))
    fail("Pose model", "output dimensions must be positive integers.");
  if (format === "simcc") {
    const xScale = outputs[0][2] / width;
    const yScale = outputs[1][2] / height;
    if (Math.abs(xScale - yScale) > 1e-4)
      fail(
        "Pose model",
        "SimCC x/y axes do not use the same coordinate scale or are reversed.",
      );
  }
}
