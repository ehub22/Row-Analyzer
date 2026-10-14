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

/** DWPose / RTMW SimCC or whole-body heatmap export with 133 joints. */
export function checkPoseContract(
  input: Shape,
  outputs: Shape[],
  format: "simcc" | "heatmap",
  width: number,
  height: number,
): void {
  const model = "Pose model";
  if (input.length !== 4)
    fail(model, `expected [1,3,H,W] input, got [${input.join(",")}].`);
  if (isNum(input[2]) && input[2] !== height)
    fail(model, `input height is ${input[2]} but settings say ${height}.`);
  if (isNum(input[3]) && input[3] !== width)
    fail(model, `input width is ${input[3]} but settings say ${width}.`);
  if (format === "simcc") {
    if (outputs.length !== 2)
      fail(
        model,
        `SimCC needs two outputs (x, y); found ${outputs.length}. A 17-joint body model is not whole-body.`,
      );
    for (const o of outputs)
      if (o.length !== 3 || (isNum(o[1]) && o[1] !== 133))
        fail(
          model,
          `expected 133 joints per SimCC axis, got [${o.join(",")}]. A COCO-17 body checkpoint is not supported.`,
        );
  } else {
    const o = outputs[0];
    if (
      outputs.length !== 1 ||
      !o ||
      o.length !== 4 ||
      (isNum(o[1]) && o[1] !== 133)
    )
      fail(model, `expected one [1,133,H,W] heatmap output.`);
  }
}
