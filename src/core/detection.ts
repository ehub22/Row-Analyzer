import type { Box } from "./types";
export function iou(a: Box, b: Box) {
  const w = Math.max(
      0,
      Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x),
    ),
    h = Math.max(
      0,
      Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y),
    );
  return (w * h) / (a.width * a.height + b.width * b.height - w * h || 1);
}
export function nms(boxes: Box[], threshold = 0.45): Box[] {
  const result: Box[] = [];
  for (const b of [...boxes].sort((a, b) => b.confidence - a.confidence)) {
    if (result.every((a) => iou(a, b) < threshold)) result.push(b);
  }
  return result;
}
// YOLOX raw-grid export, [1,8400,85], raw BGR 0..255 input, top-left letterbox.
export function decodeYolox(
  data: Float32Array,
  dims: readonly number[],
  size: number,
  ratio: number,
): Box[] {
  if (
    dims.length !== 3 ||
    dims[0] !== 1 ||
    dims[2] !== 85 ||
    dims[1] !== [8, 16, 32].reduce((n, s) => n + (size / s) ** 2, 0) ||
    data.length !== dims[1] * 85 ||
    !Number.isFinite(ratio) ||
    ratio <= 0
  )
    throw Error(
      "Expected YOLOX output [1,N,85]. Use yolox_l.onnx with raw grid outputs.",
    );
  const boxes: Box[] = [];
  let row = 0;
  for (const stride of [8, 16, 32])
    for (let y = 0; y < size / stride; y++)
      for (let x = 0; x < size / stride; x++) {
        const k = row++ * 85,
          score = data[k + 4] * data[k + 5];
        if (!Number.isFinite(score) || score < 0.35) continue;
        const cx = ((data[k] + x) * stride) / ratio,
          cy = ((data[k + 1] + y) * stride) / ratio,
          w = (Math.exp(data[k + 2]) * stride) / ratio,
          h = (Math.exp(data[k + 3]) * stride) / ratio;
        if (![cx, cy, w, h].every(Number.isFinite) || w <= 0 || h <= 0)
          continue;
        boxes.push({
          x: cx - w / 2,
          y: cy - h / 2,
          width: w,
          height: h,
          confidence: score,
        });
      }
  return nms(boxes);
}
// Ultralytics YOLOv8/11, RGB 0..1 input, top-left letterbox. No post-NMS exports.
export function decodeYolo(
  data: Float32Array,
  dims: readonly number[],
  ratio: number,
): Box[] {
  if (dims.length !== 3 || dims[1] !== 84)
    throw Error(
      "Expected YOLOv8/11 output [1,84,N]. End-to-end exports are not supported.",
    );
  const n = dims[2],
    boxes: Box[] = [];
  for (let i = 0; i < n; i++) {
    const score = data[4 * n + i];
    if (!Number.isFinite(score) || score < 0.35) continue;
    boxes.push({
      x: (data[i] - data[2 * n + i] / 2) / ratio,
      y: (data[n + i] - data[3 * n + i] / 2) / ratio,
      width: data[2 * n + i] / ratio,
      height: data[3 * n + i] / ratio,
      confidence: score,
    });
  }
  return nms(boxes);
}
export function paddedCrop(
  b: Box,
  inputW: number,
  inputH: number,
  scale = 1.25,
): Box {
  let w = b.width * scale,
    h = b.height * scale;
  if (w / h > inputW / inputH) h = (w * inputH) / inputW;
  else w = (h * inputW) / inputH;
  return {
    x: b.x + b.width / 2 - w / 2,
    y: b.y + b.height / 2 - h / 2,
    width: w,
    height: h,
    confidence: b.confidence,
  };
}
