export const jointNames = [
  "Nose",
  "Left eye",
  "Right eye",
  "Left ear",
  "Right ear",
  "Left shoulder",
  "Right shoulder",
  "Left elbow",
  "Right elbow",
  "Left wrist",
  "Right wrist",
  "Left hip",
  "Right hip",
  "Left knee",
  "Right knee",
  "Left ankle",
  "Right ankle",
  "Left big toe",
  "Left small toe",
  "Left heel",
  "Right big toe",
  "Right small toe",
  "Right heel",
  ...Array.from({ length: 68 }, (_, i) => `Face ${i + 1}`),
  ...Array.from({ length: 21 }, (_, i) => `Left hand ${i}`),
  ...Array.from({ length: 21 }, (_, i) => `Right hand ${i}`),
];
export const bones: [number, number][] = [
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [15, 17],
  [15, 19],
  [16, 20],
  [16, 22],
  [0, 5],
  [0, 6],
  ...([91, 112].flatMap((base) =>
    Array.from({ length: 5 }, (_, f) => [
      [base, base + 1 + f * 4],
      [base + 1 + f * 4, base + 2 + f * 4],
      [base + 2 + f * 4, base + 3 + f * 4],
      [base + 3 + f * 4, base + 4 + f * 4],
    ]).flat(),
  ) as [number, number][]),
];
export const flipIndices = Array.from({ length: 133 }, (_, i) => i);
for (const [a, b] of [
  [1, 2],
  [3, 4],
  [5, 6],
  [7, 8],
  [9, 10],
  [11, 12],
  [13, 14],
  [15, 16],
  [17, 20],
  [18, 21],
  [19, 22],
]) {
  flipIndices[a] = b;
  flipIndices[b] = a;
}
// COCO WholeBody facial mirror mapping (68-point iBUG convention).
const face = [
  16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 26, 25, 24, 23, 22,
  21, 20, 19, 18, 17, 27, 28, 29, 30, 35, 34, 33, 32, 31, 45, 44, 43, 42, 47,
  46, 39, 38, 37, 36, 41, 40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55,
  64, 63, 62, 61, 60, 67, 66, 65,
];
face.forEach((v, i) => (flipIndices[23 + i] = 23 + v));
for (let i = 0; i < 21; i++) {
  flipIndices[91 + i] = 112 + i;
  flipIndices[112 + i] = 91 + i;
}
