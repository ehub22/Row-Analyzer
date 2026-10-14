export type Origin = "observed" | "inferred" | "manual" | "missing";
export interface Point {
  x: number;
  y: number;
  confidence: number;
  origin: Origin;
  flags?: string[];
}
export interface Vec3 {
  x: number;
  y: number;
  z: number;
}
export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
}
export interface PoseFrame {
  time: number;
  seat: number;
  points: Point[];
  box: Box;
  identityConfidence: number;
  lifted?: LiftedPose;
  oars?: OarCandidate[];
}
export interface Boat {
  name: string;
  seats: number;
  discipline: "sweep" | "scull";
  rigging: ("port" | "starboard")[];
  bestSeat: number;
  bowLeft: boolean;
}
export interface Metric {
  value: number | null;
  unit: string;
  confidence: number;
  reason?: string;
}
export interface Stroke {
  seat: number;
  index: number;
  catch: number;
  finish: number;
  nextCatch: number;
  rate: Metric;
  ratio: Metric;
  confidence: number;
  observedFraction: number;
}
export interface CameraEstimate {
  azimuth: number | null;
  confidence: number;
  reason: string;
  source: "unresolved" | "calibrated" | "synthetic" | "estimated";
  axis: { x: number; y: number };
  uncertaintyDegrees?: number;
}
export interface Correction {
  seat: number;
  time: number;
  joint: number;
  before: Point;
  after: Point;
}
export interface Session {
  version: 1;
  name: string;
  demo: boolean;
  width: number;
  height: number;
  duration: number;
  timestamps: number[];
  boat: Boat;
  frames: PoseFrame[];
  strokes: Stroke[];
  camera: CameraEstimate;
  corrections: Correction[];
  warnings: string[];
  model: string;
  provider: string;
}
export interface ModelSettings {
  detector: "yolox" | "yolo";
  format: "simcc" | "heatmap";
  width: number;
  height: number;
  tta: boolean;
  wasm: boolean;
  oars?: boolean;
  region: { left: number; top: number; right: number; bottom: number };
}
export const defaultModels: ModelSettings = {
  detector: "yolox",
  format: "simcc",
  width: 288,
  height: 384,
  tta: true,
  wasm: false,
  region: { left: 0, top: 0, right: 100, bottom: 100 },
};
export const boatClasses: Record<
  string,
  { seats: number; discipline: "sweep" | "scull" }
> = {
  Single: { seats: 1, discipline: "scull" },
  Double: { seats: 2, discipline: "scull" },
  Pair: { seats: 2, discipline: "sweep" },
  "Coxless pair": { seats: 2, discipline: "sweep" },
  Quad: { seats: 4, discipline: "scull" },
  "Coxless four": { seats: 4, discipline: "sweep" },
  "Coxed four": { seats: 4, discipline: "sweep" },
  Eight: { seats: 8, discipline: "sweep" },
  Custom: { seats: 4, discipline: "sweep" },
};
export const defaultBoat: Boat = {
  name: "Eight",
  seats: 8,
  discipline: "sweep",
  rigging: Array.from({ length: 8 }, (_, i) => (i % 2 ? "starboard" : "port")),
  bestSeat: 8,
  bowLeft: true,
};
export const colors = [
  "#d68349",
  "#54969b",
  "#877abb",
  "#bf777e",
  "#709b64",
  "#cead53",
  "#608dc3",
  "#c25e49",
];
export const missing = (): Point => ({
  x: 0,
  y: 0,
  confidence: 0,
  origin: "missing",
});

export interface LiftedPose {
  method: "kinematic-weak-perspective-v1";
  points: ((Vec3 & { confidence: number }) | null)[];
  units: "torso lengths";
  ambiguous: true;
}
export interface OarCandidate {
  side: "left" | "right";
  start: { x: number; y: number };
  end: { x: number; y: number };
  confidence: number;
  support: number;
  age: number;
  source: "image-line-candidate";
}
export interface BiomechanicsRow {
  seat: number;
  stroke: number;
  metrics: Record<string, Metric>;
}
