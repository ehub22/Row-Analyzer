import type { TtaMode } from "./tta";
import type { PerformanceReport } from "./perf";
import type { AngleFrame } from "./angles";

/** Latest processed frame only; bitmap ownership stays with the UI coordinator. */
export interface LivePreview {
  bitmap: ImageBitmap;
  width: number;
  height: number;
  time: number;
  processed: number;
  frames: PoseFrame[];
}

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
  /** Pose is measured on this frame; the crop itself may come from a prior frame. */
  boxSource?: "detected" | "predicted";
  /** Number of decoded frames since the last detector inference for this crop. */
  detectorAge?: number;
}
export interface Boat {
  name: string;
  seats: number;
  discipline: "sweep" | "scull";
  rigging: ("port" | "starboard")[];
  bestSeat: number;
  bowLeft: boolean;
  /**
   * Feet are strapped to the stretcher below the gunwale, so on-water footage
   * almost never shows them. When true (the default), ankle and foot outputs
   * are discarded as model guesses instead of being tracked (see
   * core/occlusion.ts). Set false for ergometer or overhead footage where the
   * feet really are in view. Boats saved before this field existed have it
   * undefined: new analyses treat that as true, while an imported session
   * keeps the frames it was analysed with.
   */
  feetHidden?: boolean;
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
  source: "unresolved" | "calibrated" | "synthetic";
  axis: { x: number; y: number };
}
/**
 * What the user measured or entered about the boat and the camera.
 *
 * The camera solve is not automatic, and ROADMAP.md says why: a shell axis in
 * the image alone does not determine an azimuth. These are the lengths and the
 * lens that turn the crew's own seat line into metric geometry — a rigging tape
 * measure and a lens specification, not a calibration board. Absent or null
 * fields are abstentions waiting to happen, and the solve reports which one.
 */
export interface BoatCalibration {
  /** Seat spacing along the shell, metres, centre to centre. */
  seatSpacing?: number | null;
  /** Shell length bow to stern, metres. Reported, not used by the solve. */
  shellLength?: number | null;
  /** Focal length in pixels, when it is known directly. */
  focalPx?: number | null;
  /** Focal length as a 35 mm equivalent, converted with the image width. */
  focal35mm?: number | null;
  /** Principal point in pixels; the frame centre when unset. */
  cx?: number | null;
  cy?: number | null;
  /** Camera height above the waterline, metres, for handle height. */
  cameraHeight?: number | null;
  /** The user checked the shell axis and which end of the image the bow is at. */
  axisVerified?: boolean;
  /**
   * The user checked that the clip is level, so the camera's vertical is the
   * boat's. Nothing a crew does on camera can say whether it is, and a rolled
   * clip tilts every lifted angle by however far the horizon is off.
   */
  level?: boolean;
}
/** One measurement behind a camera solve: what it was, and how far it is
 * trusted. Cues that disagreed are kept, so a withheld answer can say why. */
export interface CameraCue {
  name: string;
  value: number;
  confidence: number;
  detail: string;
}
/**
 * A camera estimate plus the metric geometry that came with it. Optional
 * fields so an estimate that abstained — and a session saved before the solve
 * existed — are both valid values of this type.
 */
export interface CameraSolve extends CameraEstimate {
  /** Distance from the camera to the shell centreline, metres. */
  distance?: number | null;
  /** Pixels per metre at the shell centreline. */
  pxPerMetre?: number | null;
  /** Principal point the solve used, pixels. */
  principal?: { x: number; y: number } | null;
  /** Focal length the solve used, pixels: what a lift projects through. */
  focalPx?: number | null;
  /** Residual of the seat-line fit, pixels. */
  residual?: number | null;
  /** The cues behind the answer, including the ones that argued with it. */
  cues?: CameraCue[];
}
/**
 * What produced a session's lifted angles, or why there are none.
 *
 * A lift is an inference about depth that no image can confirm on its own, so
 * the session records which model ran, how many frames it was asked about, how
 * many it placed, how many it refused and why. "No angles" is an answer with a
 * reason, not a blank.
 */
export interface LiftProvenance {
  ran: boolean;
  /** Model description, when one ran. */
  model?: string;
  /** Why it did not run, or why the frames it refused were refused. */
  reason?: string;
  attempted?: number;
  placed?: number;
  refused?: number;
  /** Median reprojection error over the placed frames, pixels. */
  reprojection?: number | null;
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
  camera: CameraSolve;
  /** Boat and camera measurements the solve was given; absent on sessions
   * saved before it existed, which is why the solve abstains on them. */
  calibration?: BoatCalibration;
  /**
   * Normalized joint angles from a calibrated 3D lift, one row per entry of
   * `frames`, in the same order. Absent when the lift did not run or the camera
   * was not solved, and on sessions saved before it existed. Every value in a
   * row is either an angle in the boat's frame or a withheld metric with its
   * reason; nothing here is an observation of depth.
   */
  angles?: AngleFrame[];
  /** What produced `angles`, or why there are none. */
  lift?: LiftProvenance;
  corrections: Correction[];
  warnings: string[];
  model: string;
  provider: string;
  /** Actual pose-model output layout; older sessions used 133. */
  poseJoints?: 17 | 133;
  /** Where the run's time went (see core/perf.ts); absent on imports from
   * builds that predate it. */
  performance?: PerformanceReport;
}
export interface ModelSettings {
  detector: "yolox" | "yolo";
  format: "simcc" | "heatmap";
  /** COCO body-only or COCO WholeBody output layout. */
  poseJoints: 17 | 133;
  width: number;
  height: number;
  /** Test-time augmentation policy: one pass, a cascaded adaptive plan, or
   * the full four-pass fusion (see core/tta.ts). */
  tta: TtaMode;
  /** Auto may reuse guarded crops between detector anchors; every-frame is the baseline. */
  detectorCadence: "auto" | "every-frame";
  wasm: boolean;
  /** Parallel pose workers. 0 = automatic (see core/parallel.ts); 1–8 = fixed.
   * Each worker owns its own pose-model session, so more workers use more RAM. */
  workers: number;
  region: { left: number; top: number; right: number; bottom: number };
}
export const defaultModels: ModelSettings = {
  detector: "yolox",
  format: "simcc",
  poseJoints: 133,
  width: 288,
  height: 384,
  tta: "adaptive",
  detectorCadence: "every-frame",
  wasm: false,
  workers: 0,
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
  feetHidden: true,
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
