/**
 * Camera normalization from a horizon and the shell axis, then body angles
 * measured in the rower's sagittal plane.
 *
 * Model: pinhole camera, square pixels, principal point at the image centre
 * unless supplied, focal length `f` in pixels. Camera coordinates: x right,
 * y down, z forward.
 *
 * Geometry used (all exact for ideal inputs, no depth needed):
 * - A world-horizontal line projects to an image line l. Its back-projection
 *   plane has normal K^T l. Hence the world up direction (the horizon's
 *   normal) is up ∝ K^T l_horizon. Roll and pitch follow from `up`; roll needs
 *   no focal length at all.
 * - The shell axis is horizontal. Given its image line, its 3D direction is
 *   d ∝ (K^T l_axis) × up. The bow/stern image order fixes the sign of d.
 * - The sagittal plane contains d and up. Its normal is n = d × up. Azimuth
 *   0° = shell across the view (side-on), 90° = bow/stern-on.
 * - Image points map to that plane by X = r / (n·r), where r = K⁻¹[u,1]. This
 *   is an exact reconstruction for points lying on the plane, and a pure
 *   similarity (no perspective error) for side-on views. Body points off the
 *   plane (arms out to the side, lateral hip sway) and oblique views add
 *   projective error that no 2D cue removes. The solver therefore gates on
 *   azimuth and reports its limits.
 *
 * Nothing here is calibrated biomechanics: the output is a heuristic
 * plane-normalized estimate. It is withheld whenever a gate fails.
 */
import type { Point } from "./types";

export type Vec = [number, number, number];
/** Homogeneous image line a·x + b·y + c = 0 (pixels), with (a,b) unit length. */
export type Line = Vec;
export interface Pt2 {
  x: number;
  y: number;
}

const EPS = 1e-12;
const dot = (a: Vec, b: Vec) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec, b: Vec): Vec => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const norm = (a: Vec) => Math.hypot(a[0], a[1], a[2]);
const scale = (a: Vec, k: number): Vec => [a[0] * k, a[1] * k, a[2] * k];
const add = (a: Vec, b: Vec): Vec => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: Vec, b: Vec): Vec => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const deg = (rad: number) => (rad * 180) / Math.PI;
const finite = (...v: number[]) => v.every((x) => Number.isFinite(x));

function unit(a: Vec): Vec | null {
  const n = norm(a);
  return n > EPS ? scale(a, 1 / n) : null;
}

/** Normalize a homogeneous line so (a,b) has unit length; null if degenerate. */
export function normalizeLine(l: Line): Line | null {
  const n = Math.hypot(l[0], l[1]);
  return n > EPS && finite(...l) ? scale(l, 1 / n) : null;
}

/** Line through two image points (unit-normalized). */
export function lineThrough(p: Pt2, q: Pt2): Line | null {
  const l = cross([p.x, p.y, 1], [q.x, q.y, 1]);
  return normalizeLine(l);
}

/** Finite intersection point of two lines, or null if parallel / at infinity. */
export function intersect(l1: Line, l2: Line): Pt2 | null {
  const h = cross(l1, l2);
  if (Math.abs(h[2]) < 1e-9 * norm(h)) return null;
  return { x: h[0] / h[2], y: h[1] / h[2] };
}

/** Signed perpendicular distance from pixel p to a unit-normalized line. */
export function pointLineDistance(p: Pt2, l: Line): number {
  return l[0] * p.x + l[1] * p.y + l[2];
}

/** Smallest eigenpair of a symmetric 3×3 matrix (Jacobi rotations). */
function smallestEigenvector(m: number[][]): Vec {
  const a = m.map((r) => r.slice());
  const v = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let sweep = 0; sweep < 50; sweep++) {
    const off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (off < 1e-14) break;
    for (const [p, q] of [
      [0, 1],
      [0, 2],
      [1, 2],
    ]) {
      if (Math.abs(a[p][q]) < 1e-18) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t =
        Math.sign(theta || 1) /
        (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1),
        s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[k][p],
          akq = a[k][q];
        a[k][p] = c * akp - s * akq;
        a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p][k],
          aqk = a[q][k];
        a[p][k] = c * apk - s * aqk;
        a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k][p],
          vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq;
        v[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const idx = [0, 1, 2].sort((i, j) => a[i][i] - a[j][j])[0];
  return [v[0][idx], v[1][idx], v[2][idx]];
}

export interface VanishingPoint {
  /** Finite image point, or null when the family is parallel to the image plane. */
  point: Pt2 | null;
  /** Lines that agreed with the fit after one outlier rejection pass. */
  inliers: number;
  /** RMS point-to-line residual of the inliers, pixels. */
  rmsPx: number;
}

/**
 * Least-squares vanishing point of a family of image lines, with one
 * median-based outlier rejection pass. Lines must be unit-normalized.
 */
/** Pixel distance from the (possibly homogeneous) point v to unit line l. */
function residualPx(l: Line, v: Vec): number {
  const w = Math.abs(v[2]) > 1e-9 * norm(v) ? Math.abs(v[2]) : norm(v);
  return Math.abs(dot(l, v)) / w;
}

export function vanishingPoint(lines: Line[]): VanishingPoint | null {
  if (lines.length < 2) return null;
  const TOL_PX = 2;
  const fit = (set: Line[]) => {
    const m = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    for (const l of set)
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++) m[i][j] += l[i] * l[j];
    return smallestEigenvector(m);
  };
  const inliersOf = (v: Vec) => lines.filter((l) => residualPx(l, v) <= TOL_PX);

  // RANSAC over pairwise intersections: every pair proposes a vanishing point;
  // keep the one most lines agree with. Deterministic, bounded sampling.
  const n = lines.length;
  const pairs: [number, number][] = [];
  if (n <= 30) {
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) pairs.push([i, j]);
  } else {
    let seed = 12345;
    const rnd = () => (seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32;
    for (let k = 0; k < 400; k++) {
      const i = Math.floor(rnd() * n);
      let j = Math.floor(rnd() * n);
      if (j === i) j = (j + 1) % n;
      pairs.push([i, j]);
    }
  }
  let bestSet: Line[] = [];
  for (const [i, j] of pairs) {
    const v = cross(lines[i], lines[j]);
    if (norm(v) < EPS) continue;
    const set = inliersOf(v);
    if (set.length > bestSet.length) bestSet = set;
  }
  if (bestSet.length < 2) return null;
  // Refit on the consensus set, then re-collect inliers once.
  let v = fit(bestSet);
  bestSet = inliersOf(v);
  if (bestSet.length >= 2) v = fit(bestSet);
  const use = bestSet.length >= 2 ? bestSet : lines;
  const rmsPx = Math.sqrt(
    use.reduce((acc, l) => acc + residualPx(l, v) ** 2, 0) / use.length,
  );
  const point =
    Math.abs(v[2]) > 1e-9 * norm(v) ? { x: v[0] / v[2], y: v[1] / v[2] } : null;
  return { point, inliers: use.length, rmsPx };
}

/**
 * Focal length (px) from two image vanishing points of orthogonal world
 * directions with known principal point (Caprile & Torre): f² = -(v1-p)·(v2-p).
 */
export function focalFromOrthogonalVanishingPoints(
  v1: Pt2,
  v2: Pt2,
  principal: Pt2,
): number | null {
  const f2 = -(
    (v1.x - principal.x) * (v2.x - principal.x) +
    (v1.y - principal.y) * (v2.y - principal.y)
  );
  return f2 > 0 ? Math.sqrt(f2) : null;
}

export interface CameraInput {
  width: number;
  height: number;
  /** Horizon in source-image pixels (detected or user-placed). */
  horizon: Line;
  /** Confidence in the horizon, 0–1. Manual placement may pass 1. */
  horizonConfidence?: number;
  /** Bow and stern points on the gunwale line, source-image pixels. */
  bow: Pt2;
  stern: Pt2;
  /** Known focal length in pixels, if the lens is calibrated. */
  focalPx?: number;
  /**
   * Optional family of image lines that are vertical in the world (stanchions,
   * door frames, pillars). With the shell axis they give the focal length by
   * self-calibration when `focalPx` is absent.
   */
  verticalLines?: Line[];
  principal?: Pt2;
}

export type FocalSource = "supplied" | "vanishing-points" | "none";

export interface CameraSolution {
  /** True only when every gate passed; body angles require this. */
  usable: boolean;
  reasons: string[];
  focalPx: number | null;
  focalSource: FocalSource;
  principal: Pt2;
  /** Image roll: 0 = horizon level. Needs no focal length. */
  rollDeg: number;
  /** Camera elevation: positive = looking up. Null without focal length. */
  pitchDeg: number | null;
  /** 0 = shell across the view (side-on), 90 = bow/stern-on. Null without focal. */
  azimuthDeg: number | null;
  /** Heuristic 0–1 quality score. Not a calibrated probability. */
  confidence: number;
  /** Camera-frame unit vectors (internal use for mapping points). */
  up: Vec | null;
  fore: Vec | null;
  lateral: Vec | null;
}

export const MAX_AZIMUTH_DEG = 60;
export const MAX_ROLL_DEG = 30;
export const MAX_PITCH_DEG = 45;

/**
 * Solve camera orientation relative to the horizon and the shell. Never
 * throws for bad input: it returns `usable: false` with reasons instead.
 */
export function solveCameraNormalization(input: CameraInput): CameraSolution {
  const reasons: string[] = [];
  const principal = input.principal ?? {
    x: input.width / 2,
    y: input.height / 2,
  };
  const base = {
    principal,
    focalPx: null as number | null,
    focalSource: "none" as FocalSource,
    rollDeg: NaN,
    pitchDeg: null as number | null,
    azimuthDeg: null as number | null,
    confidence: 0,
    up: null as Vec | null,
    fore: null as Vec | null,
    lateral: null as Vec | null,
  };
  const fail = (reason: string) => ({
    ...base,
    usable: false,
    reasons: [...reasons, reason],
  });

  if (
    !finite(input.width, input.height, principal.x, principal.y) ||
    input.width <= 0 ||
    input.height <= 0
  )
    return fail("Image size is invalid.");
  const hz = normalizeLine(input.horizon);
  if (!hz) return fail("Horizon line is degenerate.");
  const bowLine = lineThrough(input.bow, input.stern);
  if (!bowLine) return fail("Bow and stern points coincide.");
  const hConf = Math.max(0, Math.min(1, input.horizonConfidence ?? 1));
  if (hConf < 0.3)
    reasons.push(
      `Horizon confidence ${Math.round(hConf * 100)}% is too low to lock the camera roll.`,
    );

  // Roll from the horizon's direction alone (no focal length). A line has no
  // orientation, so report it in (-90°, 90°]: the image-up sign is irrelevant.
  let rollDeg = deg(Math.atan2(hz[0], -hz[1]));
  if (rollDeg > 90) rollDeg -= 180;
  else if (rollDeg <= -90) rollDeg += 180;
  base.rollDeg = rollDeg;
  if (Math.abs(rollDeg) > MAX_ROLL_DEG)
    reasons.push(
      `Camera roll ${rollDeg.toFixed(1)}° exceeds the ±${MAX_ROLL_DEG}° solver range.`,
    );

  // Focal length: supplied, or self-calibrated from shell axis and verticals.
  let f: number | null = null;
  let source: FocalSource = "none";
  if (input.focalPx !== undefined) {
    const fp = input.focalPx;
    if (finite(fp) && fp >= 0.25 * input.width && fp <= 5 * input.width) {
      f = fp;
      source = "supplied";
    } else {
      reasons.push("Supplied focal length is outside 0.25–5× the image width.");
    }
  } else if (input.verticalLines && input.verticalLines.length >= 2) {
    const vAxisPt = intersect(bowLine, hz);
    const vVert = vanishingPoint(
      input.verticalLines
        .map(normalizeLine)
        .filter((l): l is Line => l !== null),
    );
    if (!vAxisPt)
      reasons.push(
        "Shell axis is parallel to the horizon in the image; focal self-calibration failed.",
      );
    else if (!vVert?.point)
      reasons.push(
        "Vertical lines do not converge; focal self-calibration failed.",
      );
    else {
      const est = focalFromOrthogonalVanishingPoints(
        vAxisPt,
        vVert.point,
        principal,
      );
      if (est !== null && est >= 0.25 * input.width && est <= 5 * input.width) {
        f = est;
        source = "vanishing-points";
      } else {
        reasons.push(
          "Shell-axis and vertical vanishing points are not consistent with a physical focal length.",
        );
      }
    }
  } else {
    reasons.push(
      "Focal length unknown: pitch, azimuth and body angles are withheld. Supply a calibrated focal length or two vertical lines.",
    );
  }
  base.focalPx = f;
  base.focalSource = source;
  if (f === null) return { ...base, usable: false, reasons, confidence: 0 };

  // Camera-frame world up from the horizon: up ∝ K^T l_h, sign so image-up is up.
  // K^T l = (f·a, f·b, cx·a + cy·b + c) for K = [[f,0,cx],[0,f,cy],[0,0,1]].
  const kt = (l: Line): Vec => [
    f * l[0],
    f * l[1],
    principal.x * l[0] + principal.y * l[1] + l[2],
  ];
  let up = unit(kt(hz));
  if (!up)
    return {
      ...base,
      usable: false,
      reasons: [...reasons, "Horizon is degenerate."],
    };
  if (up[1] > 0) up = scale(up, -1);
  const pitchDeg = deg(Math.asin(Math.max(-1, Math.min(1, up[2]))));
  if (Math.abs(pitchDeg) > MAX_PITCH_DEG)
    reasons.push(
      `Camera pitch ${pitchDeg.toFixed(1)}° exceeds the ±${MAX_PITCH_DEG}° solver range.`,
    );

  // Shell axis direction: horizontal, lies in the back-projection plane of its image line.
  const nAxis = kt(bowLine);
  let fore = unit(cross(nAxis, up));
  if (!fore)
    return {
      ...base,
      usable: false,
      reasons: [...reasons, "Shell axis is degenerate."],
    };
  // Sign: the 3D direction's image-plane component must point from stern to bow.
  const e = Math.hypot(fore[0], fore[1]);
  const towardBow =
    fore[0] * (input.bow.x - input.stern.x) +
    fore[1] * (input.bow.y - input.stern.y);
  if (e < 0.2) {
    reasons.push(
      "Shell points almost at the camera (bow/stern-on): bow direction is ambiguous.",
    );
  } else if (towardBow < 0) fore = scale(fore, -1);
  // Sagittal normal faces the lens (positive z), so the plane lies in front of
  // the camera. Flipping n alone would mirror the in-plane y coordinate.
  let lateral = unit(cross(fore, up));
  if (!lateral)
    return {
      ...base,
      usable: false,
      reasons: [...reasons, "Sagittal plane is degenerate."],
    };
  if (lateral[2] < 0) lateral = scale(lateral, -1);

  // Azimuth: angle between the shell and the camera's horizontal forward direction.
  const zAxis: Vec = [0, 0, 1];
  const fh = unit(sub(zAxis, scale(up, dot(zAxis, up))));
  const dh = unit(sub(fore, scale(up, dot(fore, up))));
  let azimuthDeg: number | null = null;
  if (fh && dh && Math.abs(pitchDeg) <= MAX_PITCH_DEG) {
    const cosC = Math.min(1, Math.abs(dot(fh, dh)));
    azimuthDeg = 90 - deg(Math.acos(cosC));
  }
  base.pitchDeg = pitchDeg;
  base.azimuthDeg = azimuthDeg;
  base.up = up;
  base.fore = fore;
  base.lateral = lateral;

  const azimuthOk = azimuthDeg !== null && azimuthDeg < MAX_AZIMUTH_DEG;
  if (azimuthDeg !== null && !azimuthOk)
    reasons.push(
      `Azimuth ${azimuthDeg.toFixed(0)}° is too oblique: the sagittal plane projects almost edge-on, so body angles are withheld.`,
    );
  const azScore =
    azimuthDeg === null
      ? 0
      : Math.max(0, Math.cos((azimuthDeg * Math.PI) / 180));
  const fScore = source === "supplied" ? 1 : 0.85;
  const confidence = hConf * fScore * azScore;
  const usable = reasons.length === 0 && azimuthOk && hConf >= 0.3;
  if (usable)
    reasons.push(
      "Plane-normalized estimate from a horizon and shell axis. Depth error and non-planar body motion are not corrected; not validated.",
    );
  return { ...base, usable, reasons, confidence, pitchDeg };
}

/**
 * Map an image pixel to the rower's sagittal plane (x toward bow, y up).
 * Units are arbitrary: the plane sits at unit distance, so lengths scale with
 * the unknown depth, but all angles are exact. Returns null for rays parallel
 * to the plane.
 */
export function toSagittalPlane(solution: CameraSolution, p: Pt2): Pt2 | null {
  if (
    !solution.up ||
    !solution.fore ||
    !solution.lateral ||
    solution.focalPx === null
  )
    return null;
  const f = solution.focalPx;
  const r: Vec = [
    (p.x - solution.principal.x) / f,
    (p.y - solution.principal.y) / f,
    1,
  ];
  const s = dot(solution.lateral, r);
  if (Math.abs(s) < 1e-6) return null;
  const X = scale(r, 1 / s);
  return { x: dot(solution.fore, X), y: dot(solution.up, X) };
}

export interface BodyAngleSet {
  /** Signed; positive = trunk leaning toward the bow. */
  trunkFromVerticalDeg: number | null;
  /** Hip→knee direction, degrees above the bow-ward horizontal. */
  thighFromHorizontalDeg: number | null;
  /** Knee→ankle direction, degrees above the bow-ward horizontal. */
  shinFromHorizontalDeg: number | null;
  /** Interior angle at the hip between trunk and thigh. */
  hipDeg: number | null;
  /** Interior knee angle (hip–knee–ankle). */
  kneeDeg: number | null;
  /** Interior elbow angle (shoulder–elbow–wrist). */
  elbowDeg: number | null;
}

export interface BodyAngles {
  status: "measured" | "withheld";
  /** Side whose joints were used, or null when withheld. */
  side: "left" | "right" | null;
  angles: BodyAngleSet;
  /** Segments that were too foreshortened for their angles to be trusted. */
  foreshortened: string[];
  reasons: string[];
}

export type SegmentName = "trunk" | "thigh" | "shin" | "upperArm" | "forearm";
/** In-plane reference lengths (plane units), from `estimateReferenceLengths`. */
export type ReferenceLengths = Partial<Record<SegmentName, number>>;

/** A segment shorter than this share of its reference is out of the plane. */
export const FORESHORTEN_RATIO = 0.92;

/** COCO-17 / COCO-WholeBody body indices (identical for the first 17 slots). */
const JOINTS = {
  left: { shoulder: 5, elbow: 7, wrist: 9, hip: 11, knee: 13, ankle: 15 },
  right: { shoulder: 6, elbow: 8, wrist: 10, hip: 12, knee: 14, ankle: 16 },
} as const;

const usableJoint = (p: Point | undefined, minConfidence: number) =>
  !!p &&
  (p.origin === "observed" || p.origin === "manual") &&
  p.confidence >= minConfidence &&
  finite(p.x, p.y);

const interiorAngle = (a: Pt2, b: Pt2, c: Pt2): number | null => {
  const u = { x: a.x - b.x, y: a.y - b.y },
    v = { x: c.x - b.x, y: c.y - b.y };
  const d = Math.hypot(u.x, u.y) * Math.hypot(v.x, v.y);
  if (d < 1e-9) return null;
  return deg(Math.acos(Math.max(-1, Math.min(1, (u.x * v.x + u.y * v.y) / d))));
};

const directionDeg = (from: Pt2, to: Pt2): number | null => {
  const dx = to.x - from.x,
    dy = to.y - from.y;
  return Math.hypot(dx, dy) < 1e-9 ? null : deg(Math.atan2(dy, dx));
};

interface PlaneJoints {
  shoulder: Pt2 | null;
  elbow: Pt2 | null;
  wrist: Pt2 | null;
  hip: Pt2 | null;
  knee: Pt2 | null;
  ankle: Pt2 | null;
}

/** Joints of one side mapped to the sagittal plane; unusable joints are null. */
function planeJoints(
  points: Point[],
  side: "left" | "right",
  solution: CameraSolution,
  minConfidence: number,
): PlaneJoints {
  const j = JOINTS[side];
  const at = (i: number) =>
    usableJoint(points[i], minConfidence)
      ? toSagittalPlane(solution, points[i])
      : null;
  return {
    shoulder: at(j.shoulder),
    elbow: at(j.elbow),
    wrist: at(j.wrist),
    hip: at(j.hip),
    knee: at(j.knee),
    ankle: at(j.ankle),
  };
}

const segLen = (a: Pt2 | null, b: Pt2 | null): number | null =>
  a && b ? Math.hypot(a.x - b.x, a.y - b.y) : null;

/** In-plane lengths of the five body segments for one side of one pose. */
function segmentLengths(j: PlaneJoints): Partial<Record<SegmentName, number>> {
  const out: Partial<Record<SegmentName, number>> = {};
  const set = (k: SegmentName, v: number | null) => {
    if (v !== null && v > 0) out[k] = v;
  };
  set("trunk", segLen(j.shoulder, j.hip));
  set("thigh", segLen(j.hip, j.knee));
  set("shin", segLen(j.knee, j.ankle));
  set("upperArm", segLen(j.shoulder, j.elbow));
  set("forearm", segLen(j.elbow, j.wrist));
  return out;
}

/**
 * Reference in-plane length for each segment across a clip: the 95th
 * percentile of observed in-plane lengths (both sides pooled). A limb that is
 * held in the plane shows its full length at some point in a stroke, so the
 * upper tail estimates it. Lengths are in plane units, so the reference is
 * only meaningful for poses taken with the same camera solution.
 */
export function estimateReferenceLengths(
  poses: Point[][],
  solution: CameraSolution,
  minConfidence = 0.3,
): ReferenceLengths {
  const samples: Record<SegmentName, number[]> = {
    trunk: [],
    thigh: [],
    shin: [],
    upperArm: [],
    forearm: [],
  };
  if (!solution.usable) return {};
  for (const pts of poses)
    for (const side of ["left", "right"] as const) {
      const lens = segmentLengths(
        planeJoints(pts, side, solution, minConfidence),
      );
      for (const k of Object.keys(lens) as SegmentName[])
        samples[k].push(lens[k]!);
    }
  const out: ReferenceLengths = {};
  for (const k of Object.keys(samples) as SegmentName[]) {
    const v = samples[k].sort((a, b) => a - b);
    if (v.length >= 5)
      out[k] = v[Math.min(v.length - 1, Math.floor(0.95 * v.length))];
  }
  return out;
}

/**
 * Body angles in the sagittal plane for one pose. Uses observed or manual
 * joints only; inferred and missing joints are never used. Each side is
 * evaluated separately and the side with more valid angles is reported.
 * With `reference` lengths (from `estimateReferenceLengths`), any angle that
 * depends on a foreshortened segment is withheld instead of reported.
 */
export function bodyAnglesFromPose(
  points: Point[],
  solution: CameraSolution,
  options: { minConfidence?: number; reference?: ReferenceLengths } = {},
): BodyAngles {
  const minConfidence = options.minConfidence ?? 0.3;
  const reference = options.reference ?? {};
  const empty: BodyAngleSet = {
    trunkFromVerticalDeg: null,
    thighFromHorizontalDeg: null,
    shinFromHorizontalDeg: null,
    hipDeg: null,
    kneeDeg: null,
    elbowDeg: null,
  };
  if (!solution.usable)
    return {
      status: "withheld",
      side: null,
      angles: empty,
      foreshortened: [],
      reasons: solution.reasons.length
        ? solution.reasons
        : ["Camera solution is not usable."],
    };

  const evalSide = (side: "left" | "right") => {
    const j = planeJoints(points, side, solution, minConfidence);
    const lens = segmentLengths(j);
    // A segment counts as out of the plane when it is clearly shorter than its
    // reference; the angles that use it are then withheld.
    const foreshort = new Set<SegmentName>();
    if (Object.keys(reference).length)
      for (const k of Object.keys(lens) as SegmentName[]) {
        const ref = reference[k];
        if (ref && lens[k]! < FORESHORTEN_RATIO * ref) foreshort.add(k);
      }
    const usable = (...segs: SegmentName[]) =>
      segs.every((s) => !foreshort.has(s));
    const out: BodyAngleSet = { ...empty };
    if (j.shoulder && j.hip && usable("trunk"))
      out.trunkFromVerticalDeg = deg(
        Math.atan2(j.shoulder.x - j.hip.x, j.shoulder.y - j.hip.y),
      );
    if (j.hip && j.knee && usable("thigh"))
      out.thighFromHorizontalDeg = directionDeg(j.hip, j.knee);
    if (j.knee && j.ankle && usable("shin"))
      out.shinFromHorizontalDeg = directionDeg(j.knee, j.ankle);
    if (j.shoulder && j.hip && j.knee && usable("trunk", "thigh"))
      out.hipDeg = interiorAngle(j.shoulder, j.hip, j.knee);
    if (j.hip && j.knee && j.ankle && usable("thigh", "shin"))
      out.kneeDeg = interiorAngle(j.hip, j.knee, j.ankle);
    if (j.shoulder && j.elbow && j.wrist && usable("upperArm", "forearm"))
      out.elbowDeg = interiorAngle(j.shoulder, j.elbow, j.wrist);
    const count = Object.values(out).filter((v) => v !== null).length;
    return { side, out, count, foreshort: [...foreshort] as string[] };
  };

  const l = evalSide("left"),
    r = evalSide("right");
  const best = r.count > l.count ? r : l;
  if (best.count === 0)
    return {
      status: "withheld",
      side: null,
      angles: empty,
      foreshortened: best.foreshort,
      reasons: [
        best.foreshort.length
          ? `Segments ${best.foreshort.join(", ")} are foreshortened (out of the sagittal plane), so the angles they define are withheld.`
          : "No observed or manual shoulder/hip/knee/elbow joints above the confidence threshold on either side.",
      ],
    };
  return {
    status: "measured",
    side: best.side,
    angles: best.out,
    foreshortened: best.foreshort,
    reasons: [
      ...solution.reasons,
      ...(best.foreshort.length
        ? [
            `Segments ${best.foreshort.join(", ")} are foreshortened (out of the sagittal plane); the angles they define are withheld.`,
          ]
        : []),
    ],
  };
}
