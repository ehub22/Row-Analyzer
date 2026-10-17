// Camera solve from the crew itself.
//
// Normalized angles need the boat's frame, and the boat's frame needs to know
// how the camera was turned relative to the shell. ROADMAP.md is blunt about why
// that is not free: "A shell axis in the image alone does not determine
// azimuth", and an unknown focal length leaves azimuth and distance locked
// together. So this module does the one thing that *is* determined, does it from
// cues the app can actually observe, and abstains when they do not agree.
//
// What is determined. The seats lie on a straight line in the boat, equally
// spaced, at a spacing the user can supply (a rigging tape measure, or a
// manufacturer's spec). A pinhole camera maps a straight line in space to a
// projective map of its parameter:
//
//     u(x) = (a·x + b) / (c·x + 1)
//
// where x is a seat's position along the shell in metres and u its image column.
// Three seats fix (a, b, c) by linear least squares; more seats, and a stroke's
// worth of frames averaged per seat, make it robust. Those three numbers carry
// everything the geometry offers:
//
//   * u∞ = a/c is the vanishing point of the shell axis — where the line the
//     seats sit on meets the image plane at infinity;
//   * cx − u∞ = f / tan(azimuth), so a focal length and a principal point turn
//     the vanishing point into an azimuth, in the sign convention
//     core/geometry.ts's `rotateY` already uses: positive azimuth means the
//     stern end, the end towards the rowers' feet, is the nearer one, and the
//     seat spacing widens towards it;
//   * c = −sin(azimuth) / depth gives the depth of the seat line's centre, and
//     a − b·c the pixels per metre there.
//
// What is not determined, which is why calibration is a conversation rather than
// a button. b is simply the image column of the seat line's own origin, so the
// principal point is not recoverable from the seat line: it comes from the
// calibration or is assumed at the frame centre, and that assumption is priced
// in degrees in the cue list rather than hidden. The fit is also invariant to
// the spacing's *scale* — halve the spacing and the vanishing point does not
// move — so a mis-measured tape leaves the azimuth right and the distance and
// pixels-per-metre wrong by the same factor. Normalized angles depend only on
// the rotation, which is why they can be reported from this solve while metric
// lengths stay withheld. And the camera's roll is not observable from a crew at
// all, so a level clip is something the user confirms, not something inferred.
//
// At azimuth 0 — the square side view most launch footage aims for — the spacing
// is uniform in the image, c is zero, the vanishing point is at infinity and no
// amount of noise-free data separates azimuth from distance. That is not a
// failure of this code; it is the ambiguity, and the honest output is
// `source: "unresolved"` with the reason.
//
// Validated against tests/fixtures/rowingSim3d.ts, which projects a known crew
// through a known pinhole camera. That validates the geometry, not monocular
// depth recovery on real footage.

import type {
  Boat,
  BoatCalibration,
  CameraCue,
  CameraSolve,
  PoseFrame,
} from "./types";
import { seen } from "./occlusion";
import { HIP_JOINTS } from "./strokes";
import { median } from "./geometry";

/** Hips are the joint the seat line is read from: the one joint that stays
 * above the gunwale all stroke, and the seat's own position along the shell. */
const SEAT_JOINTS = HIP_JOINTS;

/** Least seats whose spacing can fix a projective map's three parameters. */
export const MIN_SEATS_FOR_SOLVE = 3;
/** Least hip samples behind one seat's mean position. */
export const MIN_SEAT_SAMPLES = 12;
/** Fit residual, as a fraction of the projected seat spacing, that still reads
 * as one straight equally spaced line. */
export const MAX_RESIDUAL_FRACTION = 0.02;
/** Azimuth beyond which a launch view of a shell stops being believable: the
 * far seats are then so foreshortened that their hips are not worth trusting. */
export const MAX_AZIMUTH = 60;
/** Azimuth below which the seat line's perspective is inside the keypoint noise,
 * however clean the clip looks. */
export const MIN_AZIMUTH = 1;
/** Azimuth at which the perspective counts as fully conditioned, for confidence:
 * below it the vanishing point runs away and a pixel of noise is degrees. */
const CONDITIONED_AZIMUTH = 12;
/** Degrees of azimuth that an unverified assumption — a principal point taken as
 * the frame centre, a camera that turned between the halves of the clip — may
 * cost before the solve stops trusting itself. Boat-frame angles tolerate a few
 * degrees; a solve that cannot promise that is not worth reporting. */
const CUE_TOLERANCE_DEGREES = 6;
/** 35 mm equivalent frame width, so a phone's "26 mm" becomes pixels. */
const FULL_FRAME_WIDTH_MM = 36;

/** Focal length in pixels, or null when the calibration does not give one. */
export function focalPx(
  calibration: BoatCalibration,
  width: number,
): number | null {
  if (calibration.focalPx !== undefined && calibration.focalPx !== null)
    return calibration.focalPx > 0 ? calibration.focalPx : null;
  const f35 = calibration.focal35mm;
  if (f35 === undefined || f35 === null || !(f35 > 0)) return null;
  // A 35 mm equivalent is a statement about the horizontal angle of view of a
  // 36 mm frame, so it converts with the image width alone.
  return (f35 * width) / FULL_FRAME_WIDTH_MM;
}

/** One seat's mean image position, and how much of the clip it came from. */
export interface SeatMark {
  seat: number;
  /** Mean image column of the hip midpoint over the clip. */
  u: number;
  /** Mean image row, for the seat line's direction in the image. */
  v: number;
  samples: number;
}

/**
 * Mean hip position per seat.
 *
 * Averaged over the whole clip rather than read off one frame, because a seat
 * slides some 35 cm along the shell during a stroke: over complete cycles every
 * seat slides by the same amount, so the *spacing* of the means is the true
 * spacing even though no single frame's hips are where the seats are. Both hips
 * are averaged together, which cancels their lateral offset — a hip is 16 cm off
 * the centreline, and at 16 m that is worth a degree of azimuth.
 */
export function seatMarks(
  frames: readonly PoseFrame[],
  minConfidence = 0.5,
): SeatMark[] {
  const by = new Map<number, { u: number; v: number; n: number }>();
  for (const f of frames) {
    const hips = SEAT_JOINTS.map((j) => f.points[j]).filter(
      (p) => seen(p) && p.confidence >= minConfidence,
    );
    if (hips.length < SEAT_JOINTS.length) continue;
    const at = by.get(f.seat) ?? { u: 0, v: 0, n: 0 };
    at.u += hips.reduce((s, p) => s + p.x, 0) / hips.length;
    at.v += hips.reduce((s, p) => s + p.y, 0) / hips.length;
    at.n++;
    by.set(f.seat, at);
  }
  return [...by.entries()]
    .filter(([, m]) => m.n >= MIN_SEAT_SAMPLES)
    .map(([seat, m]) => ({ seat, u: m.u / m.n, v: m.v / m.n, samples: m.n }))
    .sort((p, q) => p.seat - q.seat);
}

/**
 * Least-squares fit of u(x) = (a·x + b) / (c·x + 1) to seats at known spacing.
 *
 * Rearranged, u·(c·x + 1) = a·x + b is linear in (a, b, c):
 * −x·a − b + (u·x)·c = −u. Normal equations on three unknowns, solved exactly.
 * The residual is what says whether these seats really were one equally spaced
 * straight line, which is the whole claim the solve rests on.
 */
export function fitSeatLine(
  xs: readonly number[],
  us: readonly number[],
): { a: number; b: number; c: number; residual: number } | null {
  const n = xs.length;
  if (n < MIN_SEATS_FOR_SOLVE) return null;
  const A = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ],
    y = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const row = [-xs[i], -1, us[i] * xs[i]],
      target = -us[i];
    for (let r = 0; r < 3; r++) {
      for (let col = 0; col < 3; col++) A[r][col] += row[r] * row[col];
      y[r] += row[r] * target;
    }
  }
  const x = solve3(A, y);
  if (!x) return null;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const den = x[2] * xs[i] + 1;
    if (Math.abs(den) < 1e-9) return null;
    const d = (x[0] * xs[i] + x[1]) / den - us[i];
    sum += d * d;
  }
  return { a: x[0], b: x[1], c: x[2], residual: Math.sqrt(sum / n) };
}

/** Gaussian elimination on a 3×3 system; null when it is singular. */
function solve3(A: number[][], y: number[]): [number, number, number] | null {
  const m = A.map((row, i) => [...row, y[i]]);
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let r = col + 1; r < 3; r++)
      if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
    if (Math.abs(m[pivot][col]) < 1e-12) return null;
    [m[col], m[pivot]] = [m[pivot], m[col]];
    for (let r = 0; r < 3; r++) {
      if (r === col) continue;
      const k = m[r][col] / m[col][col];
      for (let c = col; c < 4; c++) m[r][c] -= k * m[col][c];
    }
  }
  return [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]];
}

const rad = (deg: number) => (deg * Math.PI) / 180;
const deg = (r: number) => (r * 180) / Math.PI;

function cue(
  name: string,
  value: number,
  confidence: number,
  detail: string,
): CameraCue {
  return { name, value, confidence, detail };
}

/** An abstention, with a reason a user can act on. */
function unresolved(
  reason: string,
  axis: { x: number; y: number },
  cues: CameraCue[] = [],
): CameraSolve {
  return {
    azimuth: null,
    confidence: 0,
    source: "unresolved",
    axis,
    reason,
    distance: null,
    pxPerMetre: null,
    principal: null,
    residual: null,
    cues,
  };
}

/** Image-plane direction of the seat line, from the seat means. */
function imageAxis(marks: readonly SeatMark[]): { x: number; y: number } {
  if (marks.length < 2) return { x: 1, y: 0 };
  const dx = marks.at(-1)!.u - marks[0].u,
    dy = marks.at(-1)!.v - marks[0].v,
    n = Math.hypot(dx, dy);
  return n > 1 ? { x: dx / n, y: dy / n } : { x: 1, y: 0 };
}

/**
 * Solves the camera from the crew's own seat line.
 *
 * Needs three things a user can supply — the seat spacing, a focal length (or a
 * 35 mm equivalent), and confirmation that the clip is level — and at least
 * three seats whose hips the model found through a stroke. It returns an
 * azimuth, a depth and a pixels-per-metre with the cues behind them, or an
 * abstention naming what was missing.
 */
export function solveCamera(
  frames: readonly PoseFrame[],
  boat: Pick<Boat, "seats" | "bowLeft">,
  calibration: BoatCalibration,
  imageSize: { width: number; height: number },
): CameraSolve {
  const marks = seatMarks(frames);
  const axis = imageAxis(marks);
  if (marks.length < MIN_SEATS_FOR_SOLVE)
    return unresolved(
      `Needs hips on at least ${MIN_SEATS_FOR_SOLVE} seats through a stroke to ` +
        `read the seat line; found ${marks.length}. One seat carries no spacing ` +
        `to measure perspective against.`,
      axis,
    );
  const spacing = calibration.seatSpacing;
  if (spacing === undefined || spacing === null || !(spacing > 0))
    return unresolved(
      "Needs the seat spacing in metres, off a rigging tape or the " +
        "manufacturer's spec. Without a known length on the shell the image " +
        "gives the shape of the seat line but not its scale.",
      axis,
      [cue("seat line", marks.length, 0.5, `${marks.length} seats found`)],
    );
  const f = focalPx(calibration, imageSize.width);
  if (f === null)
    return unresolved(
      "Needs a focal length in pixels, or a 35 mm equivalent to convert. The " +
        "seat line's vanishing point fixes focal ÷ tan(azimuth) and nothing " +
        "else, so without one of them azimuth and distance stay locked " +
        "together.",
      axis,
      [cue("seat line", marks.length, 0.5, `${marks.length} seats found`)],
    );

  // Seats are numbered from the bow, and +x is towards the rowers' feet — the
  // stern — so seat 1 sits at the negative end of the seat line. Which end of
  // the *image* that is comes from the hips; `bowLeft` is a cross-check on it.
  const xs = marks.map((m, i) => (i - (marks.length - 1) / 2) * spacing);
  const us = marks.map((m) => m.u);
  const fit = fitSeatLine(xs, us);
  if (!fit) return unresolved("The seat line could not be fitted.", axis);

  const cues: CameraCue[] = [];
  const spacingPx = Math.abs(us.at(-1)! - us[0]) / (marks.length - 1);
  if (!(fit.residual <= MAX_RESIDUAL_FRACTION * spacingPx))
    return unresolved(
      `The seat line does not fit one equally spaced straight line: residual ` +
        `${fit.residual.toFixed(1)} px against a ${spacingPx.toFixed(0)} px ` +
        `spacing. Either a seat's track is wrong — check the crew timing view — ` +
        `or the camera moved during the clip and one azimuth does not describe ` +
        `it.`,
      axis,
      [
        cue(
          "seat line",
          fit.residual,
          0,
          `residual ${fit.residual.toFixed(2)} px of ${spacingPx.toFixed(0)} px spacing`,
        ),
      ],
    );
  cues.push(
    cue(
      "seat line",
      fit.residual,
      1 - fit.residual / (MAX_RESIDUAL_FRACTION * spacingPx),
      `${marks.length} seats at ${spacing} m, ${Math.round(
        median(marks.map((m) => m.samples)) ?? 0,
      )} hips each, residual ${fit.residual.toFixed(2)} px of ${spacingPx.toFixed(0)} px`,
    ),
  );

  // A shell parallel to the image plane: no perspective in the spacing at all.
  if (Math.abs(fit.c) < 1e-7 || !Number.isFinite(fit.a / fit.c))
    return unresolved(
      "The shell is parallel to the image plane — a square side view — so the " +
        "seat spacing is uniform in the image and azimuth and distance cannot " +
        "be separated. Film from an angle, or enter an azimuth measured some " +
        "other way.",
      axis,
      cues,
    );
  const vanishing = fit.a / fit.c;

  // Which end of the image the bow is at follows from the seat numbers and the
  // measured columns. A user who says otherwise has either the bow direction or
  // the seat order wrong, and the solve would come out mirrored.
  const bowToLeft = us[0] < us.at(-1)!;
  cues.push(
    cue(
      "bow direction",
      bowToLeft ? 0 : 1,
      boat.bowLeft === bowToLeft ? 1 : 0.2,
      boat.bowLeft === bowToLeft
        ? `seat 1 is on the ${bowToLeft ? "left" : "right"}, as the boat says`
        : `seat 1 is on the ${bowToLeft ? "left" : "right"} but the boat says ` +
            `the bow is on the ${boat.bowLeft ? "left" : "right"}, which would ` +
            `mirror the whole solve`,
    ),
  );

  // The principal point is not in the seat line: b is the image column of the
  // line's own origin, wherever the crew sat along the shell. So it comes from
  // the calibration or is assumed at the frame centre — and the assumption is
  // priced here, in degrees, instead of being hidden.
  const given =
    calibration.cx !== undefined && calibration.cx !== null
      ? calibration.cx
      : null;
  const cx = given ?? imageSize.width / 2;
  const offCentre = imageSize.width * 0.05;
  const swing =
    Math.abs(
      deg(Math.atan(f / (cx + offCentre - vanishing))) -
        deg(Math.atan(f / (cx - offCentre - vanishing))),
    ) / 2;
  cues.push(
    cue(
      "principal point",
      cx,
      given ? 1 : Math.max(0, 1 - swing / CUE_TOLERANCE_DEGREES),
      given
        ? `entered as ${given.toFixed(0)} px`
        : `assumed at the frame centre; ${offCentre.toFixed(0)} px of ` +
            `off-centre framing would be worth about ${swing.toFixed(1)}° here`,
    ),
  );

  // The boat's vertical is the camera's vertical only if the clip is level, and
  // nothing about a crew can say whether it is.
  if (!calibration.level)
    return unresolved(
      "Needs the clip confirmed level: the solve turns a lifted pose into boat " +
        "coordinates about a vertical it cannot measure, so a rolled clip would " +
        "tilt every angle by however far the horizon is off.",
      axis,
      cues,
    );

  const azimuth = deg(Math.atan(f / (cx - vanishing)));
  if (!(Math.abs(azimuth) <= MAX_AZIMUTH))
    return unresolved(
      `The seat line puts the camera ${Math.abs(azimuth).toFixed(0)}° off ` +
        `square, beyond the ${MAX_AZIMUTH}° this solve will report: the far ` +
        `seats are too foreshortened to trust at that angle. Move closer to ` +
        `the beam.`,
      axis,
      cues,
    );
  if (Math.abs(azimuth) < MIN_AZIMUTH)
    return unresolved(
      `The camera is ${Math.abs(azimuth).toFixed(1)}° off square — close enough ` +
        `to a perfect side view that the seat line's perspective is inside the ` +
        `keypoint noise. Azimuth and distance cannot be separated from this ` +
        `clip.`,
      axis,
      cues,
    );

  // Depth of the seat line's centre, and the image scale there: c is
  // −sin(azimuth)/depth, and du/dx at the centre is a − b·c, wherever along the
  // shell the camera happened to be.
  //
  // Two things this number is not. It is the depth of the line the *hips* lie
  // on, not of the stretchers — a rower's hip sits some 70 cm from its own
  // seat's datum, which is worth a quarter of a metre of depth at 24°. And
  // pixels per metre is directional: a metre along the shell axis, foreshortened
  // by cos(azimuth) and by the depth changing as it runs away, is what the fit
  // measures; a metre across the boat, perpendicular to the line of sight, spans
  // f/depth pixels instead. Anything metric that leaves this module has to say
  // which of the two it meant.
  const distance = -Math.sin(rad(azimuth)) / fit.c;
  if (!(distance > 0) || !Number.isFinite(distance))
    return unresolved(
      "The seat line's perspective narrows towards the end the seat numbers say " +
        "is the stern, so the camera would have to be behind the shell: the " +
        "clip is mirrored, or the bow is at the other end of the image from " +
        "what the crew says. Check which way round the boat is.",
      axis,
      cues,
    );
  const pxPerMetre = Math.abs(fit.a - fit.b * fit.c);

  // The same solve over each half of the clip. A launch follows the shell, so
  // the camera both translates and pans, and one azimuth for the whole clip is
  // a fiction unless the halves agree.
  const halves = stability(frames, calibration, imageSize, xs);
  if (halves) cues.push(halves);

  // Confidence: as good as the weakest cue, tempered by how well conditioned the
  // perspective is. A minimum, not a product — several cues each 95% sound would
  // otherwise multiply down to a refusal about a solve that is plainly right.
  const conditioned = Math.min(
    1,
    Math.abs(Math.tan(rad(azimuth))) / Math.tan(rad(CONDITIONED_AZIMUTH)),
  );
  const confidence = Math.max(
    0,
    Math.min(1, conditioned * Math.min(...cues.map((c) => c.confidence))),
  );

  return {
    azimuth,
    confidence,
    source: confidence >= 0.8 ? "calibrated" : "unresolved",
    axis,
    reason:
      confidence >= 0.8
        ? `Seat line of ${marks.length} seats at ${spacing} m through a ` +
          `${f.toFixed(0)} px lens: ${Math.abs(azimuth).toFixed(1)}° off square ` +
          `with the ${azimuth > 0 ? "stern" : "bow"} end nearer, ` +
          `${distance.toFixed(1)} m out, ${pxPerMetre.toFixed(0)} px per metre ` +
          `at the shell.`
        : `The cues do not support an azimuth: confidence ` +
          `${Math.round(confidence * 100)}% of the 80% normalized angles need. ` +
          cues.map((c) => `${c.name}: ${c.detail}`).join(" "),
    distance,
    pxPerMetre,
    principal: { x: cx, y: imageSize.height / 2 },
    residual: fit.residual,
    cues,
  };
}

/**
 * How much the solve moves between the first and second half of the clip, or
 * null when either half is too short or too few seats to solve on its own — no
 * cue, rather than a made-up one.
 */
function stability(
  frames: readonly PoseFrame[],
  calibration: BoatCalibration,
  imageSize: { width: number; height: number },
  xs: readonly number[],
): CameraCue | null {
  const times = frames.map((f) => f.time);
  if (!times.length) return null;
  const mid = (Math.min(...times) + Math.max(...times)) / 2;
  const f = focalPx(calibration, imageSize.width);
  const cx = calibration.cx ?? imageSize.width / 2;
  const azs: number[] = [];
  for (const half of [
    frames.filter((frame) => frame.time <= mid),
    frames.filter((frame) => frame.time > mid),
  ]) {
    const hm = seatMarks(half);
    if (hm.length < MIN_SEATS_FOR_SOLVE || f === null) continue;
    const fit = fitSeatLine(
      xs.slice(0, hm.length),
      hm.map((m) => m.u),
    );
    if (!fit || Math.abs(fit.c) < 1e-7) continue;
    azs.push(deg(Math.atan(f / (cx - fit.a / fit.c))));
  }
  if (azs.length < 2) return null;
  const spread = Math.abs(azs[0] - azs[1]);
  return cue(
    "stability",
    spread,
    Math.max(0, 1 - spread / CUE_TOLERANCE_DEGREES),
    `first half ${azs[0].toFixed(1)}°, second ${azs[1].toFixed(1)}° — ` +
      `${spread.toFixed(1)}° apart`,
  );
}

/**
 * Whether normalized angles may be reported from this solve.
 *
 * The bar is the one `metricVisibility` in core/geometry.ts set: a calibrated
 * source, real confidence, an azimuth shallow enough that the lift is not being
 * asked to see along its own depth axis, and the scale that turns a lifted pose
 * into metres.
 */
export function solveAllowsAngles(solve: CameraSolve): boolean {
  return (
    solve.source === "calibrated" &&
    solve.confidence >= 0.8 &&
    solve.azimuth !== null &&
    Math.abs(solve.azimuth) < MAX_AZIMUTH &&
    solve.distance !== null &&
    solve.pxPerMetre !== null
  );
}

/** Median samples per seat: how much of the clip the solve was built from. */
export function solveSupport(marks: readonly SeatMark[]): number | null {
  return median(marks.map((m) => m.samples));
}
