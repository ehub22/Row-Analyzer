import type { Boat, Point, PoseFrame } from "./types";
import { OCCLUDED_KNEE_FLAG, seen } from "./occlusion";
import { median, quantile } from "./geometry";
import { THIGHS, TROUGH_FRACTION, thighElevation } from "./strokes";

// Knees at the finish.
//
// The catch is the easy end of the stroke: the knee is the highest point of
// the leg, clear of the hull and clear of the loom, and pose models place it
// well. The finish is not. With the legs flat the knee drops to the gunwale
// and the oar crosses the legs on its way out, so the joint is partly covered
// for a large part of every cycle — and a pose model still answers. It answers
// with the hull edge (a knee pinned to a line it cannot be below) or with the
// shaft (a knee dragged up onto the oar), at a confidence that routinely
// clears the 0.3 observed threshold.
//
// Tracked as observations, those answers are worse than useless. Every stroke
// event is timed from thigh elevation, measured hip→knee, so a knee dragged up
// by a few centimetres lifts the trough of the cycle and a knee pinned to the
// gunwale flattens it: in the simulation the finish proxy then wandered over a
// quarter of the cycle, and the guessed knee was tracked as an observation in
// every frame the hull or the loom covered it — 16–21 px from where the knee
// actually was (tests/knees.test.ts). A drag towards the handle instead of
// across the shaft doubles the reported rate outright.
//
// Unlike the feet, this cannot be fixed by declaring a joint hidden for the
// whole session (see core/occlusion.ts): the same knee is perfectly visible a
// tenth of a second earlier, at the catch, and that sample is the one the
// stroke timing most needs. So the knee is judged only where it can be judged,
// and against the seat's own clear-view knee:
//
//   * the top of a seat's thigh-elevation range is the catch, which neither
//     the hull nor the loom reaches, so those frames calibrate what a visible
//     knee looks like on this footage: its confidence and its thigh length;
//   * below that band a frame is a *candidate* when its knee — both legs
//     together, since a side view sees one joint behind one occluder — is
//     weaker than that clear view, and candidates are only believed where they
//     are dense: an occluder covers the joint through a window of tenths of a
//     second, while a clear knee fails a threshold now and then on its own;
//   * each dense window is confirmed as a whole and trimmed back to the frames
//     that still look occluded, so a window neither swallows the visible ends
//     of the slide nor leaves a guess standing at its edge;
//   * a model does not answer the same covered knee the same way twice, so a
//     crossing it answers inconsistently — the hull edge on one frame, nothing
//     on the next — is read as one window rather than as runs too short to
//     believe, and where an occluder dominates the low part of the cycle the
//     weak frames between its windows are that occluder answered differently
//     and are discarded too.
//
// Discarded knees become explicit missing points with a reason, exactly like
// hidden feet: not gap-filled, not counted as evidence, not drawn as
// observations. The finish they would have mistimed is taken from the bridged
// trough instead (see `MAX_TROUGH_GAP` in core/strokes.ts), which is where it
// already comes from whenever a hull drops the knee rather than guessing it.

/** COCO-WholeBody knees. */
export const KNEE_JOINTS: readonly number[] = [13, 14];

/** Flag carried by every knee discarded as an occlusion answer. Declared in
 * core/occlusion.ts with the other expected-occlusion flags, which the evidence
 * gate and the viewer read. */
export { OCCLUDED_KNEE_FLAG };

/**
 * Fraction of a seat's thigh-elevation range, measured up from its lowest,
 * inside which the hull or the loom could be covering the knee. Generous on
 * purpose: an occluder moves the reported knee *up* — onto the gunwale edge or
 * onto the shaft — so a guess can sit well above the true trough. What has to
 * stay outside the band is the catch, and the catch is the top of the range.
 */
export const AT_RISK_FRACTION = 0.6;

/**
 * A knee inside the band is an occlusion candidate below this fraction of the
 * confidence the same joint carries when it is clearly visible near the catch,
 * or with an apparent thigh that far off the seat's own. The two confidence
 * distributions overlap frame by frame — an occluded knee is often reported at
 * a perfectly ordinary confidence, and the far knee of a side view is simply
 * the weaker knee — so no single frame is believed on this alone; see
 * `MIN_DENSITY`. Pooling both legs halves that noise.
 */
export const CLEAR_CONFIDENCE_RATIO = 0.9;

/** Tolerance on the apparent hip→knee length inside the band, as a fraction of
 * the seat's clear-view thigh. Loose enough for the perspective and box-scale
 * change between the catch and the finish, tight enough for a knee reported on
 * the loom, which slides along the shaft and so changes the thigh. */
export const THIGH_TOLERANCE = 0.18;

/** Half-width, in seconds, of the neighbourhood a candidate density is measured
 * over (±2 frames at 30 fps, ±4 at 60 fps). Wide enough that the share below is
 * measured over several frames, so one clear frame inside a window cannot split
 * it in two. */
export const DENSITY_HALF_WIDTH = 0.08;

/** Share of that neighbourhood that must be candidates before a frame is taken
 * to be inside an occlusion window. A clear knee fails the thresholds in about a
 * sixth of its frames, so chance alone rarely sustains a majority over five of
 * them and then over a whole window; a hull or a loom crossing the joint fails
 * them in nearly every frame, so one clear sample inside a window neither hides
 * the window nor splits it in two. */
export const MIN_DENSITY = 0.5;

/** Seconds a dense window must span, and judgeable frames it must hold, to be
 * confirmed and discarded. */
export const MIN_WINDOW = 0.15;
export const MIN_WINDOW_SAMPLES = 4;

/**
 * Share of a window's frames the model must have reported a knee on for the
 * window to be about reporting at all. A crossing it dropped throughout has
 * nothing in it to discard, and the few frames it did report around the edges
 * of one are ordinary clear knees that happen to measure weak — the same chance
 * weakness `MIN_DENSITY` is sized against — not guesses at a covered joint. A
 * model that alternates between the two answers the crossing in roughly every
 * other frame, so a quarter leaves it well inside and an honest dropout well
 * outside.
 */
export const MIN_ANSWERED_SHARE = 0.25;

/** Share of the time the knee is low that confirmed windows must cover before
 * the weak frames *between* them are discarded too.
 *
 * A model does not answer a covered knee the same way twice: it finds the hull
 * edge on one frame, loses the joint on the next, and finds it again, so a
 * window confirms on part of a crossing and the rest of it is left standing as
 * scattered bumps in the trough. Where an occluder dominates the low part of
 * the cycle — a gunwale, which the knee is behind for half of every stroke —
 * those leftovers are the same occluder, and removing them leaves one clean
 * blind spot the finish is timed from.
 *
 * Where it does not — a loom crossing the legs for a seventh of the cycle —
 * the trough between crossings is genuinely visible, and discarding the odd
 * weak frame in it pokes holes in the samples the finish is read from. Chance
 * gives clear footage a window or two per clip, nowhere near this share. */
export const KNOWN_OCCLUSION_SHARE = 0.5;

/** Least elevation range (degrees) and frame count a seat needs before its
 * knees are judged at all. Without a real stroke in view there is nothing to
 * calibrate against, and the honest answer is to leave the frames alone. */
const MIN_RANGE = 12;
const MIN_SAMPLES = 60;
/** Least clear-view samples of one knee needed to judge that knee. */
const MIN_CLEAR = 8;

/** What a clearly visible knee looks like on one seat. */
export interface KneeClear {
  /** Median confidence of this knee above the at-risk band. */
  confidence: number;
  /** Median hip→knee length as a fraction of the seat's median box height. */
  thigh: number | null;
}

export interface KneeCheck {
  /** Thigh elevation (degrees) at or below which the knee may be covered. */
  bandLine: number;
  /** Thigh elevation (degrees) at or below which core/strokes.ts will bridge a
   * blind spot as a hull trough. Nothing above it may be discarded on a hunch,
   * or a blind spot would end too high to bridge and the cycle would split. */
  troughLine: number;
  /** Median detection-box height of the seat: the length scale of `thigh`. */
  boxScale: number;
  /** Clear-view reference per knee joint. A joint the seat never saw clearly
   * enough is absent, and is then never judged. */
  clear: Record<number, KneeClear>;
  /** Frames the check was calibrated on. */
  samples: number;
}

/** One frame's evidence about the knee, both legs together. */
export interface KneeSample {
  /** Legs this frame could be judged on. */
  legs: number;
  /** Reported confidence ÷ clear-view confidence, over the judgeable legs
   * (their median: for two legs their mean). Null without one. */
  ratio: number | null;
  /** Relative deviation of the apparent thigh from the clear-view thigh, over
   * the judgeable legs. */
  thighDev: number | null;
}

/**
 * Whether this boat setup can put the hull or the loom in front of a knee.
 *
 * The session already answers this for the feet: a hull that hides feet
 * strapped to the stretcher also hides a knee that drops to the gunwale with
 * the legs flat, and the oar crossing those legs is only there on the water.
 * Ergometer and overhead footage (setup *Feet in view: Visible*) has neither,
 * so its knees are left exactly as the model reported them.
 */
export function kneesAtRisk(boat: Pick<Boat, "feetHidden">): boolean {
  return boat.feetHidden !== false;
}

/**
 * Calibrates one seat: where its at-risk band starts, and what its knee looks
 * like while it is clearly visible. Null when the seat has too little signal
 * to calibrate on, which leaves its frames untouched.
 */
export function kneeCheck(seq: readonly PoseFrame[]): KneeCheck | null {
  const elev = seq.map(thighElevation);
  const known = elev
    .filter((x): x is number => x !== null)
    .sort((a, b) => a - b);
  if (known.length < MIN_SAMPLES) return null;
  const lo = quantile(known, 0.05),
    hi = quantile(known, 0.95);
  if (!(hi - lo >= MIN_RANGE)) return null;
  const boxScale = median(seq.map((f) => f.box.height).filter((h) => h > 0));
  if (boxScale === null || !(boxScale > 0)) return null;
  const bandLine = lo + AT_RISK_FRACTION * (hi - lo),
    troughLine = lo + TROUGH_FRACTION * (hi - lo);
  const clear: Record<number, KneeClear> = {};
  for (const [hip, knee] of THIGHS) {
    const confidences: number[] = [],
      thighs: number[] = [];
    for (let i = 0; i < seq.length; i++) {
      // Above the band is the catch end of the stroke, where the knee is the
      // highest point of the leg: what the model reports there is what it saw.
      if (elev[i] === null || elev[i]! <= bandLine) continue;
      const h = seq[i].points[hip],
        k = seq[i].points[knee];
      if (!seen(h) || !seen(k)) continue;
      confidences.push(k.confidence);
      thighs.push(Math.hypot(k.x - h.x, k.y - h.y) / boxScale);
    }
    const confidence = median(confidences),
      thigh = median(thighs);
    if (confidence === null || confidences.length < MIN_CLEAR) continue;
    clear[knee] = { confidence, thigh };
  }
  return Object.keys(clear).length
    ? { bandLine, troughLine, boxScale, clear, samples: seq.length }
    : null;
}

/**
 * This frame's evidence about the knee, pooled over the legs that can be judged.
 *
 * Both knees are the same joint in a side view, a few centimetres apart in
 * depth, behind the same gunwale and the same loom, so their evidence belongs
 * together: pooling it halves the noise of a single leg, and the far knee of a
 * side view — permanently the weaker one — cannot convict the joint alone.
 */
export function kneeSample(
  check: KneeCheck,
  points: readonly Point[],
  boxScale = check.boxScale,
): KneeSample {
  const ratios: number[] = [],
    devs: number[] = [];
  let legs = 0;
  for (const [hip, knee] of THIGHS) {
    const clear = check.clear[knee],
      h = points[hip],
      k = points[knee];
    // A knee the model already dropped, or a leg this seat could not
    // calibrate, needs no second opinion.
    if (!clear || !seen(h) || !seen(k)) continue;
    legs++;
    ratios.push(k.confidence / clear.confidence);
    if (clear.thigh !== null)
      devs.push(
        Math.abs(Math.hypot(k.x - h.x, k.y - h.y) / boxScale / clear.thigh - 1),
      );
  }
  return {
    legs,
    ratio: median(ratios),
    thighDev: median(devs),
  };
}

const weakSample = (s: KneeSample) =>
  s.ratio !== null &&
  (s.ratio < CLEAR_CONFIDENCE_RATIO || (s.thighDev ?? 0) > THIGH_TOLERANCE);

/** Whether confirmed windows cover most of the time this seat's knee is low. */
function dominates(
  windows: readonly [number, number][],
  times: readonly number[],
  inBand: (i: number) => boolean,
): boolean {
  if (!windows.length) return false;
  let low = 0;
  for (let i = 0; i < times.length; i++) {
    if (!inBand(i)) continue;
    const next = times[i + 1];
    low += next === undefined ? 0 : next - times[i];
  }
  const covered = windows.reduce(
    (sum, [a, b]) => sum + (times[b] - times[a]),
    0,
  );
  return low > 0 && covered >= KNOWN_OCCLUSION_SHARE * low;
}

/** Maximal runs where candidates are dense, at least `MIN_WINDOW` long. */
function denseWindows(
  candidate: readonly boolean[],
  times: readonly number[],
): [number, number][] {
  const dense = candidate.map((_, i) => {
    let n = 0,
      c = 0;
    for (
      let k = i;
      k >= 0 && times[i] - times[k] <= DENSITY_HALF_WIDTH + 1e-9;
      k--
    )
      (n++, candidate[k] && c++);
    for (
      let k = i + 1;
      k < candidate.length && times[k] - times[i] <= DENSITY_HALF_WIDTH + 1e-9;
      k++
    )
      (n++, candidate[k] && c++);
    return c / Math.max(1, n) >= MIN_DENSITY;
  });
  const out: [number, number][] = [];
  let a = -1;
  for (let i = 0; i <= dense.length; i++) {
    if (i < dense.length && dense[i]) {
      if (a < 0) a = i;
      continue;
    }
    if (a >= 0 && times[i - 1] - times[a] >= MIN_WINDOW) out.push([a, i - 1]);
    a = -1;
  }
  return out;
}

export interface KneeReport {
  frames: PoseFrame[];
  /** Knee samples replaced by flagged missing points. */
  discarded: number;
  /** Confirmed occlusion windows. */
  windows: number;
  /** Seats whose knees could be judged at all. */
  judged: number;
  /** Seats in the run. */
  seats: number;
}

/**
 * Discards the knees the hull or the loom was in front of.
 *
 * Manual points are kept: a person who can see a knee in a frame outranks the
 * expectation, exactly as for hidden feet. So are the frames of a seat that
 * could not be calibrated, and every frame of a session that says the feet —
 * and so the hull — are not in the way.
 */
export function discardOccludedKnees(
  input: readonly PoseFrame[],
  boat: Pick<Boat, "feetHidden">,
): KneeReport {
  const frames = input.slice();
  const seats = [...new Set(input.map((f) => f.seat))];
  const report: KneeReport = {
    frames,
    discarded: 0,
    windows: 0,
    judged: 0,
    seats: seats.length,
  };
  if (!kneesAtRisk(boat)) return report;
  for (const seat of seats) {
    const at = input
      .map((f, i) => (f.seat === seat ? i : -1))
      .filter((i) => i >= 0)
      .sort((p, q) => input[p].time - input[q].time);
    const seq = at.map((i) => input[i]);
    const check = kneeCheck(seq);
    if (!check) continue;
    report.judged++;
    const times = seq.map((f) => f.time),
      elev = seq.map(thighElevation),
      samples = seq.map((f) => kneeSample(check, f.points));
    // Nearest frame either side whose elevation is known. A frame whose knees
    // are both gone has none of its own, but the stroke is continuous, so it
    // sits where its neighbours sit: without this, a model that alternates
    // between guessing a covered knee and dropping it breaks its own window in
    // two, and neither half confirms.
    const knownBefore: number[] = [],
      knownAfter: number[] = [];
    for (let i = 0, k = -1; i < elev.length; i++) {
      if (elev[i] !== null) k = i;
      knownBefore.push(k);
    }
    for (let i = elev.length - 1, k = -1; i >= 0; i--) {
      if (elev[i] !== null) k = i;
      knownAfter[i] = k;
    }
    const low = (i: number) => elev[i] !== null && elev[i]! <= check.bandLine;
    const inBand = (i: number) => {
      if (elev[i] !== null) return low(i);
      const p = knownBefore[i],
        q = knownAfter[i];
      if (p < 0) return q >= 0 && low(q);
      if (q < 0) return low(p);
      return low(p) && low(q);
    };
    // A knee the model dropped in the band is occlusion behaviour too: the same
    // window, answered differently. Counting it keeps a window that alternates
    // between a guess and a dropout from breaking in two, which neither half
    // would then confirm on its own.
    const dropped = seq.map((f) =>
      THIGHS.some(
        ([hip, knee]) => seen(f.points[hip]) && !seen(f.points[knee]),
      ),
    );
    const candidate = samples.map(
      (s, i) => inBand(i) && (weakSample(s) || dropped[i]),
    );
    if (!candidate.some(Boolean)) continue;
    const confirmed: [number, number][] = [];
    for (let [a, b] of denseWindows(candidate, times)) {
      // Confirm the window as a whole: an occluder crossing the joint makes
      // all of it weaker than this seat's clear-view knee, while a run of
      // ordinary frames on clear footage measures like the clear view it is.
      const ratios: number[] = [],
        devs: number[] = [];
      let answered = 0;
      for (let i = a; i <= b; i++) {
        const s = samples[i];
        if (s.legs) answered++;
        if (s.ratio !== null) ratios.push(s.ratio);
        if (s.thighDev !== null) devs.push(s.thighDev);
      }
      const ratio = median(ratios),
        dev = median(devs);
      // A window the model reported a knee on for barely any of its frames is
      // not about reporting: see MIN_ANSWERED_SHARE.
      if (answered < MIN_ANSWERED_SHARE * (b - a + 1)) continue;
      if (!(
        (ratios.length >= MIN_WINDOW_SAMPLES &&
          ratio !== null &&
          ratio < CLEAR_CONFIDENCE_RATIO) ||
        (devs.length >= MIN_WINDOW_SAMPLES &&
          dev !== null &&
          dev > THIGH_TOLERANCE)
      ))
        continue;
      confirmed.push([a, b]);
      // Trim the visible ends of the slide back out of the window. A frame the
      // knee measures clear on is not behind an occluder, and keeping it would
      // raise the end of the blind spot out of the trough the finish is timed
      // from; a frame it still measures weak on is a guess, however
      // convincing, and leaving it standing would put a bump back into the
      // trough the window was confirmed on.
      while (a < b && !candidate[a]) a++;
      while (b > a && !candidate[b]) b--;
      report.windows++;
      discard(a, b);
    }
    // An occluder that dominates the low part of the cycle (see
    // KNOWN_OCCLUSION_SHARE) answered the frames its windows did not cover the
    // same way, so they go too — but only down to the trough line, never above
    // it: a blind spot that ended higher would not be bridged, and the cycle
    // would split.
    if (dominates(confirmed, times, inBand))
      for (let i = 0; i < seq.length; i++)
        if (
          elev[i] !== null &&
          elev[i]! <= check.troughLine &&
          weakSample(samples[i]) &&
          !covered(i)
        )
          discard(i, i);
    /** Replaces both knees of the frames `a`…`b` with flagged missing points. */
    function discard(a: number, b: number) {
      for (let i = a; i <= b; i++) {
        const points = seq[i].points.slice();
        let touched = false;
        for (const knee of KNEE_JOINTS) {
          const p = points[knee];
          if (!p || p.origin === "missing" || p.origin === "manual") continue;
          points[knee] = {
            x: 0,
            y: 0,
            confidence: 0,
            origin: "missing",
            flags: [...new Set([...(p.flags ?? []), OCCLUDED_KNEE_FLAG])],
          };
          touched = true;
          report.discarded++;
        }
        if (touched) frames[at[i]] = { ...seq[i], points };
      }
    }
    /** Whether frame i already sits inside a confirmed window. */
    function covered(i: number) {
      return confirmed.some(([a, b]) => i >= a && i <= b);
    }
  }
  return report;
}

/** One line of analysis notes for what this pass did, or null when it did nothing. */
export function describeKneeOcclusion(report: KneeReport): string | null {
  if (!report.discarded) return null;
  return (
    `Knees at the finish: ${report.discarded} knee samples in ${report.windows} ` +
    `occlusion windows (${report.judged} of ${report.seats} seats judgeable) were ` +
    `discarded as gunwale or oar occlusion. With the legs flat the knee drops to the ` +
    `hull edge and the loom crosses it, and the model still answers — pinned to the ` +
    `gunwale, or dragged onto the shaft — at a confidence that clears the observed ` +
    `threshold. Those answers are not tracked, counted as evidence, or drawn. The catch ` +
    `end of the stroke, where the knee is clear of both, is left untouched and each ` +
    `finish is timed from the bridged trough instead of from a guess.`
  );
}
