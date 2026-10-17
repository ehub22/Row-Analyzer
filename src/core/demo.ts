import type { Session, PoseFrame, Point, Boat } from "./types";
import { defaultBoat, missing } from "./types";
import { segmentStrokes } from "./strokes";
import { hiddenJoints, suppressHiddenJoints } from "./occlusion";
import { discardOccludedKnees } from "./knees";

/** Thigh and shank of the procedural rower, in pixels: constant, as a leg is. */
const LEG = 85;

/**
 * Knee of a two-link leg between `ankle` and `hip`, bowing upwards in the
 * image (the same rig as tests/fixtures/rowingSim.ts). Past full extension the
 * leg straightens along the hip–ankle line, which is all a real leg can do.
 *
 * The demo used to slide its knee along a straight line, which made the thigh
 * grow by a quarter of its length between the catch and the finish. A leg that
 * changes length is not a leg, and it is exactly what the knee occlusion pass
 * looks for (core/knees.ts measures the apparent thigh against the seat's own),
 * so the procedural crew is now built from rigid segments.
 */
function kneeBetween(
  hip: { x: number; y: number },
  ankle: { x: number; y: number },
  thigh = LEG,
  shank = LEG,
): { x: number; y: number } {
  const dx = hip.x - ankle.x,
    dy = hip.y - ankle.y,
    span = Math.hypot(dx, dy) || 1e-6,
    d = Math.min(Math.max(span, Math.abs(thigh - shank) + 1e-3), thigh + shank),
    ux = dx / span,
    uy = dy / span,
    a = (thigh * thigh - shank * shank + d * d) / (2 * d),
    h = Math.sqrt(Math.max(0, thigh * thigh - a * a));
  return { x: ankle.x + a * ux + h * uy, y: ankle.y + a * uy - h * ux };
}

export function createDemo(boat: Boat = defaultBoat): Session {
  const built: PoseFrame[] = [],
    timestamps: number[] = [];
  // Behave like an on-water session with the same setup: when the feet are
  // declared hidden, the generated foot joints are discarded exactly as the
  // analysis worker discards a pose model's guesses for them, and the knees go
  // through the same occlusion pass — which, on a clear procedural crew, finds
  // nothing to discard (tests/knees.test.ts pins that).
  const hidden = hiddenJoints(boat);
  const duration = 12,
    fps = 30,
    width = 1440,
    height = 720;
  for (let i = 0; i < duration * fps; i++) {
    const time = i / fps;
    timestamps.push(time);
    for (let seat = 1; seat <= boat.seats; seat++) {
      const period = 2.4,
        lag = seat === boat.seats ? 0 : (((seat * 7) % 10) - 4) * 0.012,
        phase = ((((time - lag) % period) + period) % period) / period;
      const ext =
        phase < 0.4
          ? (1 - Math.cos((phase / 0.4) * Math.PI)) / 2
          : (1 + Math.cos(((phase - 0.4) / 0.6) * Math.PI)) / 2;
      const spacing = Math.min(145, 1100 / boat.seats),
        base = 190 + (seat - 1) * spacing,
        ankle = { x: base - 70, y: 470 },
        hip = { x: base + ext * 80, y: 405 },
        knee = kneeBetween(hip, ankle),
        shoulder = { x: hip.x - 34 + ext * 60, y: 310 },
        wrist = { x: base - 70 + ext * 88, y: 375 + ext * 4 },
        elbow = { x: (shoulder.x + wrist.x) / 2 + ext * 18, y: 350 };
      const points: Point[] = Array.from({ length: 133 }, missing);
      const set = (j: number, x: number, y: number) =>
        (points[j] = { x, y, confidence: 0.92, origin: "observed" });
      for (const [j, p] of [
        [5, shoulder],
        [7, elbow],
        [9, wrist],
        [11, hip],
        [13, knee],
        [15, ankle],
      ] as const) {
        set(j, p.x, p.y);
        set(j + 1, p.x + 8, p.y + 5);
      }
      set(0, shoulder.x - 3, shoulder.y - 26);
      [17, 18, 19].forEach((j, k) => set(j, ankle.x - 8 + k * 4, ankle.y + 5));
      [20, 21, 22].forEach((j, k) => set(j, ankle.x + k * 4, ankle.y + 10));
      [91, 112].forEach((baseHand, h) => {
        for (let j = 0; j < 21; j++)
          set(
            baseHand + j,
            wrist.x + h * 8 + (j % 4) * 2,
            wrist.y + h * 5 + Math.floor(j / 4) * 1.5,
          );
      });
      if (seat === 3 && i % 90 > 40 && i % 90 < 47) {
        points[9] = { ...points[9], origin: "inferred", confidence: 0.42 };
      }
      built.push({
        time,
        seat,
        points: suppressHiddenJoints(points, hidden),
        box: {
          x: base - 85,
          y: 280,
          width: 150,
          height: 195,
          confidence: 0.97,
        },
        identityConfidence: 0.98,
      });
    }
  }
  const frames = discardOccludedKnees(built, boat).frames;
  return {
    version: 1,
    name: "Crew study · synthetic demonstration",
    demo: true,
    width,
    height,
    duration,
    timestamps,
    boat: { ...boat, feetHidden: hidden.length > 0 },
    frames,
    strokes: segmentStrokes(frames),
    camera: {
      azimuth: 0,
      confidence: 1,
      source: "synthetic",
      axis: { x: 1, y: 0 },
      reason: "Known synthetic side view, not estimated from footage.",
    },
    corrections: [],
    warnings: [
      "SYNTHETIC DEMONSTRATION — not measured athlete performance.",
      "Timing values are calculated from generated poses; they are not an accuracy benchmark.",
      "Normalized 3D biomechanics are not implemented.",
    ],
    model: "Procedural rowing cycle (no AI inference)",
    provider: "Synthetic",
  };
}
