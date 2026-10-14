import type { Session, PoseFrame, Point, Boat } from "./types";
import { defaultBoat, missing } from "./types";
import { segmentStrokes } from "./strokes";
export function createDemo(boat: Boat = defaultBoat): Session {
  const frames: PoseFrame[] = [],
    timestamps: number[] = [];
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
        hip = { x: base + ext * 50, y: 410 },
        shoulder = { x: hip.x - 34 + ext * 60, y: 310 },
        ankle = { x: base - 60, y: 460 },
        knee = { x: base - 40 + ext * 20, y: 365 + ext * 60 },
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
      frames.push({
        time,
        seat,
        points,
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
  return {
    version: 1,
    name: "Crew study · synthetic demonstration",
    demo: true,
    width,
    height,
    duration,
    timestamps,
    boat,
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
