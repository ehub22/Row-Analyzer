import type { OarCandidate, PoseFrame } from "./types";
import { reliable } from "./calibration";
export interface EdgeImage {
  width: number;
  height: number;
  gx: Float32Array;
  gy: Float32Array;
}
export function imageEdges(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
): EdgeImage {
  const gray = new Float32Array(width * height),
    gx = new Float32Array(gray.length),
    gy = new Float32Array(gray.length);
  for (let i = 0; i < gray.length; i++)
    gray[i] =
      rgba[i * 4] * 0.299 + rgba[i * 4 + 1] * 0.587 + rgba[i * 4 + 2] * 0.114;
  for (let y = 1; y < height - 1; y++)
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      gx[i] = gray[i + 1] - gray[i - 1];
      gy[i] = gray[i + width] - gray[i - width];
    }
  return { width, height, gx, gy };
}
const delta = (a: number, b: number) =>
  Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));
/** Appearance-supported line tracker, NOT a semantic blade detector. No extrapolated tracks. */
export class OarTracker {
  private previous = new Map<
    string,
    { candidate: OarCandidate; time: number }
  >();
  update(
    image: EdgeImage,
    frames: PoseFrame[],
    videoWidth: number,
    videoHeight: number,
  ) {
    const sx = image.width / videoWidth,
      sy = image.height / videoHeight;
    for (const f of frames) {
      f.oars = [];
      for (const [joint, side] of [
        [9, "left"],
        [10, "right"],
      ] as const) {
        const key = `${f.seat}:${side}`,
          p = f.points[joint],
          prior = this.previous.get(key);
        if (f.identityConfidence < 0.6 || !reliable(p)) {
          this.previous.delete(key);
          continue;
        }
        const x = p.x * sx,
          y = p.y * sy;
        const minimum = Math.max(35, f.box.height * sy * 0.65),
          maximum = Math.min(image.width * 0.5, minimum * 2.5);
        const candidates: {
          angle: number;
          score: number;
          length: number;
          offset: number;
        }[] = [];
        // Search a thin ridge originating near (not necessarily at) the observed wrist.
        for (let a = 0; a < 360; a += 3) {
          const angle = (a * Math.PI) / 180,
            dx = Math.cos(angle),
            dy = Math.sin(angle);
          let best = { angle, score: 0, length: 0, offset: 0 };
          for (const offset of [-4, -2, 0, 2, 4]) {
            let hits = 0,
              samples = 0,
              run = 0,
              maxRun = 0,
              length = 0;
            for (let t = 12; t < maximum; t += 2) {
              const px = Math.round(x + dx * t - dy * offset),
                py = Math.round(y + dy * t + dx * offset);
              if (
                px < 2 ||
                py < 2 ||
                px >= image.width - 2 ||
                py >= image.height - 2
              )
                break;
              samples++;
              let edge = false;
              for (const n of [-2, 0, 2]) {
                const nx = Math.round(px - dy * n),
                  ny = Math.round(py + dx * n),
                  i = ny * image.width + nx;
                if (nx < 0 || ny < 0 || nx >= image.width || ny >= image.height)
                  continue;
                const mag = Math.hypot(image.gx[i], image.gy[i]);
                if (
                  mag > 35 &&
                  Math.abs(-dy * image.gx[i] + dx * image.gy[i]) / mag > 0.85
                )
                  edge = true;
              }
              if (edge) {
                hits++;
                run++;
                length = t;
              } else run = 0;
              maxRun = Math.max(maxRun, run);
            }
            const score = samples ? hits / samples : 0;
            if (length >= minimum && maxRun >= 8 && score > best.score)
              best = { angle, score, length, offset };
          }
          if (best.score > 0.65) candidates.push(best);
        }
        candidates.sort((a, b) => b.score - a.score);
        const best = candidates[0];
        if (!best) {
          this.previous.delete(key);
          continue;
        }
        const rival = candidates.find(
          (c) => delta(c.angle, best.angle) > Math.PI / 9,
        );
        if (rival && best.score - rival.score < 0.08) {
          this.previous.delete(key);
          continue;
        }
        const dx = Math.cos(best.angle),
          dy = Math.sin(best.angle);
        const oldAngle = prior
          ? Math.atan2(
              (prior.candidate.end.y - prior.candidate.start.y) * sy,
              (prior.candidate.end.x - prior.candidate.start.x) * sx,
            )
          : 0;
        const continuous =
          prior &&
          f.time - prior.time <= 0.2 &&
          f.time > prior.time &&
          delta(oldAngle, best.angle) < Math.PI / 6;
        const age = continuous ? prior.candidate.age + 1 : 1;
        const candidate: OarCandidate = {
          side,
          start: {
            x: (x - dy * best.offset) / sx,
            y: (y + dx * best.offset) / sy,
          },
          end: {
            x: (x + dx * best.length - dy * best.offset) / sx,
            y: (y + dy * best.length + dx * best.offset) / sy,
          },
          confidence: Math.min(0.6, best.score * (age >= 3 ? 0.7 : 0.4)),
          support: best.score,
          age,
          source: "image-line-candidate",
        };
        f.oars.push(candidate);
        this.previous.set(key, { candidate, time: f.time });
      }
    }
    return frames;
  }
}
