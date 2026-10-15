import type { PoseFrame, Stroke, Metric } from "./types";
import { evidence } from "./occlusion";
export const withheld = (unit: string, reason: string): Metric => ({
  value: null,
  confidence: 0,
  unit,
  reason,
});
// Compression signal: visible hip-to-ankle distance normalized by person height.
// This is image-space cycle timing, not a normalized joint-angle measurement.
export function segmentStrokes(frames: PoseFrame[]): Stroke[] {
  const result: Stroke[] = [];
  for (const seat of new Set(frames.map((f) => f.seat))) {
    const seq = frames
      .filter((f) => f.seat === seat)
      .sort((a, b) => a.time - b.time);
    if (seq.length < 10) continue;
    const raw = seq.map((f) => {
      const distances: number[] = [];
      for (const [hip, ankle] of [
        [11, 15],
        [12, 16],
      ]) {
        const a = f.points[hip],
          b = f.points[ankle];
        if (
          a.confidence > 0.3 &&
          b.confidence > 0.3 &&
          a.origin !== "missing" &&
          b.origin !== "missing"
        )
          distances.push(Math.hypot(a.x - b.x, a.y - b.y) / f.box.height);
      }
      return distances.length ? Math.max(...distances) : null;
    });
    const signal = raw.map((v, i) => {
      if (v === null) return null;
      const values = raw
        .slice(Math.max(0, i - 3), i + 4)
        .filter((x): x is number => x !== null);
      return values.reduce((a, b) => a + b, 0) / values.length;
    });
    const catches: number[] = [];
    for (let i = 2; i < signal.length - 2; i++) {
      const v = signal[i];
      if (v === null || signal[i - 1] === null || signal[i + 1] === null)
        continue;
      if (v <= signal[i - 1]! && v < signal[i + 1]!) {
        const last = catches.at(-1);
        if (last === undefined || seq[i].time - seq[last].time > 1.2) {
          catches.push(i);
        } else if (v < signal[last]!) catches[catches.length - 1] = i;
      }
    }
    for (let k = 0; k < catches.length - 1; k++) {
      const a = catches[k],
        b = catches[k + 1],
        period = seq[b].time - seq[a].time;
      if (period < 1 || period > 10) continue;
      let finish = a;
      for (let i = a + 1; i < b; i++)
        if (signal[i] !== null && signal[i]! > (signal[finish] ?? -Infinity))
          finish = i;
      const amplitude = signal[finish]! - (signal[a]! + signal[b]!) / 2,
        drive = seq[finish].time - seq[a].time,
        recovery = seq[b].time - seq[finish].time;
      if (amplitude < 0.04 || drive < 0.2 || recovery < 0.2) continue;
      const points = seq
          .slice(a, b + 1)
          .flatMap((f) => [
            f.points[11],
            f.points[12],
            f.points[15],
            f.points[16],
          ]),
        ev = evidence(points),
        confidence = Math.min(
          ev.confidence,
          seq.slice(a, b + 1).reduce((s, f) => s + f.identityConfidence, 0) /
            (b - a + 1),
        );
      const reliable = ev.fraction >= 0.65 && confidence >= 0.5;
      result.push({
        seat,
        index: result.filter((s) => s.seat === seat).length + 1,
        catch: seq[a].time,
        finish: seq[finish].time,
        nextCatch: seq[b].time,
        rate: reliable
          ? { value: 60 / period, unit: "spm", confidence }
          : withheld("spm", "Insufficient observed hip/ankle evidence"),
        ratio: reliable
          ? { value: drive / recovery, unit: ":1", confidence }
          : withheld(":1", "Insufficient observed hip/ankle evidence"),
        observedFraction: ev.fraction,
        confidence,
      });
    }
  }
  return result;
}
export function crewTiming(strokes: Stroke[], strokeSeat: number) {
  const ref = strokes.filter(
    (s) => s.seat === strokeSeat && s.rate.value !== null,
  );
  return strokes
    .filter((s) => s.seat !== strokeSeat && s.rate.value !== null)
    .flatMap((s) => {
      const r = ref.reduce<Stroke | undefined>(
        (best, v) =>
          !best || Math.abs(v.catch - s.catch) < Math.abs(best.catch - s.catch)
            ? v
            : best,
        undefined,
      );
      if (!r || Math.abs(r.catch - s.catch) > (r.nextCatch - r.catch) * 0.25)
        return [];
      return [
        {
          seat: s.seat,
          index: s.index,
          catchMs: (s.catch - r.catch) * 1000,
          finishMs: (s.finish - r.finish) * 1000,
          phase: ((s.catch - r.catch) / (r.nextCatch - r.catch)) * 360,
          confidence: Math.min(s.confidence, r.confidence),
        },
      ];
    });
}
export const mean = (values: (number | null)[]) => {
  const valid = values.filter(
    (v): v is number => v !== null && Number.isFinite(v),
  );
  return valid.length ? valid.reduce((a, b) => a + b) / valid.length : null;
};
