import type { Box, PoseFrame, Boat, Point } from "./types";
import { missing } from "./types";
interface Track {
  seat: number;
  box: Box;
  missed: number;
}
export interface TrackedDetection {
  box: Box;
  points: Point[];
  boxSource?: "detected" | "predicted";
  detectorAge?: number;
}
// Exact global assignment for <=8 seats using bitmask DP over candidate detections.
export class SeatTracker {
  private tracks: Track[] = [];
  constructor(private boat: Boat) {}

  /** Check a proposed assignment without advancing any track. */
  preview(detections: TrackedDetection[], time: number): PoseFrame[] {
    const copy = new SeatTracker(this.boat);
    copy.tracks = this.tracks.map((t) => ({ ...t, box: { ...t.box } }));
    return copy.update(detections, time);
  }

  update(detections: TrackedDetection[], time: number): PoseFrame[] {
    if (!this.tracks.length) {
      const sorted = [...detections].sort(
        (a, b) => (a.box.x - b.box.x) * (this.boat.bowLeft ? 1 : -1),
      );
      if (sorted.length !== this.boat.seats)
        throw Error(
          `Found ${sorted.length} people for ${this.boat.seats} seats. Set the correct seat count or use a tighter video crop to exclude the coxswain/bystanders. Automatic seat identity cannot safely initialize.`,
        );
      this.tracks = sorted.map((d, i) => ({
        seat: i + 1,
        box: d.box,
        missed: 0,
      }));
    }
    const candidates = detections.slice(0, 12);
    const memo = new Map<string, { cost: number; ids: number[] }>();
    const cost = (t: Track, d: (typeof candidates)[number]) => {
      const dx = d.box.x + d.box.width / 2 - t.box.x - t.box.width / 2,
        dy = d.box.y + d.box.height / 2 - t.box.y - t.box.height / 2;
      return Math.hypot(dx, dy) / Math.max(t.box.width, t.box.height, 1);
    };
    const solve = (
      i: number,
      mask: number,
    ): { cost: number; ids: number[] } => {
      if (i === this.tracks.length) return { cost: 0, ids: [] };
      const key = `${i}:${mask}`;
      if (memo.has(key)) return memo.get(key)!;
      let r = solve(i + 1, mask),
        best = { cost: r.cost + 0.75, ids: [-1, ...r.ids] };
      candidates.forEach((d, j) => {
        if (mask & (1 << j)) return;
        const c = cost(this.tracks[i], d);
        if (c > 0.7) return;
        const next = solve(i + 1, mask | (1 << j));
        if (c + next.cost < best.cost)
          best = { cost: c + next.cost, ids: [j, ...next.ids] };
      });
      memo.set(key, best);
      return best;
    };
    const assignment = solve(0, 0).ids;
    return this.tracks.map((t, i) => {
      const j = assignment[i],
        d = candidates[j];
      let ambiguous = false;
      if (d) {
        const cs = this.tracks
          .map((track) => cost(track, d))
          .sort((a, b) => a - b);
        ambiguous = cs.length > 1 && cs[1] - cs[0] < 0.1;
      }
      if (!d || ambiguous || t.missed > 30) {
        t.missed++;
        return {
          time,
          seat: t.seat,
          box: t.box,
          points: Array.from({ length: 133 }, missing),
          identityConfidence: 0,
        };
      }
      const c = cost(t, d);
      t.box = d.box;
      t.missed = 0;
      const age = d.boxSource === "predicted" ? (d.detectorAge ?? 1) : 0;
      return {
        time,
        seat: t.seat,
        box: d.box,
        points: d.points,
        boxSource: d.boxSource ?? "detected",
        detectorAge: age,
        // Reusing a box has no displacement cost, but is not a fresh identity
        // observation. Never let zero motion manufacture certainty.
        identityConfidence: Math.min(
          Math.max(0, 1 - c),
          age ? Math.max(0, 0.9 - age * 0.15) : 1,
        ),
      };
    });
  }
}
