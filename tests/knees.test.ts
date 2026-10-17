// Knees at the finish.
//
// The catch is the easy end of the stroke: the knee is the highest point of the
// leg, clear of the hull and clear of the loom, and pose models place it well.
// Near the finish it is not — with the legs flat the knee drops to the gunwale,
// and the loom crosses the legs on the way out. A whole-body model answers
// anyway: pinned to the hull edge, or dragged up onto the shaft, at a
// confidence that clears the observed threshold. Tracked as observations those
// answers lifted the trough every stroke event is timed from, and the finish
// proxy wandered over a quarter of the cycle (core/knees.ts).
//
// These tests pin down that such an answer can no longer reach tracking,
// evidence or stroke segmentation — simulated both ways, as a guess and as a
// dropout, by fixtures/rowingSim.ts — that the catch end of the stroke the pass
// must not touch is left byte for byte as the model reported it, and that the
// finish is timed from the bridged trough instead of from a guess.
import { describe, expect, it } from "vitest";
import {
  simulateRower,
  type SimOptions,
  type SimResult,
} from "./fixtures/rowingSim";
import {
  OCCLUDED_KNEE_FLAG,
  evidence,
  expectedOcclusion,
  hiddenJoints,
  recoverOcclusion,
  seen,
  suppressHiddenJoints,
} from "../src/core/occlusion";
import {
  KNEE_JOINTS,
  describeKneeOcclusion,
  discardOccludedKnees,
  kneeCheck,
  kneesAtRisk,
  type KneeReport,
} from "../src/core/knees";
import { CYCLE_JOINTS, segmentStrokes } from "../src/core/strokes";
import { createDemo } from "../src/core/demo";
import {
  defaultBoat,
  type Boat,
  type PoseFrame,
  type Stroke,
} from "../src/core/types";

/** The worker's knee pass, between tracking and the post-run stages. */
function analyse(
  sim: SimResult,
  boat: Boat = defaultBoat,
  knees = true,
): { frames: PoseFrame[]; report: KneeReport; strokes: Stroke[] } {
  const suppressed = sim.frames.map((f) => ({
    ...f,
    points: suppressHiddenJoints(f.points, hiddenJoints(boat)),
  }));
  const report: KneeReport = knees
    ? discardOccludedKnees(suppressed, boat)
    : { frames: suppressed, discarded: 0, windows: 0, judged: 0, seats: 1 };
  return {
    frames: report.frames,
    report,
    strokes: segmentStrokes(recoverOcclusion(report.frames)),
  };
}

/** True catches far enough from the clip edges for a cycle to be timed. */
const timeableCycles = (catches: number[], seconds: number) =>
  catches
    .slice(0, -1)
    .filter((c, i) => c > 0.75 && catches[i + 1] < seconds - 0.75);

const near = (strokes: Stroke[], t: number, tolerance: number) =>
  strokes.some((s) => Math.abs(s.catch - t) <= tolerance);

/** Where in its cycle a stroke's finish proxy landed. */
const phase = (s: Stroke) => (s.finish - s.catch) / (s.nextCatch - s.catch);

/** The frames the fixture says an occluder covered the near knee on. */
const occluded = (sim: SimResult) =>
  new Set(
    sim.truth
      .map((t, i) => (t.hullHidKnee || t.oarHidKnee ? i : -1))
      .filter((i) => i >= 0),
  );

/** Gunwale at hip height: the near knee is covered for 46% of every cycle. */
const HULL: SimOptions = { kneeGunwale: 0, kneeHidden: "snap" };
/** The loom crossing the legs near the finish, for 14% of every cycle. */
const LOOM: SimOptions = { oarKnee: 0.14, oarHidden: "drag" };
/**
 * The same two occluders with a model that does not answer the same covered
 * knee the same way twice: every frame is a fresh coin flip between a guess and
 * a dropout. This is the ordinary case in real footage, and neither answer on
 * its own carries enough evidence — a dropout has nothing to judge, and a guess
 * between dropouts is one frame.
 */
const FLICKERING_HULL: SimOptions = { kneeGunwale: 0, kneeHidden: "mixed" };
const FLICKERING_LOOM: SimOptions = { oarKnee: 0.14, oarHidden: "mixed" };

describe("a knee the gunwale or the loom covers", () => {
  const clips: [string, SimOptions][] = [
    ["gunwale", HULL],
    ["loom", LOOM],
    ["gunwale and the loom", { ...HULL, ...LOOM }],
  ];

  it.each(clips)(
    "marks the guess at the %s missing with a reason instead of tracking it",
    (_label, options) => {
      const sim = simulateRower({ seconds: 12, rate: 24, seed: 1, ...options });
      const { frames, report } = analyse(sim);
      const hidden = [...occluded(sim)];
      expect(hidden.length).toBeGreaterThan(30);
      expect(report.judged).toBe(1);
      expect(report.windows).toBeGreaterThan(0);
      expect(report.discarded).toBeGreaterThan(0);
      let flagged = 0;
      for (const f of frames)
        for (const j of KNEE_JOINTS) {
          const p = f.points[j];
          if (!p.flags?.includes(OCCLUDED_KNEE_FLAG)) continue;
          flagged++;
          // Not an observation any more, and the reason travels with it.
          expect(p.origin).toBe("missing");
          expect(p.confidence).toBe(0);
        }
      expect(flagged).toBe(report.discarded);
      // Nothing the fixture says was covered is still read as an observation.
      for (const i of hidden)
        for (const j of KNEE_JOINTS)
          expect(seen(frames[i].points[j])).toBe(false);
    },
    20_000,
  );

  it("reads a crossing the model answers differently every frame as one window", () => {
    for (const options of [FLICKERING_HULL, FLICKERING_LOOM]) {
      const sim = simulateRower({ seconds: 20, rate: 24, seed: 1, ...options });
      const { frames, report, strokes } = analyse(sim);
      expect(strokes.length).toBeGreaterThan(5);
      // One window per crossing, not one per run of like answers: the frames a
      // dropout leaves without an elevation still sit where their neighbours
      // sit, so they cannot split the crossing they belong to.
      expect(report.windows).toBeGreaterThan(strokes.length - 3);
      expect(report.windows).toBeLessThanOrEqual(strokes.length + 2);
      expect(report.discarded).toBeGreaterThan(0);
      // A guess the trim leaves standing at the edge of a loom window, and the
      // hole the existing short-gap recovery interpolates across a dropout
      // beside it, are what is left of a crossing the model answered
      // inconsistently: labelled, a few pixels off, and a twentieth of it —
      // against all of it before the pass.
      const hidden = occluded(sim);
      let left = 0;
      for (const i of hidden)
        for (const j of KNEE_JOINTS) if (seen(frames[i].points[j])) left++;
      expect(left).toBeLessThan(0.05 * hidden.size * KNEE_JOINTS.length);
    }
  }, 20_000);

  it("would otherwise track a guess far from where the knee is", () => {
    // The failure the pass exists for, on the same clips without it.
    for (const options of [HULL, LOOM]) {
      const sim = simulateRower({ seconds: 12, rate: 24, seed: 1, ...options });
      const before = analyse(sim, defaultBoat, false).frames;
      const hidden = [...occluded(sim)];
      let tracked = 0,
        error = 0;
      for (const i of hidden) {
        const p = before[i].points[13];
        if (!seen(p)) continue;
        tracked++;
        error += Math.hypot(
          p.x - sim.truth[i].knee.x,
          p.y - sim.truth[i].knee.y,
        );
      }
      expect(tracked / hidden.length).toBeGreaterThan(0.9);
      expect(error / tracked).toBeGreaterThan(8);
    }
  }, 20_000);

  it.each(clips)(
    "leaves the catch end of the stroke as the %s left it",
    (_label, options) => {
      const sim = simulateRower({ seconds: 12, rate: 24, seed: 1, ...options });
      const raw = analyse(sim, defaultBoat, false).frames;
      const { frames } = analyse(sim);
      let kept = 0,
        lost = 0;
      for (let i = 0; i < frames.length; i++) {
        // The pass only ever replaces a knee, so anything still standing is
        // byte for byte what the model reported.
        for (const j of KNEE_JOINTS)
          if (seen(frames[i].points[j]))
            expect(frames[i].points[j]).toEqual(raw[i].points[j]);
        // The catch: legs compressed, the knee the highest point of the leg
        // and clear of both occluders.
        if (sim.truth[i].extension > 0.2 || !seen(raw[i].points[13])) continue;
        kept++;
        if (!seen(frames[i].points[13])) lost++;
      }
      expect(kept).toBeGreaterThan(40);
      expect(lost).toBe(0);
    },
    20_000,
  );

  it("discards hardly any knee an occluder was not in front of", () => {
    let clear = 0,
      discarded = 0;
    for (const options of [HULL, LOOM, { ...HULL, ...LOOM }])
      for (const rate of [20, 28])
        for (const seed of [1, 2]) {
          const sim = simulateRower({ seconds: 20, rate, seed, ...options });
          const raw = analyse(sim, defaultBoat, false).frames;
          const { frames } = analyse(sim);
          const hidden = occluded(sim);
          for (let i = 0; i < frames.length; i++) {
            if (hidden.has(i)) continue;
            for (const j of KNEE_JOINTS) {
              if (!seen(raw[i].points[j])) continue;
              clear++;
              if (expectedOcclusion(frames[i].points[j])) discarded++;
            }
          }
        }
    expect(clear).toBeGreaterThan(2000);
    // A frame or two at the edge of a window costs nothing: the timing reads
    // the catch peaks and the bridged trough, not every sample of the slide.
    expect(discarded / clear).toBeLessThan(0.03);
  }, 60_000);

  it("keeps a manually labeled knee", () => {
    const sim = simulateRower({ seconds: 12, rate: 24, seed: 1, ...HULL });
    const at = analyse(sim).frames.findIndex((f) =>
      f.points[13].flags?.includes(OCCLUDED_KNEE_FLAG),
    );
    expect(at).toBeGreaterThan(0);
    const frames = sim.frames.map((f) => ({ ...f, points: f.points.slice() }));
    frames[at].points[13] = { ...frames[at].points[13], origin: "manual" };
    const kept = discardOccludedKnees(frames, defaultBoat).frames[at]
      .points[13];
    expect(kept.origin).toBe("manual");
    expect(kept.flags?.includes(OCCLUDED_KNEE_FLAG)).toBeFalsy();
  }, 20_000);

  it("leaves footage whose feet are in view alone", () => {
    // An ergometer or overhead setup has no gunwale and no loom across the
    // legs, so its knees are exactly what the model reported.
    expect(kneesAtRisk(defaultBoat)).toBe(true);
    expect(kneesAtRisk({ ...defaultBoat, feetHidden: false })).toBe(false);
    const sim = simulateRower({
      seconds: 12,
      rate: 24,
      seed: 1,
      feet: "visible",
      ...HULL,
      ...LOOM,
    });
    const report = discardOccludedKnees(sim.frames, {
      ...defaultBoat,
      feetHidden: false,
    });
    expect(report.discarded).toBe(0);
    expect(report.windows).toBe(0);
    expect(report.frames).toEqual(sim.frames);
  }, 20_000);

  it("leaves a seat it cannot calibrate alone", () => {
    const sim = simulateRower({ seconds: 12, rate: 24, seed: 1, ...HULL });
    // Too short to have seen a stroke, so there is no clear-view knee to
    // compare an uncertain one against.
    const short = sim.frames.slice(0, 40);
    expect(kneeCheck(short)).toBeNull();
    expect(discardOccludedKnees(short, defaultBoat).discarded).toBe(0);
    // No elevation range at all: a stationary rower, or a joint that never moved.
    const still = Array.from({ length: 100 }, (_, i) => ({
      ...sim.frames[0],
      time: i / 30,
    }));
    expect(kneeCheck(still)).toBeNull();
    expect(discardOccludedKnees(still, defaultBoat).discarded).toBe(0);
  }, 20_000);

  const dropped: [string, SimOptions][] = [
    ["gunwale", { ...HULL, kneeHidden: "dropout" }],
    ["loom", { ...LOOM, oarHidden: "dropout" }],
  ];

  it.each(dropped)(
    "changes nothing when the model already dropped the knee at the %s",
    (_label, options) => {
      // A dropped knee is already honest: below threshold, not tracked, and
      // the finish already comes from the bridged trough.
      const sim = simulateRower({ seconds: 20, rate: 24, seed: 1, ...options });
      const before = analyse(sim, defaultBoat, false);
      const after = analyse(sim);
      expect(after.report.discarded).toBe(0);
      expect(after.report.windows).toBe(0);
      expect(after.frames).toEqual(before.frames);
      expect(after.strokes.map((s) => s.catch)).toEqual(
        before.strokes.map((s) => s.catch),
      );
      expect(after.strokes.length).toBeGreaterThan(5);
    },
    20_000,
  );

  it("is idempotent", () => {
    const sim = simulateRower({ seconds: 12, rate: 24, seed: 1, ...HULL });
    const once = discardOccludedKnees(sim.frames, defaultBoat);
    expect(once.discarded).toBeGreaterThan(0);
    const twice = discardOccludedKnees(once.frames, defaultBoat);
    expect(twice.discarded).toBe(0);
    expect(twice.windows).toBe(0);
    expect(twice.frames).toEqual(once.frames);
  }, 20_000);

  it("says what it did in the analysis notes, and nothing when it did nothing", () => {
    const sim = simulateRower({ seconds: 12, rate: 24, seed: 1, ...HULL });
    const report = analyse(sim).report;
    const note = describeKneeOcclusion(report);
    expect(note).toContain("gunwale");
    expect(note).toContain(String(report.discarded));
    expect(note).toContain(String(report.windows));
    const dropped = simulateRower({
      seconds: 12,
      rate: 24,
      seed: 1,
      ...HULL,
      kneeHidden: "dropout",
    });
    expect(describeKneeOcclusion(analyse(dropped).report)).toBeNull();
  }, 20_000);
});

describe("stroke timing with an occluded knee", () => {
  /**
   * Floors on a cycle's observed fraction. A knee the pass discarded was never
   * observable, so it is not evidence the model failed to find
   * (core/occlusion.ts), and what is left should be nearly all observed. A knee
   * the model dropped honestly is a different thing: a real absence, which the
   * pass has no business explaining away, so the floor falls with the share of
   * the crossing the clip drops — nothing for a guess, about half of it for a
   * clip that flickers between the two answers, all of it for a dropout (whose
   * evidence is the two hips and whatever the far side shows).
   */
  const GUESS = 0.9,
    FLICKER = 0.8,
    DROP = 0.65;

  const runs: [string, SimOptions, number, number][] = [
    ["clear knees", {}, 24, GUESS],
    ["gunwale guess", HULL, 20, GUESS],
    ["gunwale guess", HULL, 28, GUESS],
    ["gunwale dropout", { ...HULL, kneeHidden: "dropout" }, 24, DROP],
    ["flickering gunwale", FLICKERING_HULL, 20, FLICKER],
    ["flickering gunwale", FLICKERING_HULL, 28, FLICKER],
    ["loom drag", LOOM, 20, GUESS],
    ["loom drag", LOOM, 28, GUESS],
    ["loom dropout", { ...LOOM, oarHidden: "dropout" }, 24, DROP],
    ["flickering loom", FLICKERING_LOOM, 20, FLICKER],
    ["gunwale and loom", { ...HULL, ...LOOM }, 24, GUESS],
  ];

  it.each(
    runs.map(([label, options, rate, floor]) => [
      `${label} at ${rate} spm`,
      options,
      rate,
      floor,
    ]),
  )(
    "finds every cycle once with accurate timing (%s)",
    (_label, options, rate, floor) => {
      const seconds = 20;
      let matched = 0;
      for (const seed of [1, 2]) {
        const sim = simulateRower({ seconds, rate, seed, ...options });
        const { strokes, report } = analyse(sim);
        const expected = timeableCycles(sim.catches, seconds);
        // Every timeable cycle is found, and nothing else is.
        for (const c of expected) {
          expect(near(strokes, c, 0.15)).toBe(true);
          matched++;
        }
        for (const s of strokes)
          expect(sim.catches.some((c) => Math.abs(c - s.catch) <= 0.15)).toBe(
            true,
          );
        expect(strokes.length).toBeLessThanOrEqual(expected.length + 2);
        for (const s of strokes) {
          expect(s.rate.value).not.toBeNull();
          expect(Math.abs(s.rate.value! - rate)).toBeLessThan(1.5);
          // The simulated legs are flat from 28% to 55% of the cycle, so the
          // finish proxy belongs in the middle of that band — whether it was
          // read from a visible knee or from a trough the hull bridged.
          expect(Math.abs(phase(s) - 0.415)).toBeLessThan(0.09);
          expect(s.confidence).toBeGreaterThan(0.5);
          expect(s.observedFraction).toBeGreaterThan(
            report.discarded ? floor : DROP,
          );
        }
      }
      // Not a vacuous pass: two seeds of 20 s is 10–14 cycles at these rates.
      expect(matched).toBeGreaterThan(8);
    },
    60_000,
  );

  it("does not count an expected occlusion against a cycle's evidence", () => {
    const sim = simulateRower({ seconds: 20, rate: 24, seed: 1, ...HULL });
    const { frames, strokes } = analyse(sim);
    const cycle = frames.filter(
      (f) => f.time >= strokes[0].catch && f.time <= strokes[0].nextCatch,
    );
    const joints = cycle.flatMap((f) => CYCLE_JOINTS.map((j) => f.points[j]));
    // The gunwale covered the knee for most of this cycle. Counted naively
    // that reads as a cycle nobody observed; counted as what it is — a joint
    // the hull was in front of — the hips and the catch end still carry it.
    expect(evidence(joints).fraction).toBeLessThan(0.85);
    expect(
      evidence(joints.filter((p) => !expectedOcclusion(p))).fraction,
    ).toBeGreaterThan(0.95);
    expect(strokes[0].observedFraction).toBeGreaterThan(0.95);
    expect(strokes[0].rate.value).toBeCloseTo(24, 0);
  }, 20_000);

  it("flags exactly the absences the setup or the footage explains", () => {
    const point = (flags?: string[]) => ({
      x: 0,
      y: 0,
      confidence: 0,
      origin: "missing" as const,
      flags,
    });
    expect(expectedOcclusion(point([OCCLUDED_KNEE_FLAG]))).toBe(true);
    expect(expectedOcclusion(undefined)).toBe(false);
    // A joint the model simply did not find is not an expected occlusion: it
    // stays in the evidence count and stays eligible for a short gap fill.
    expect(expectedOcclusion(point())).toBe(false);
    expect(expectedOcclusion(point(["Low-confidence jump; not bridged"]))).toBe(
      false,
    );
  });
});

describe("the demonstration crew", () => {
  it("discards nothing and keeps its generated cycles", () => {
    // A procedural crew with a rigid thigh and shank (core/demo.ts): the pass
    // judges every seat and finds no occlusion to remove. A false positive
    // here would be a false positive on any clear clip.
    const session = createDemo(defaultBoat);
    expect(session.boat.feetHidden).toBe(true);
    const report = discardOccludedKnees(session.frames, defaultBoat);
    expect(report.judged).toBe(defaultBoat.seats);
    expect(report.discarded).toBe(0);
    expect(report.windows).toBe(0);
    expect(report.frames).toEqual(session.frames);
    const rows = session.strokes.filter((s) => s.seat === defaultBoat.seats);
    expect(rows.length).toBeGreaterThan(2);
    for (const s of rows) expect(s.rate.value).toBeCloseTo(25, 0);
  }, 20_000);
});

// Accuracy table behind docs/VALIDATION.md, before and after the knee pass.
// Opt-in because it runs 72 simulated 30 s clips:
//   ROW_SIM_REPORT=1 npx vitest run tests/knees.test.ts
describe.runIf(process.env.ROW_SIM_REPORT)("knee occlusion report", () => {
  it(
    "prints knee error and cycle accuracy by occlusion scenario and rate",
    { timeout: 900_000 },
    () => {
      const seconds = 30;
      const scenarios: [string, SimOptions][] = [
        ["clear", {}],
        ["hull snap", HULL],
        ["hull flicker", FLICKERING_HULL],
        ["hull dropout", { ...HULL, kneeHidden: "dropout" }],
        ["oar drag", LOOM],
        ["oar flicker", FLICKERING_LOOM],
        ["oar dropout", { ...LOOM, oarHidden: "dropout" }],
        ["snap + drag", { ...HULL, ...LOOM }],
      ];
      const lines = [
        "pass   | scenario     | spm | cycles/found/matched | max rate err | finish phase | knee tracked while hidden (mean px off) | clear knees discarded | windows",
      ];
      for (const fixed of [false, true])
        for (const [label, scenario] of scenarios)
          for (const rate of [20, 28]) {
            let cycles = 0,
              found = 0,
              matched = 0,
              hidden = 0,
              tracked = 0,
              errSum = 0,
              clear = 0,
              lost = 0,
              windows = 0;
            const errs: number[] = [],
              phases: number[] = [];
            for (const seed of [1, 2, 3]) {
              const sim = simulateRower({ seconds, rate, seed, ...scenario });
              const { frames, report, strokes } = analyse(
                sim,
                defaultBoat,
                fixed,
              );
              windows += report.windows;
              const expected = timeableCycles(sim.catches, seconds);
              cycles += expected.length;
              found += strokes.length;
              for (const c of expected) if (near(strokes, c, 0.15)) matched++;
              for (const s of strokes) {
                if (s.rate.value !== null)
                  errs.push(Math.abs(s.rate.value - rate));
                phases.push(phase(s));
              }
              const blind = occluded(sim);
              for (let i = 0; i < frames.length; i++) {
                // The fixture reports truth for the near knee only.
                if (blind.has(i)) {
                  hidden++;
                  const p = frames[i].points[13];
                  if (!seen(p)) continue;
                  tracked++;
                  errSum += Math.hypot(
                    p.x - sim.truth[i].knee.x,
                    p.y - sim.truth[i].knee.y,
                  );
                } else
                  for (const j of KNEE_JOINTS) {
                    if (!seen(sim.frames[i].points[j])) continue;
                    clear++;
                    if (expectedOcclusion(frames[i].points[j])) lost++;
                  }
              }
            }
            const mm = (v: number[]) =>
              v.length
                ? `${Math.min(...v).toFixed(2)}–${Math.max(...v).toFixed(2)}`
                : "-";
            lines.push(
              [
                (fixed ? "fixed" : "before").padEnd(6),
                label.padEnd(12),
                String(rate).padStart(3),
                `${cycles}/${found}/${matched}`.padStart(20),
                Math.max(...errs, 0)
                  .toFixed(2)
                  .padStart(12),
                mm(phases).padStart(12),
                `${Math.round((100 * tracked) / Math.max(1, hidden))}% (${tracked ? (errSum / tracked).toFixed(1) : "-"})`.padStart(
                  37,
                ),
                `${((100 * lost) / Math.max(1, clear)).toFixed(2)}%`.padStart(
                  21,
                ),
                String(windows).padStart(7),
              ].join(" | "),
            );
          }
      console.log(lines.join("\n"));
    },
  );
});
