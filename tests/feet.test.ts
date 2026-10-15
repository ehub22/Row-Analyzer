// Permanently hidden feet.
//
// On-water footage almost never shows the feet: they are strapped to the
// stretcher below the gunwale. A whole-body model still answers for them, so
// these tests pin down that its guesses (simulated in fixtures/rowingSim.ts)
// can no longer reach tracking, evidence, the adaptive TTA gate or stroke
// segmentation — and that cycles survive the knees, too, dropping behind the
// hull while the legs are flat.
import { describe, it, expect } from "vitest";
import {
  FOOT_JOINTS,
  HIDDEN_FEET_FLAG,
  hiddenJoints,
  recoverOcclusion,
  suppressHiddenJoints,
} from "../src/core/occlusion";
import {
  CYCLE_JOINTS,
  prominentPeaks,
  segmentStrokes,
  thighElevation,
} from "../src/core/strokes";
import { assessPose, gateJoints, BODY_JOINTS } from "../src/core/tta";
import { createDemo } from "../src/core/demo";
import { validateSession } from "../src/core/export";
import {
  defaultBoat,
  missing,
  type Point,
  type PoseFrame,
  type Stroke,
} from "../src/core/types";
import { rng, simulateRower, type SimOptions } from "./fixtures/rowingSim";

/** The worker's per-frame treatment, followed by the post-run stages. */
function analyse(frames: PoseFrame[], hidden = FOOT_JOINTS): Stroke[] {
  return segmentStrokes(
    recoverOcclusion(
      frames.map((f) => ({
        ...f,
        points: suppressHiddenJoints(f.points, hidden),
      })),
    ),
  );
}

/** True catches far enough from the clip edges for a cycle to be timed. */
function timeableCycles(catches: number[], seconds: number) {
  return catches
    .slice(0, -1)
    .filter((c, i) => c > 0.75 && catches[i + 1] < seconds - 0.75);
}

const near = (strokes: Stroke[], t: number, tolerance: number) =>
  strokes.some((s) => Math.abs(s.catch - t) <= tolerance);

describe("expected occlusion of the feet", () => {
  it("declares ankles and feet hidden unless the setup says they are visible", () => {
    expect([...FOOT_JOINTS]).toEqual([15, 16, 17, 18, 19, 20, 21, 22]);
    expect(hiddenJoints(defaultBoat)).toEqual(FOOT_JOINTS);
    expect(defaultBoat.feetHidden).toBe(true);
    // Sessions from older builds have no field: new analyses default to the
    // on-water case the app is built for.
    expect(hiddenJoints({})).toEqual(FOOT_JOINTS);
    expect(hiddenJoints({ feetHidden: false })).toEqual([]);
  });

  it("replaces model guesses with flagged missing joints and keeps the rest", () => {
    const { frames } = simulateRower({ seconds: 1 });
    const before = frames[0].points;
    const manual: Point = { x: 5, y: 6, confidence: 1, origin: "manual" };
    const input = before.map((p, j) => (j === 20 ? manual : p));
    const after = suppressHiddenJoints(input, FOOT_JOINTS);
    for (const j of FOOT_JOINTS.filter((j) => j !== 20)) {
      expect(after[j]).toMatchObject({ origin: "missing", confidence: 0 });
      expect(after[j].flags).toEqual([HIDDEN_FEET_FLAG]);
    }
    // A person who can see a foot in a frame outranks the expectation.
    expect(after[20]).toEqual(manual);
    for (let j = 0; j < 15; j++) expect(after[j]).toBe(input[j]);
    expect(input[15]).toBe(before[15]); // input array untouched
    expect(input[15].flags).toBeUndefined();
    expect(suppressHiddenJoints(input, [])).toBe(input);
  });

  it("keeps hallucinated feet out of gap filling and the overlay", () => {
    const { frames } = simulateRower({ seconds: 4, feet: "hallucinate" });
    const raw = recoverOcclusion(frames);
    // Without suppression the guesses are tracked as observations…
    expect(
      raw.filter((f) => f.points[15].origin === "observed").length,
    ).toBeGreaterThan(frames.length * 0.3);
    // …with it, no foot joint is ever observed or inferred.
    const clean = recoverOcclusion(
      frames.map((f) => ({
        ...f,
        points: suppressHiddenJoints(f.points, FOOT_JOINTS),
      })),
    );
    for (const f of clean)
      for (const j of FOOT_JOINTS) expect(f.points[j].origin).toBe("missing");
  });
});

describe("adaptive TTA gate with hidden feet", () => {
  it("leaves hidden joints out of the gate", () => {
    expect(gateJoints()).toEqual(BODY_JOINTS);
    expect([...gateJoints(FOOT_JOINTS)]).toEqual(
      Array.from({ length: 15 }, (_, i) => i),
    );
    const points = Array.from({ length: 133 }, (_, j) => ({
      x: j,
      y: j,
      confidence: j === 15 || j === 16 ? 0.1 : 0.9,
      origin:
        j === 15 || j === 16 ? ("missing" as const) : ("observed" as const),
    }));
    expect(assessPose(points).accepted).toBe(false);
    const gated = assessPose(points, 0.5, 0.3, gateJoints(FOOT_JOINTS));
    expect(gated.accepted).toBe(true);
    expect(assessPose(points).reason).toContain("of 17 body joints");
  });

  it("stops paying four passes on every crop because of the ankles", () => {
    const { frames } = simulateRower({ seconds: 10, feet: "hallucinate" });
    const rate = (joints: readonly number[]) =>
      frames.filter((f) => assessPose(f.points, 0.5, 0.3, joints).accepted)
        .length / frames.length;
    // Guessed ankles fall below 0.3 in most crops: adaptive ≈ always.
    expect(rate(BODY_JOINTS)).toBeLessThan(0.5);
    expect(rate(gateJoints(FOOT_JOINTS))).toBeGreaterThan(0.85);
  });
});

describe("thigh-elevation cycle signal", () => {
  const frame = (hip: [number, number], knee: [number, number], c = 0.9) => {
    const points = Array.from({ length: 133 }, missing);
    for (const [j, [x, y]] of [
      [11, hip],
      [12, hip],
      [13, knee],
      [14, knee],
    ] as const)
      points[j] = { x, y, confidence: c, origin: "observed" };
    return {
      time: 0,
      seat: 1,
      points,
      box: { x: 0, y: 0, width: 100, height: 100, confidence: 1 },
      identityConfidence: 1,
    } satisfies PoseFrame;
  };

  it("measures the knee rise above the hip, whichever way the crew faces", () => {
    expect(thighElevation(frame([100, 100], [140, 60]))).toBeCloseTo(45);
    expect(thighElevation(frame([100, 100], [60, 60]))).toBeCloseTo(45);
    expect(thighElevation(frame([100, 100], [150, 110]))).toBeLessThan(0);
    expect(thighElevation(frame([100, 100], [140, 60], 0.2))).toBeNull();
  });

  it("never reads the ankles or feet", () => {
    expect(CYCLE_JOINTS).toEqual([11, 12, 13, 14]);
    const { frames } = simulateRower({ seconds: 20, feet: "visible" });
    const R = rng(99);
    const garbage = frames.map((f) => ({
      ...f,
      points: f.points.map((p, j) =>
        FOOT_JOINTS.includes(j)
          ? {
              x: R.uniform(0, 1280),
              y: R.uniform(0, 720),
              confidence: R.uniform(0.3, 1),
              origin: "observed" as const,
            }
          : p,
      ),
    }));
    expect(segmentStrokes(garbage)).toEqual(segmentStrokes(frames));
  });
});

describe("prominent peaks", () => {
  const t = (n: number) => Array.from({ length: n }, (_, i) => i / 30);

  it("ignores noise on a plateau and keeps real peaks", () => {
    const R = rng(3);
    // Two catches 3 s apart with a long, noisy legs-flat plateau between.
    const v = t(180).map((s) => {
      const d = Math.min(Math.abs(s - 1), Math.abs(s - 4));
      return (d < 0.6 ? 40 * (1 - d / 0.6) : 0) + R.gauss() * 1.2;
    });
    const peaks = prominentPeaks(v, t(180), 8, 1).map((p) => t(180)[p.index]);
    expect(peaks).toHaveLength(2);
    expect(peaks[0]).toBeCloseTo(1, 0);
    expect(peaks[1]).toBeCloseTo(4, 0);
  });

  it("judges a peak at the start of the data by the side it can see", () => {
    // Data starts just before a peak: the left side never rises above it.
    const v = [38, 39, 40, 30, 10, 0, 0, 10, 30, 41, 30, 0];
    const peaks = prominentPeaks(
      v,
      v.map((_, i) => i),
      8,
      1,
    );
    expect(peaks.map((p) => p.index)).toEqual([2, 9]);
    expect(peaks[0].prominence).toBe(40);
  });

  it("keeps the higher of two peaks closer than the minimum distance", () => {
    const v = [0, 30, 0, 35, 0];
    const peaks = prominentPeaks(v, [0, 0.2, 0.4, 0.6, 0.8], 8, 1);
    expect(peaks.map((p) => p.index)).toEqual([3]);
  });
});

describe("stroke cycles with permanently hidden feet", () => {
  const scenarios: SimOptions[] = [
    { rate: 18, seed: 1 },
    { rate: 24, seed: 2, feetToRight: false },
    { rate: 32, seed: 3, feet: "dropout" },
    { rate: 40, seed: 4 },
  ];

  it.each(scenarios)(
    "finds every cycle once with accurate timing ($rate spm)",
    (options) => {
      const seconds = 30;
      const sim = simulateRower({ seconds, feet: "hallucinate", ...options });
      const strokes = analyse(sim.frames);
      const expected = timeableCycles(sim.catches, seconds);
      // Every timeable cycle is found, and nothing else is.
      for (const c of expected) expect(near(strokes, c, 0.15)).toBe(true);
      for (const s of strokes)
        expect(sim.catches.some((c) => Math.abs(c - s.catch) <= 0.15)).toBe(
          true,
        );
      expect(strokes.length).toBeLessThanOrEqual(expected.length + 2);
      for (const s of strokes) {
        expect(s.rate.value).not.toBeNull();
        expect(Math.abs(s.rate.value! - options.rate!)).toBeLessThan(1.5);
        // Finish proxy: inside the legs-flat part of the cycle (simulated
        // legs flat from 28% to 55% of the cycle).
        const phase = (s.finish - s.catch) / (s.nextCatch - s.catch);
        expect(phase).toBeGreaterThan(0.28);
        expect(phase).toBeLessThan(0.55);
      }
    },
    20_000,
  );

  it("does not double the rate at low rates (long legs-flat plateau)", () => {
    for (const seed of [5, 6, 7]) {
      const sim = simulateRower({ seconds: 40, rate: 18, seed });
      const rates = analyse(sim.frames).map((s) => s.rate.value!);
      expect(rates.length).toBeGreaterThanOrEqual(10);
      expect(Math.max(...rates)).toBeLessThan(19.5);
      expect(Math.min(...rates)).toBeGreaterThan(16.5);
    }
  }, 20_000);

  it("gives the same cycles whether the model guesses, drops or sees the feet", () => {
    const run = (feet: SimOptions["feet"], hidden: readonly number[]) =>
      analyse(simulateRower({ seconds: 20, seed: 8, feet }).frames, hidden).map(
        (s) => Math.round(s.catch * 30),
      );
    const reference = run("visible", []);
    expect(reference.length).toBeGreaterThan(5);
    // Guesses are discarded, and the signal never read them anyway.
    expect(run("hallucinate", FOOT_JOINTS)).toEqual(reference);
    expect(run("hallucinate", [])).toEqual(reference);
    expect(run("dropout", FOOT_JOINTS)).toEqual(reference);
  }, 20_000);

  it("does not claim a cycle across a tracking blind spot (hips lost too)", () => {
    const sim = simulateRower({ seconds: 30, rate: 30, seed: 9 });
    const frames = sim.frames.map((f) =>
      f.time > 10 && f.time < 12.2
        ? {
            ...f,
            points: f.points.map((p, j) =>
              CYCLE_JOINTS.includes(j) ? missing() : p,
            ),
          }
        : f,
    );
    const strokes = analyse(frames);
    expect(strokes.some((s) => s.catch < 10 && s.nextCatch > 12.2)).toBe(false);
    expect(strokes.some((s) => s.nextCatch <= 10)).toBe(true);
    expect(strokes.some((s) => s.catch >= 12.2)).toBe(true);
  }, 20_000);

  it.each([18, 32])(
    "keeps every cycle when the hull also hides the knees with the legs flat (%i spm)",
    (rate) => {
      // Saxboard at hip height, knees dropped every frame they are below it:
      // 46% of every cycle without a knee, far past the 0.5 s gap limit.
      const seconds = 30;
      const sim = simulateRower({
        seconds,
        rate,
        seed: 11,
        kneeGunwale: 0,
        kneeHidden: "dropout",
      });
      const strokes = analyse(sim.frames);
      for (const c of timeableCycles(sim.catches, seconds))
        expect(near(strokes, c, 0.15)).toBe(true);
      for (const s of strokes) {
        expect(Math.abs(s.rate.value! - rate)).toBeLessThan(1.5);
        // Finish from the middle of the blind spot: the same legs-flat
        // phase as when the knees are visible (0.42–0.49 of the cycle).
        const phase = (s.finish - s.catch) / (s.nextCatch - s.catch);
        expect(phase).toBeGreaterThan(0.4);
        expect(phase).toBeLessThan(0.52);
      }
    },
    20_000,
  );

  it("never bridges a blind spot that could hide a catch", () => {
    for (const rate of [24, 36]) {
      const sim = simulateRower({ seconds: 30, rate, seed: 12 });
      const period = 60 / rate,
        catchTime = sim.catches[6];
      // Only the knees vanish, from legs flat before a catch to legs flat
      // after it: low on both sides with the hips in view, like the hull —
      // but the catch inside means the cycle across it is two strokes long.
      const from = catchTime - 0.5 * period,
        to = catchTime + 0.45 * period;
      const frames = sim.frames.map((f) =>
        f.time > from && f.time < to
          ? {
              ...f,
              points: f.points.map((p, j) =>
                j === 13 || j === 14 ? missing() : p,
              ),
            }
          : f,
      );
      const strokes = analyse(frames);
      expect(strokes.some((s) => s.catch < from && s.nextCatch > to)).toBe(
        false,
      );
      for (const s of strokes)
        expect(Math.abs(s.rate.value! - rate)).toBeLessThan(1.5);
    }
  }, 20_000);

  it("withholds rates when hips and knees are only inferred", () => {
    const sim = simulateRower({ seconds: 20, seed: 10 });
    const frames = sim.frames.map((f) => ({
      ...f,
      points: f.points.map((p, j) =>
        CYCLE_JOINTS.includes(j) ? { ...p, origin: "inferred" as const } : p,
      ),
    }));
    const strokes = analyse(frames);
    expect(strokes.length).toBeGreaterThan(0);
    expect(strokes.every((s) => s.rate.value === null)).toBe(true);
    expect(strokes[0].rate.reason).toContain("hip/knee");
  });
});

describe("sessions record the feet setting", () => {
  it("demo discards foot joints like an on-water analysis", () => {
    const demo = createDemo(defaultBoat);
    expect(demo.boat.feetHidden).toBe(true);
    for (const j of FOOT_JOINTS)
      expect(demo.frames[0].points[j].flags).toEqual([HIDDEN_FEET_FLAG]);
    expect(demo.strokes.length).toBeGreaterThan(0);
    const visible = createDemo({ ...defaultBoat, feetHidden: false });
    expect(visible.boat.feetHidden).toBe(false);
    expect(visible.frames[0].points[15].origin).toBe("observed");
  });

  it("validates the optional field on import", { timeout: 20_000 }, () => {
    const demo = createDemo(defaultBoat);
    expect(validateSession(structuredClone(demo)).boat.feetHidden).toBe(true);
    const legacy = structuredClone(demo);
    delete legacy.boat.feetHidden;
    expect(validateSession(legacy).boat.feetHidden).toBeUndefined();
    const bad = structuredClone(demo) as unknown as {
      boat: { feetHidden: unknown };
    };
    bad.boat.feetHidden = "yes";
    expect(() => validateSession(bad)).toThrow();
  });
});

// Accuracy table behind docs/VALIDATION.md. Opt-in because it runs 75
// simulated 40 s clips:  ROW_SIM_REPORT=1 npx vitest run tests/feet.test.ts
describe.runIf(process.env.ROW_SIM_REPORT)("simulation report", () => {
  it(
    "prints accuracy by occlusion scenario and stroke rate",
    { timeout: 300_000 },
    () => {
      const seconds = 40;
      const lines = [
        "scenario    | spm | cycles/found/matched | mean rate | max rate err | catch err (mean, max) | finish phase | TTA gate rejects (old gate)",
      ];
      const scenarios: [string, SimOptions][] = [
        ["visible", { feet: "visible" }],
        ["hallucinate", { feet: "hallucinate" }],
        ["dropout", { feet: "dropout" }],
        // Feet guessed, and knees dropped whenever they sit below a saxboard
        // at hip height (legs flat): the hardest case for the thigh signal.
        ["+knees hid", { kneeGunwale: 0, kneeHidden: "dropout" }],
      ];
      for (const [label, scenario] of scenarios)
        for (const rate of [18, 20, 26, 32, 40]) {
          let cycles = 0,
            found = 0,
            matched = 0,
            gated = 0,
            oldGated = 0,
            n = 0;
          const rates: number[] = [],
            errors: number[] = [],
            phases: number[] = [];
          for (const seed of [1, 2, 3, 4, 5]) {
            const sim = simulateRower({ ...scenario, rate, seed, seconds });
            const hidden = scenario.feet === "visible" ? [] : FOOT_JOINTS;
            const strokes = analyse(sim.frames, hidden);
            cycles += sim.catches.length - 1;
            found += strokes.length;
            for (const s of strokes) {
              const nearest = sim.catches.reduce((a, c) =>
                Math.abs(c - s.catch) < Math.abs(a - s.catch) ? c : a,
              );
              if (Math.abs(nearest - s.catch) <= 0.25) matched++;
              errors.push(s.catch - nearest);
              if (s.rate.value !== null) rates.push(s.rate.value);
              phases.push((s.finish - s.catch) / (s.nextCatch - s.catch));
            }
            for (const f of sim.frames) {
              n++;
              if (!assessPose(f.points, 0.5, 0.3, gateJoints(hidden)).accepted)
                gated++;
              if (!assessPose(f.points).accepted) oldGated++;
            }
          }
          const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
          const ms = (v: number) => `${Math.round(v * 1000)}`;
          lines.push(
            [
              label.padEnd(11),
              String(rate).padStart(3),
              `${cycles}/${found}/${matched}`.padStart(20),
              mean(rates).toFixed(2).padStart(9),
              Math.max(...rates.map((r) => Math.abs(r - rate)))
                .toFixed(2)
                .padStart(12),
              `${ms(mean(errors))} ms, ${ms(Math.max(...errors.map(Math.abs)))} ms`.padStart(
                21,
              ),
              `${Math.min(...phases).toFixed(2)}–${Math.max(...phases).toFixed(2)}`.padStart(
                12,
              ),
              `${Math.round((gated / n) * 100)}% (${Math.round((oldGated / n) * 100)}%)`,
            ].join(" | "),
          );
        }
      console.log(lines.join("\n"));
    },
  );
});
