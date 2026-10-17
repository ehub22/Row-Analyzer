// The camera solve.
//
// Normalized angles need the boat's frame, and the boat's frame needs to know
// how the camera was turned relative to the shell. ROADMAP.md is blunt that
// this is not a consequence of having keypoints: "A shell axis in the image
// alone does not determine azimuth", and an unknown focal length leaves azimuth
// and distance locked together. These tests pin both halves of that claim —
// what the solve recovers when it is given a spacing, a lens and a crew of
// three or more seats, and that it abstains, with a reason a user can act on,
// when it is not.
//
// Ground truth is tests/fixtures/rowingSim3d.ts: a crew in boat-frame metres
// projected through a pinhole camera at a known azimuth, focal length and
// distance. That validates the geometry — the projection, the fit that inverts
// it, the sign convention shared with core/geometry.ts — and it is not evidence
// about real footage. Monocular depth recovery on a launch clip with a real
// hull, real spray and a real pan is still an open gate (docs/VALIDATION.md).
import { describe, expect, it } from "vitest";
import { simulateCrew, type Sim3DResult } from "./fixtures/rowingSim3d";
import {
  MIN_SEATS_FOR_SOLVE,
  fitSeatLine,
  focalPx,
  seatMarks,
  solveAllowsAngles,
  solveCamera,
  solveSupport,
} from "../src/core/camera";
import type { BoatCalibration } from "../src/core/types";

const IMAGE = { width: 1280, height: 720 };
/** An eight with its bow to the left, as a launch following the crew sees it. */
const EIGHT = { seats: 8, bowLeft: true };

/** What an operator would enter: a spacing off the rigging tape, and the lens. */
function calibration(
  sim: Sim3DResult,
  extra: BoatCalibration = {},
): BoatCalibration {
  return {
    seatSpacing: sim.dimensions.seatSpacing,
    focalPx: sim.camera.focal,
    axisVerified: true,
    level: true,
    ...extra,
  };
}

const solve = (sim: Sim3DResult, extra: BoatCalibration = {}) =>
  solveCamera(sim.frames, EIGHT, calibration(sim, extra), IMAGE);

/**
 * The depth a solve should report: the seat line's centre, and the seat line is
 * where the hips are, not where the stretchers are — a hip sits some 70 cm from
 * its own seat's datum. Depth is linear along the shell and the seat range is
 * centred on it, so the mean over every seat and frame of the truth is exactly
 * that point, and needs no modelling of the fixture's internals.
 */
const hipDepth = (sim: Sim3DResult) =>
  sim.truth.reduce((sum, t) => sum + t.depth, 0) / sim.truth.length;

describe("the seat line as a projective map", () => {
  it("recovers the map that generated it", () => {
    // u(x) = (a·x + b) / (c·x + 1) with the shell's own numbers: an 18° view,
    // 16 m out, a 1800 px lens, the frame centre as principal point.
    const a = 1229.5,
      b = 640,
      c = -0.0203;
    const xs = Array.from({ length: 8 }, (_, i) => (i - 3.5) * 0.9);
    const fit = fitSeatLine(
      xs,
      xs.map((x) => (a * x + b) / (c * x + 1)),
    );
    expect(fit).not.toBeNull();
    expect(fit!.a).toBeCloseTo(a, 6);
    expect(fit!.b).toBeCloseTo(b, 6);
    expect(fit!.c).toBeCloseTo(c, 8);
    expect(fit!.residual).toBeLessThan(1e-6);
  });

  it("needs three seats, and says so with fewer", () => {
    expect(MIN_SEATS_FOR_SOLVE).toBe(3);
    expect(fitSeatLine([0, 0.9], [100, 200])).toBeNull();
    // Three is enough: three parameters, three seats.
    expect(fitSeatLine([0, 1, 2], [100, 210, 300])).not.toBeNull();
  });

  it("does not care where along the shell the origin sits", () => {
    const a = 1229.5,
      b = 640,
      c = -0.0203;
    const xs = [-3.15, -2.25, -1.35, -0.45],
      us = xs.map((x) => (a * x + b) / (c * x + 1));
    const from = fitSeatLine(xs, us)!,
      shifted = fitSeatLine(
        xs.map((x) => x + 3.15),
        us,
      )!;
    // The vanishing point — what the azimuth comes from — is invariant.
    expect(from.a / from.c).toBeCloseTo(shifted.a / shifted.c, 6);
  });

  it("converts a 35 mm equivalent with the image width alone", () => {
    expect(focalPx({ focal35mm: 26 }, 1280)).toBeCloseTo((26 * 1280) / 36, 6);
    // A focal length in pixels wins over an equivalent.
    expect(focalPx({ focalPx: 900, focal35mm: 26 }, 1280)).toBe(900);
    expect(focalPx({}, 1280)).toBeNull();
    expect(focalPx({ focalPx: 0 }, 1280)).toBeNull();
  });
});

describe("reading the crew's seats", () => {
  it("averages both hips of every seat over the clip", () => {
    const sim = simulateCrew({ seconds: 4 });
    const marks = seatMarks(sim.frames);
    expect(marks).toHaveLength(8);
    expect(marks.map((m) => m.seat)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    // A stroke is 2.5 s, so 4 s is a stroke and a half of hips per seat.
    expect(solveSupport(marks)!).toBeGreaterThan(100);
    // The mean spacing is the true spacing scaled by the lens and the distance,
    // and it is not uniform: that gradient is the entire cue.
    const gaps = marks.slice(1).map((m, i) => m.u - marks[i].u);
    expect(Math.abs(gaps[0])).toBeGreaterThan(20);
    expect(Math.abs(gaps.at(-1)!)).not.toBeCloseTo(Math.abs(gaps[0]), 0);
  });

  it("ignores seats it has hardly seen", () => {
    const sim = simulateCrew({ seconds: 4 });
    // Two seats' worth of frames is not a crew.
    expect(seatMarks(sim.frames.slice(0, 20))).toHaveLength(0);
  });
});

describe("solving a crew filmed from a launch", () => {
  it.each([10, 18, 30, 45])(
    "recovers a %s° view: azimuth, distance and scale",
    (azimuth) => {
      const sim = simulateCrew({ seconds: 8, camera: { azimuth } });
      const s = solve(sim);
      expect(s.source).toBe("calibrated");
      expect(solveAllowsAngles(s)).toBe(true);
      expect(Math.abs(s.azimuth! - azimuth)).toBeLessThan(0.15);
      const truthDepth = hipDepth(sim);
      // Not the camera's 16 m: the depth of the line the hips actually lie on,
      // which is 70 cm further out along the shell's own axis times sin(azimuth).
      expect(Math.abs(s.distance! - truthDepth)).toBeLessThan(0.1);
      // Pixels per metre there, against what the projection gives for a metre
      // *along the shell axis*: f·cos(az)·distance ÷ depth², the cosine being
      // the axis's foreshortening and the extra distance/depth the fact that
      // the axis runs away from the camera as it runs along the boat. A metre
      // across the boat would be f/depth instead — which is why metric outputs
      // stay withheld until something says which one it means.
      const scale =
        (sim.camera.focal *
          Math.cos((azimuth * Math.PI) / 180) *
          sim.camera.distance) /
        (truthDepth * truthDepth);
      expect(Math.abs(s.pxPerMetre! - scale)).toBeLessThan(scale * 0.01);
      expect(s.cues!.length).toBeGreaterThan(2);
      expect(s.confidence).toBeGreaterThanOrEqual(0.8);
    },
  );

  it("recovers the sign of the turn, not only its size", () => {
    // The same crew filmed from the other side of the shell. The azimuth comes
    // out negated, because positive means the stern end — the end towards the
    // rowers' feet — is the nearer one: core/geometry.ts's `rotateY` convention,
    // of which `toBoatFrame` is the inverse. Which end of the *image* the bow
    // lands on is a separate question, set by how the crew was framed, so the
    // boat still says the bow is on the left and the solve still agrees with it.
    const a = simulateCrew({ seconds: 8, camera: { azimuth: 24 } });
    const b = simulateCrew({ seconds: 8, camera: { azimuth: -24 } });
    expect(solve(a).azimuth!).toBeCloseTo(24, 1);
    const other = solveCamera(
      b.frames,
      { seats: 8, bowLeft: true },
      calibration(b),
      IMAGE,
    );
    expect(other.source).toBe("calibrated");
    expect(other.azimuth!).toBeCloseTo(-24, 1);
    // The depths are not symmetric — the hip line leans towards the camera from
    // one side and away from it from the other — so each is checked against its
    // own truth rather than against the other solve.
    expect(other.distance!).toBeCloseTo(hipDepth(b), 1);
    expect(solve(a).distance!).toBeCloseTo(hipDepth(a), 1);
  });

  it("refuses a clip that is mirrored left to right", () => {
    // Flipping the image is not the same as filming from the other side: no
    // camera position produces it, and the solve says so rather than reporting
    // an azimuth that would put every angle back to front. The metadata here is
    // consistent with what is on screen — the bow really is on the right.
    const sim = simulateCrew({
      seconds: 8,
      feetToRight: false,
      camera: { azimuth: 24 },
    });
    const s = solveCamera(
      sim.frames,
      { seats: 8, bowLeft: false },
      calibration(sim),
      IMAGE,
    );
    expect(s.source).toBe("unresolved");
    expect(s.reason).toMatch(/mirrored/);
    expect(solveAllowsAngles(s)).toBe(false);
  });

  it("finds the principal point itself when nobody supplies one", () => {
    const sim = simulateCrew({ seconds: 8, camera: { azimuth: 24 } });
    const s = solve(sim);
    // The fixture frames the shell's centre, so the derived principal point has
    // to land on the frame centre — from the seat line and the lens alone.
    expect(s.principal).not.toBeNull();
    expect(s.principal!.x).toBe(IMAGE.width / 2);
    expect(Math.abs(s.azimuth! - 24)).toBeLessThan(0.6);
    // And the assumption is priced in the cues, not buried.
    const pp = s.cues!.find((c) => c.name === "principal point")!;
    expect(pp.detail).toMatch(/assumed at the frame centre/);
    expect(pp.confidence).toBeLessThan(1);
  });

  it("reports what a wrong seat spacing does and does not change", () => {
    const sim = simulateCrew({ seconds: 8, camera: { azimuth: 24 } });
    const wrong = solve(sim, { seatSpacing: 1.2 }); // true spacing is 0.9
    // The azimuth is a ratio of the spacing to itself, so it survives; the
    // distance and the scale are absolute, so they do not. Angles are safe
    // under a mis-measured tape, metric lengths are not — which is why metric
    // outputs stay withheld while normalized angles do not.
    expect(wrong.source).toBe("calibrated");
    expect(wrong.azimuth!).toBeCloseTo(solve(sim).azimuth!, 6);
    expect(wrong.distance! / solve(sim).distance!).toBeCloseTo(1.2 / 0.9, 3);
    expect(wrong.pxPerMetre! / solve(sim).pxPerMetre!).toBeCloseTo(
      0.9 / 1.2,
      3,
    );
  });

  it("takes a lens specification instead of a focal length in pixels", () => {
    const sim = simulateCrew({ seconds: 8, camera: { azimuth: 24 } });
    const f35 = (sim.camera.focal * 36) / IMAGE.width;
    const s = solve(sim, { focalPx: null, focal35mm: f35 });
    expect(s.source).toBe("calibrated");
    expect(Math.abs(s.azimuth! - 24)).toBeLessThan(0.5);
  });
});

describe("abstaining", () => {
  it("refuses a square-on view, where there is nothing to solve", () => {
    const sim = simulateCrew({ seconds: 8, camera: { azimuth: 0 } });
    const s = solve(sim);
    expect(s.source).toBe("unresolved");
    expect(s.azimuth).toBeNull();
    expect(solveAllowsAngles(s)).toBe(false);
    // Either the seat line has no perspective at all, or what it has is inside
    // the noise: both are the same honest refusal.
    expect(s.reason).toMatch(/parallel to the image plane|off square/i);
  });

  it("refuses a view so nearly square-on that the turn is inside the noise", () => {
    const sim = simulateCrew({ seconds: 8, camera: { azimuth: 0.4 } });
    const s = solve(sim);
    expect(solveAllowsAngles(s)).toBe(false);
    expect(s.azimuth === null || Math.abs(s.azimuth) < 3).toBe(true);
  });

  it("refuses without a focal length, and says why", () => {
    const sim = simulateCrew({ seconds: 8 });
    const s = solve(sim, { focalPx: null });
    expect(s.source).toBe("unresolved");
    expect(s.reason).toMatch(/focal length/i);
    expect(s.reason).toMatch(/azimuth and distance/i);
  });

  it("refuses until the clip is confirmed level", () => {
    const sim = simulateCrew({ seconds: 8 });
    const s = solve(sim, { level: false });
    expect(s.source).toBe("unresolved");
    expect(s.reason).toMatch(/level/i);
    expect(solveAllowsAngles(s)).toBe(false);
  });

  it("refuses without a seat spacing, and says why", () => {
    const sim = simulateCrew({ seconds: 8 });
    const s = solve(sim, { seatSpacing: null });
    expect(s.source).toBe("unresolved");
    expect(s.reason).toMatch(/seat spacing/i);
  });

  it("refuses a crew too small to fit a projective map", () => {
    const sim = simulateCrew({ seconds: 8, seats: 2 });
    const s = solve(sim);
    expect(s.source).toBe("unresolved");
    expect(s.reason).toMatch(/at least 3 seats/i);
  });

  it("distrusts itself when the boat's bow direction contradicts the hips", () => {
    const sim = simulateCrew({ seconds: 8, camera: { azimuth: 24 } });
    // Seat 1 is on the left of this image; claiming the bow is on the right
    // would mirror the whole solve.
    const s = solveCamera(
      sim.frames,
      { seats: 8, bowLeft: false },
      calibration(sim),
      IMAGE,
    );
    const bow = s.cues!.find((c) => c.name === "bow direction");
    expect(bow).toBeDefined();
    expect(bow!.confidence).toBeLessThan(0.5);
    expect(bow!.detail).toMatch(/mirror/);
    expect(solveAllowsAngles(s)).toBe(false);
  });

  it("distrusts one azimuth for a clip the camera turned during", () => {
    const sim = simulateCrew({
      seconds: 8,
      camera: { azimuth: 18 },
      azimuthDrift: 14,
    });
    const s = solve(sim);
    expect(solveAllowsAngles(s)).toBe(false);
    // Either the single fit is already too poor, or the two halves of the clip
    // disagree; both are honest refusals, and both say so.
    const unstable =
      /does not fit one equally spaced straight line/.test(s.reason) ||
      s.cues!.some((c) => c.name === "stability" && c.confidence < 0.8);
    expect(unstable).toBe(true);
  });

  it("keeps the axis it measured even when it cannot solve", () => {
    const sim = simulateCrew({ seconds: 8, camera: { azimuth: 0 } });
    const s = solve(sim);
    // A square-on view still has a shell axis in the image, which is what the
    // older estimate reported; it just carries no azimuth.
    expect(Math.abs(s.axis.x)).toBeGreaterThan(0.9);
    expect(Math.abs(s.axis.y)).toBeLessThan(0.2);
  });
});
