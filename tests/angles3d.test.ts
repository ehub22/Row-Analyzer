// Normalized joint angles.
//
// The point of these is that the numbers are in the boat's frame: a knee at the
// catch is the same angle whatever corner of the launch the camera was in. So
// the tests build joints directly in boat coordinates — x along the shell
// towards the rowers' feet, y up, z to a rower's right — and check the
// definitions, the sign conventions coaches already use, and that anything not
// lifted comes back withheld with a reason rather than filled in.
import { describe, expect, it } from "vitest";
import {
  ANGLE_NAMES,
  ANGLE_UNIT,
  MIN_JOINT_CONFIDENCE,
  anglesAtStroke,
  angleRange,
  jointAngles,
  laybackAngle,
  type AngleFrame,
  type Joint3D,
} from "../src/core/angles";
import type { Metric, Stroke } from "../src/core/types";
import { withheld } from "../src/core/strokes";

const at = (
  x: number,
  y: number,
  z = 0,
  confidence = 0.9,
): Joint3D | undefined => ({ x, y, z, confidence, origin: "inferred" });

/** A seated rower: hip at the origin, trunk upright, thigh flat, shin down. */
function seated(): (Joint3D | undefined)[] {
  const j: (Joint3D | undefined)[] = Array.from(
    { length: 17 },
    () => undefined,
  );
  j[5] = at(-0.05, 0.6, -0.16); // shoulderL
  j[6] = at(-0.05, 0.6, 0.16); // shoulderR
  j[7] = at(0.1, 0.35, -0.2); // elbowL
  j[8] = at(0.1, 0.35, 0.2); // elbowR
  j[9] = at(0.2, 0.2, -0.2); // wristL
  j[10] = at(0.2, 0.2, 0.2); // wristR
  j[11] = at(0, 0, -0.16); // hipL
  j[12] = at(0, 0, 0.16); // hipR
  j[13] = at(0.45, 0.05, -0.1); // kneeL: thigh nearly flat, knee forward
  j[14] = at(0.45, 0.05, 0.1); // kneeR
  j[15] = at(0.15, -0.4, -0.1); // ankleL: shin back and down from the knee
  j[16] = at(0.15, -0.4, 0.1); // ankleR
  return j;
}

describe("angles in the boat frame", () => {
  it("measures an interior angle at the joint", () => {
    // A right angle, built so that only the definition can produce 90.
    const j: (Joint3D | undefined)[] = Array.from(
      { length: 17 },
      () => undefined,
    );
    j[11] = at(0, 1, 0);
    j[13] = at(0, 0, 0);
    j[15] = at(1, 0, 0);
    const knee = jointAngles(j).kneeLeft;
    expect(knee.value).toBeCloseTo(90, 6);
    expect(knee.unit).toBe(ANGLE_UNIT);
    // Confidence is the weakest joint's, never the average: an angle is only as
    // good as the shakiest lift behind it.
    j[15] = at(1, 0, 0, 0.4);
    expect(jointAngles(j).kneeLeft.confidence).toBeCloseTo(0.4, 6);
  });

  it("reads a compressed leg as a small knee and a flat thigh as a small hip", () => {
    const a = jointAngles(seated());
    // Thigh flat and forward, shin folded back under it: a compressed leg, the
    // kind a catch is made of, and the interior angle is small.
    expect(a.kneeLeft.value!).toBeLessThan(90);
    expect(a.kneeLeft.value!).toBeGreaterThan(30);
    expect(a.kneeRight.value!).toBeCloseTo(a.kneeLeft.value!, 6);
    // Trunk upright over a flat thigh: the hip is near 90 too.
    expect(a.hipLeft.value!).toBeGreaterThan(70);
    expect(a.hipLeft.value!).toBeLessThan(110);
    // Elbow: upper arm down and back, forearm down and forward.
    expect(a.elbowLeft.value!).toBeGreaterThan(0);
    expect(a.elbowLeft.value!).toBeLessThan(180);
  });

  it("measures the shin from the boat's vertical, positive towards the stern", () => {
    const j = seated();
    // Knee forward of the ankle and above it: the shin leans towards the feet.
    const shin = jointAngles(j).shinLeft;
    expect(shin.value!).toBeGreaterThan(0);
    // Straight down from the ankle reads zero, which is the "shins vertical"
    // position a catch is judged against.
    j[13] = at(0.15, 0.05, -0.1);
    expect(jointAngles(j).shinLeft.value!).toBeCloseTo(0, 6);
    // Knee behind the ankle reads negative.
    j[13] = at(-0.2, 0.05, -0.1);
    expect(jointAngles(j).shinLeft.value!).toBeLessThan(0);
  });

  it("makes layback negative forward of vertical and positive past it", () => {
    const j = seated();
    // Upright trunk: no layback either way.
    expect(jointAngles(j).layback.value!).toBeCloseTo(
      laybackAngle(j[5], j[6], j[11], j[12]).value!,
      6,
    );
    // Shoulders forward of the hips — towards the feet, the way a rower leans
    // at the catch — is negative.
    j[5] = at(0.35, 0.45, -0.16);
    j[6] = at(0.35, 0.45, 0.16);
    expect(jointAngles(j).layback.value!).toBeLessThan(0);
    expect(jointAngles(j).layback.value!).toBeCloseTo(-38, 0);
    // Past vertical, leaning away from the feet, is the finish: positive.
    j[5] = at(-0.35, 0.45, -0.16);
    j[6] = at(-0.35, 0.45, 0.16);
    expect(jointAngles(j).layback.value!).toBeGreaterThan(0);
  });

  it("averages both sides for layback, so a twist is not read as a lean", () => {
    const j = seated();
    // One shoulder forward, one back by the same amount: the trunk is twisted,
    // not leaned, and the average says so.
    j[5] = at(0.3, 0.55, -0.16);
    j[6] = at(-0.3, 0.55, 0.16);
    expect(Math.abs(jointAngles(j).layback.value!)).toBeLessThan(5);
    // And one side alone would have read a large lean.
    expect(
      Math.abs(laybackAngle(j[5], undefined, j[11], undefined).value!),
    ).toBeGreaterThan(20);
  });

  it("withholds an angle whose joints were not lifted, and says which", () => {
    const j = seated();
    j[13] = undefined; // a knee the gunwale hid
    const a = jointAngles(j);
    expect(a.kneeLeft.value).toBeNull();
    expect(a.kneeLeft.confidence).toBe(0);
    expect(a.kneeLeft.reason).toMatch(/the left knee was not lifted/);
    // The same knee's hip angle is gone too: it runs through the knee.
    expect(a.hipLeft.value).toBeNull();
    // The other side is untouched.
    expect(a.kneeRight.value).not.toBeNull();
    expect(a.shinLeft.value).toBeNull();
    expect(a.layback.value).not.toBeNull();
  });

  it("withholds a joint that is there but too weak to believe", () => {
    const j = seated();
    j[15] = at(0.15, -0.4, -0.1, MIN_JOINT_CONFIDENCE - 0.01);
    const a = jointAngles(j).kneeLeft;
    expect(a.value).toBeNull();
    expect(a.reason).toMatch(
      new RegExp(`${Math.round(MIN_JOINT_CONFIDENCE * 100)}%`),
    );
  });

  it("treats a discarded joint as absent even when the model returned one", () => {
    const j = seated();
    j[14] = { ...at(0.45, 0.05, 0.1)!, origin: "missing", confidence: 0.9 };
    expect(jointAngles(j).kneeRight.value).toBeNull();
    expect(jointAngles(j).kneeRight.reason).toMatch(/the right knee was not lifted/);
  });

  it("reports every angle it knows, all in degrees", () => {
    const a = jointAngles(seated());
    expect(Object.keys(a).sort()).toEqual([...ANGLE_NAMES].sort());
    for (const name of ANGLE_NAMES) expect(a[name].unit).toBe(ANGLE_UNIT);
  });
});

/** A stroke at 24 spm: catch 1.0, finish 1.75, next catch 2.5. */
const STROKE: Stroke = {
  seat: 3,
  index: 0,
  catch: 1.0,
  finish: 1.75,
  nextCatch: 2.5,
  rate: { value: 24, unit: "spm", confidence: 0.9 },
  ratio: withheld(":1", "test"),
  confidence: 0.9,
  observedFraction: 1,
};

function rows(times: number[], seat = 3, value = 50): AngleFrame[] {
  return times.map((time) => ({
    time,
    seat,
    reprojection: 1.5,
    angles: Object.fromEntries(
      ANGLE_NAMES.map((n) => [n, { value, unit: ANGLE_UNIT, confidence: 0.8 }]),
    ) as AngleFrame["angles"],
  }));
}

describe("angles at a stroke", () => {
  it("reads the nearest lifted frame to each event", () => {
    const r = rows([0.9, 1.02, 1.2, 1.7, 1.8, 2.4]);
    const { atCatch, atFinish } = anglesAtStroke(r, STROKE);
    expect(atCatch.kneeLeft.value).toBe(50);
    expect(atFinish.kneeLeft.value).toBe(50);
  });

  it("refuses to borrow an angle from the other end of the slide", () => {
    // Nothing lifted within a fifth of a stroke of the finish.
    const r = rows([0.9, 1.02, 1.2]);
    const { atCatch, atFinish } = anglesAtStroke(r, STROKE);
    expect(atCatch.kneeLeft.value).toBe(50);
    expect(atFinish.kneeLeft.value).toBeNull();
    expect(atFinish.kneeLeft.reason).toMatch(/a fifth of a stroke/);
  });

  it("says so when a seat has no lifted frames at all", () => {
    const { atCatch } = anglesAtStroke(rows([1.0], 5), STROKE);
    expect(atCatch.layback.value).toBeNull();
    expect(atCatch.layback.reason).toMatch(/no lifted frames for seat 3/);
  });
});

describe("an angle's range over a clip", () => {
  it("reports the 5th to 95th percentile, not the extremes", () => {
    // Twenty frames of a knee sweeping 50 to 170, plus one absurd sample: the
    // percentiles ignore it, min and max would not.
    const r: AngleFrame[] = [];
    for (let i = 0; i < 20; i++) {
      const row = rows([i * 0.1], 3, 50 + (i * 120) / 19)[0]!;
      r.push(row);
    }
    r.push({ ...rows([2.5], 3, 900)[0]! });
    const knee = angleRange(r, 3, "kneeLeft");
    expect(knee.lo.value!).toBeGreaterThan(50);
    expect(knee.lo.value!).toBeLessThan(70);
    expect(knee.hi.value!).toBeLessThan(180);
    expect(knee.hi.value!).toBeGreaterThan(150);
    expect(knee.observedFraction).toBe(1);
  });

  it("withholds when too few frames carry the angle", () => {
    const r = rows([0, 0.1, 0.2], 3);
    const knee = angleRange(r, 3, "kneeLeft");
    expect(knee.lo.value).toBeNull();
    expect(knee.lo.reason).toMatch(/Only 3 lifted frames/);
    expect(knee.hi.value).toBeNull();
  });

  it("counts withheld frames against how much of the clip was lifted", () => {
    const times = Array.from({ length: 20 }, (_, i) => i * 0.1);
    const r = rows(times, 3, 90);
    // A knee the gunwale hid for nearly a third of the clip: the range is still
    // reported, from what was lifted, and says how much that was.
    for (let i = 0; i < 6; i++)
      r[i]!.angles.kneeLeft = withheld(ANGLE_UNIT, "occluded");
    const knee = angleRange(r, 3, "kneeLeft");
    expect(knee.observedFraction).toBeCloseTo(14 / 20, 6);
    expect(knee.lo.value).toBeCloseTo(90, 6);
    // And the confidence cannot claim more of the clip than was lifted.
    expect(knee.lo.confidence).toBeLessThanOrEqual(14 / 20 + 0.2 + 1e-9);
  });
});
