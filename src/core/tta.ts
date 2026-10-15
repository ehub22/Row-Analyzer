// Test-time augmentation policy.
//
// Four passes per crop (two scales × flip) is the accuracy-first setting, but
// pose inference is the dominant cost of a run: an eight at 4 passes is 32
// whole-body inferences per frame. Most frames of most clips do not need all
// four — a clean, unoccluded crop already peaks above the observed/missing
// threshold on every body joint — so `adaptive` runs the single pass first and
// pays for the remaining three only when that pass is weak.
//
// The gate deliberately looks only at the 17 body joints (indices 0–16 of the
// 133-joint whole-body layout). Face, hand and foot joints are the ones most
// likely to be occluded by oars, water and the boat, and they are not what the
// stroke metrics are built on, so requiring them would make every crop pay for
// the full four-pass fusion.
//
// The ankles (15, 16) are body joints, but on-water footage hides them for the
// whole clip (feet are strapped below the gunwale). Extra passes cannot reveal
// a joint that is not in the image, so joints the session declares hidden are
// left out of the gate (`gateJoints`). Without that, a hidden ankle below 0.3
// rejected nearly every crop and `adaptive` silently cost as much as `always`.
//
// When the gate rejects a crop, the *same four passes* run and fuse exactly as
// before; `adaptive` therefore only ever skips passes that were already
// reliable, and never changes the result of a crop that needed fusion.

import type { Box, Point } from "./types";
import { paddedCrop } from "./detection";

export type TtaMode = "off" | "adaptive" | "always";

/** Whole-body layout: COCO-17 body joints come first. */
export const BODY_JOINTS: readonly number[] = Array.from(
  { length: 17 },
  (_, i) => i,
);

/** Body joints the adaptive gate should require, given joints known to be
 * hidden for the whole session (see core/occlusion.ts `hiddenJoints`). */
export function gateJoints(hidden: readonly number[] = []): readonly number[] {
  return hidden.length
    ? BODY_JOINTS.filter((j) => !hidden.includes(j))
    : BODY_JOINTS;
}

export interface PosePass {
  /** Padded crop actually fed to the model for this pass. */
  crop: Box;
  flip: boolean;
  scale: number;
}

/** Scales and flips of the full fusion, in their original order. */
export const TTA_SCALES = [1.25, 1.5] as const;

/** All passes `mode` may run for one detection, first pass first. */
export function passPlan(
  mode: TtaMode,
  box: Box,
  width: number,
  height: number,
): PosePass[] {
  const scales = mode === "off" ? [1.25] : TTA_SCALES;
  const flips = mode === "off" ? [false] : [false, true];
  const passes: PosePass[] = [];
  for (const scale of scales)
    for (const flip of flips)
      passes.push({
        crop: paddedCrop(box, width, height, scale),
        flip,
        scale,
      });
  return passes;
}

export interface PoseAssessment {
  /** True when the single pass was good enough to skip the extra passes. */
  accepted: boolean;
  /** Mean confidence across the body joints of the pass. */
  mean: number;
  /** Body joints below the observed/missing threshold. */
  weak: number;
  reason: string;
}

/**
 * Confidence gate for the adaptive policy. `minMean`/`minJoint` mirror the
 * decoders' 0.3 observed/missing threshold and one comfortably-above-it level.
 * `joints` defaults to all 17 body joints; pass `gateJoints(hidden)` to leave
 * out joints that cannot be visible in this footage.
 */
export function assessPose(
  points: Point[],
  minMean = 0.5,
  minJoint = 0.3,
  joints: readonly number[] = BODY_JOINTS,
): PoseAssessment {
  let total = 0,
    weak = 0;
  for (const j of joints) {
    const p = points[j];
    const c = p && p.origin !== "missing" ? p.confidence : 0;
    total += c;
    if (c < minJoint) weak++;
  }
  const mean = total / Math.max(1, joints.length);
  return {
    accepted: weak === 0 && mean >= minMean,
    mean,
    weak,
    reason:
      weak === 0 && mean >= minMean
        ? "single pass accepted"
        : `${weak} of ${joints.length} body joints below ${minJoint} (mean ${mean.toFixed(2)})`,
  };
}

/** Human summary of the mode, for status lines, notes and session exports. */
export function describeTta(mode: TtaMode): string {
  return mode === "off"
    ? "single pass"
    : mode === "adaptive"
      ? "adaptive flip + multi-scale TTA"
      : "flip + multi-scale TTA";
}
