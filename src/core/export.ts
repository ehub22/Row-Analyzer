import { jsPDF } from "jspdf";
import type { Session } from "./types";
import { mean, crewTiming, segmentStrokes } from "./strokes";
import { jointNames } from "./keypoints";
import { MODEL_UNAVAILABLE_FLAG } from "./bodyLayout";
import { expectedOcclusion } from "./occlusion";
import { KNEE_JOINTS } from "./knees";
export function download(
  data: BlobPart,
  name: string,
  type = "application/json",
) {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export const exportJSON = (session: Session) =>
  download(JSON.stringify(session), "row-analysis.json");
export function exportCSV(s: Session) {
  const rows = [
    ["Source", s.demo ? "SYNTHETIC DEMO" : "Experimental video analysis"],
    ["Pose joints", s.poseJoints ?? 133],
    [
      "Seat",
      "Stroke",
      "Catch (s)",
      "Finish proxy (s)",
      "Next catch (s)",
      "Rate (spm)",
      "Drive/recovery",
      "Confidence",
      "Observed fraction",
    ],
    ...s.strokes.map((r) => [
      r.seat,
      r.index,
      r.catch,
      r.finish,
      r.nextCatch,
      r.rate.value ?? "withheld",
      r.ratio.value ?? "withheld",
      r.confidence,
      r.observedFraction,
    ]),
  ];
  download(
    rows.map((row) => row.join(",")).join("\n"),
    "row-strokes.csv",
    "text/csv",
  );
}
export function exportLabels(s: Session) {
  const frames = s.frames.filter((f) =>
    f.points.some((p) => p.origin === "manual"),
  );
  download(
    JSON.stringify({
      format: "row-analyzer-corrections-v1",
      source: s.name,
      synthetic: s.demo,
      pose_joints: s.poseJoints ?? 133,
      image_size: [s.width, s.height],
      keypoint_names: jointNames,
      note: "COCO-style partial labels. Only manual points exported; extract exact timestamp frames locally before training.",
      annotations: frames.map((f) => ({
        seat: f.seat,
        timestamp: f.time,
        box: f.box,
        keypoints: f.points.flatMap((p) =>
          p.origin === "manual" ? [p.x, p.y, 2] : [0, 0, 0],
        ),
      })),
      corrections: s.corrections,
    }),
    "rowing-labels.json",
  );
}
export function exportPDF(s: Session) {
  const doc = new jsPDF();
  let y = 20;
  const text = (value: string, size = 11) => {
    doc.setFontSize(size);
    for (const line of doc.splitTextToSize(value, 175)) {
      if (y > 276) {
        doc.addPage();
        y = 20;
      }
      doc.text(line, 18, y);
      y += size * 0.5 + 2;
    }
  };
  text("ROW ANALYZER", 11);
  text("Stroke review", 26);
  text(
    s.demo
      ? "SYNTHETIC DEMONSTRATION - NOT ATHLETE DATA"
      : "EXPERIMENTAL ANALYSIS - REVIEW BEFORE COACHING",
    10,
  );
  text(
    `${s.name} | ${s.boat.name} | ${s.boat.discipline} | ${s.boat.seats} seats`,
  );
  text(`Model: ${s.model}`);
  text(`Pose layout: ${s.poseJoints ?? 133} COCO joints`);
  if (s.boat.feetHidden !== undefined)
    text(
      s.boat.feetHidden
        ? "Feet: hidden by the hull. Ankle and foot keypoints were not estimated; cycle timing uses hip and knee motion."
        : s.poseJoints === 17
          ? "Feet: visible. The body-only model may estimate ankles; toe and heel keypoints are not provided and are never counted as model observations."
          : "Feet: visible. Ankle and foot keypoints were tracked.",
    );
  // Counted from the frames themselves, so an imported session reports what it
  // contains rather than trusting a claim.
  const occludedKnees = s.frames.reduce(
    (n, f) =>
      n + KNEE_JOINTS.filter((j) => expectedOcclusion(f.points[j])).length,
    0,
  );
  if (occludedKnees)
    text(
      `Knees at the finish: ${occludedKnees} knee samples were discarded as gunwale or oar occlusion and marked missing with a reason. With the legs flat the knee drops to the hull edge and the loom crosses it, and a pose model still answers - pinned to the gunwale or dragged onto the shaft - so those answers are not tracked, counted as evidence, or drawn. The catch end of the stroke is untouched; each finish proxy is timed from the bridged trough instead of from a guess.`,
    );
  text(`Runtime: ${s.provider}`);
  text(
    `Camera azimuth: ${s.camera.azimuth === null ? "withheld" : s.camera.azimuth + " degrees"} | confidence ${Math.round(s.camera.confidence * 100)}%`,
  );
  text(s.camera.reason);
  if (s.performance) text(`Performance: ${s.performance.summary}`, 9);
  for (const w of s.warnings) text(`Caution: ${w}`, 10);
  y += 5;
  for (let seat = 1; seat <= s.boat.seats; seat++) {
    const rows = s.strokes.filter((r) => r.seat === seat);
    text(
      `Seat ${seat}${seat === 1 ? " (bow seat)" : seat === s.boat.seats ? " (stroke seat)" : ""}`,
      15,
    );
    text(
      `Complete cycles: ${rows.length}. Rate: ${mean(rows.map((r) => r.rate.value))?.toFixed(1) ?? "withheld"} spm. Evidence confidence: ${Math.round((mean(rows.map((r) => r.confidence)) ?? 0) * 100)}%.`,
    );
    const timing = crewTiming(s.strokes, s.boat.seats).filter(
      (r) => r.seat === seat,
    );
    const lag = mean(timing.map((r) => r.catchMs));
    if (lag !== null)
      text(
        `Seat ${seat} reaches the catch proxy about ${(Math.abs(lag) / 1000).toFixed(2)} sec ${lag >= 0 ? "late" : "early"} relative to stroke seat. Verify against blade entry.`,
      );
  }
  text("Per-stroke cycle timing", 16);
  for (const r of s.strokes)
    text(
      `Seat ${r.seat} / stroke ${r.index}: catch ${r.catch.toFixed(3)} s, finish proxy ${r.finish.toFixed(3)} s, rate ${r.rate.value?.toFixed(1) ?? "withheld"} spm, D:R ${r.ratio.value?.toFixed(2) ?? "withheld"}, confidence ${Math.round(r.confidence * 100)}%.`,
      9,
    );
  text("Withheld measurements", 16);
  text(
    "Normalized shin/body/joint angles, layback, trunk twist, sequencing faults, oar angles, release quality, handle crossover/height, boat set and split per 500 m require additional validated evidence.",
  );
  text("Camera guidance", 16);
  text(
    "Film alongside the shell, near perpendicular with a slight offset to separate rowers. Keep hands, hips, knees and the whole slide visible; on the water the feet sit below the gunwale and are not needed. For individual technique, use a closer single-seat view. Avoid bow-on footage for layback assessment.",
  );
  doc.save("row-analysis-report.pdf");
}
export function validateSession(value: unknown): Session {
  const s = value as Session;
  if (
    !s ||
    s.version !== 1 ||
    !Array.isArray(s.frames) ||
    !Array.isArray(s.strokes) ||
    !s.boat ||
    !s.camera ||
    !Array.isArray(s.timestamps) ||
    !Number.isFinite(s.width) ||
    s.width <= 0 ||
    !Number.isFinite(s.height) ||
    s.height <= 0 ||
    !Number.isFinite(s.duration) ||
    s.duration <= 0 ||
    !Number.isInteger(s.boat.seats) ||
    s.boat.seats < 1 ||
    s.boat.seats > 8 ||
    !Array.isArray(s.corrections) ||
    !Array.isArray(s.warnings) ||
    typeof s.name !== "string" ||
    typeof s.demo !== "boolean" ||
    typeof s.model !== "string" ||
    typeof s.provider !== "string" ||
    (s.poseJoints !== undefined &&
      s.poseJoints !== 17 &&
      s.poseJoints !== 133) ||
    !Number.isInteger(s.boat.bestSeat) ||
    s.boat.bestSeat < 1 ||
    s.boat.bestSeat > s.boat.seats ||
    typeof s.boat.name !== "string" ||
    typeof s.boat.bowLeft !== "boolean" ||
    (s.boat.feetHidden !== undefined &&
      typeof s.boat.feetHidden !== "boolean") ||
    !["sweep", "scull"].includes(s.boat.discipline) ||
    !Array.isArray(s.boat.rigging) ||
    s.boat.rigging.length < s.boat.seats ||
    s.boat.rigging.some((r) => !["port", "starboard"].includes(r)) ||
    s.timestamps.some(
      (t) => !Number.isFinite(t) || t < 0 || t > s.duration + 0.1,
    ) ||
    s.warnings.some((w) => typeof w !== "string") ||
    !["unresolved", "calibrated", "synthetic"].includes(s.camera.source) ||
    !Number.isFinite(s.camera.confidence) ||
    s.camera.confidence < 0 ||
    s.camera.confidence > 1 ||
    (s.camera.azimuth !== null &&
      (!Number.isFinite(s.camera.azimuth) ||
        s.camera.azimuth < 0 ||
        s.camera.azimuth > 90)) ||
    typeof s.camera.reason !== "string"
  )
    throw Error(
      "Invalid Row Analyzer session. Import an exported version 1 JSON session.",
    );
  if (s.frames.length > 160000) throw Error("Session exceeds the frame limit.");
  for (const f of s.frames) {
    if (
      !f ||
      !Number.isFinite(f.time) ||
      f.time < 0 ||
      f.time > s.duration + 0.1 ||
      !Number.isInteger(f.seat) ||
      f.seat < 1 ||
      f.seat > s.boat.seats ||
      !f.box ||
      !Number.isFinite(f.box.x) ||
      !Number.isFinite(f.box.y) ||
      !Number.isFinite(f.box.width) ||
      f.box.width <= 0 ||
      !Number.isFinite(f.box.height) ||
      f.box.height <= 0 ||
      !Number.isFinite(f.identityConfidence) ||
      f.identityConfidence < 0 ||
      f.identityConfidence > 1 ||
      (f.boxSource !== undefined &&
        !["detected", "predicted"].includes(f.boxSource)) ||
      (f.detectorAge !== undefined &&
        (!Number.isInteger(f.detectorAge) ||
          f.detectorAge < 0 ||
          f.detectorAge > 2)) ||
      (f.boxSource === "predicted" &&
        (!f.detectorAge ||
          f.identityConfidence > Math.max(0, 0.9 - f.detectorAge * 0.15))) ||
      (f.boxSource === "detected" && f.detectorAge !== 0) ||
      !Array.isArray(f.points) ||
      f.points.length !== 133 ||
      f.points.some(
        (p) =>
          !p ||
          !Number.isFinite(p.x) ||
          !Number.isFinite(p.y) ||
          !Number.isFinite(p.confidence) ||
          p.confidence < 0 ||
          p.confidence > 1 ||
          !["observed", "inferred", "manual", "missing"].includes(p.origin) ||
          (p.flags !== undefined &&
            (!Array.isArray(p.flags) ||
              p.flags.some((flag) => typeof flag !== "string"))),
      ) ||
      (s.poseJoints === 17 &&
        f.points
          .slice(17)
          .some(
            (p) =>
              p.origin !== "manual" &&
              (p.origin !== "missing" ||
                p.confidence !== 0 ||
                !p.flags?.includes(MODEL_UNAVAILABLE_FLAG)),
          ))
    )
      throw Error("Invalid frame/keypoint data in session.");
  }
  // Recompute derived metrics, rather than trusting imported claims or malformed rows.
  s.strokes = segmentStrokes(s.frames);
  s.timestamps = [...new Set(s.frames.map((f) => f.time))].sort(
    (a, b) => a - b,
  );
  if (!s.demo && s.camera.source === "synthetic")
    throw Error(
      "A recorded session cannot claim synthetic camera calibration.",
    );
  return s;
}
