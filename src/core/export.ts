import { jsPDF } from "jspdf";
import type { Session } from "./types";
import { mean, crewTiming, segmentStrokes } from "./strokes";
import { jointNames } from "./keypoints";
import { MODEL_UNAVAILABLE_FLAG } from "./bodyLayout";
import { expectedOcclusion } from "./occlusion";
import { KNEE_JOINTS } from "./knees";
import {
  ANGLE_LABELS,
  ANGLE_NAMES,
  angleRange,
  type AngleName,
} from "./angles";
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
  const rows: (string | number)[][] = [
    ["Source", s.demo ? "SYNTHETIC DEMO" : "Experimental video analysis"],
    ["Pose joints", s.poseJoints ?? 133],
    // The camera, with its provenance: an exported number without the cues
    // behind it is how a guess starts looking like a measurement.
    [
      "Camera azimuth (deg)",
      s.camera.azimuth === null ? "withheld" : s.camera.azimuth.toFixed(2),
    ],
    ["Camera confidence", `${Math.round(s.camera.confidence * 100)}%`],
    ["Camera source", s.camera.source],
    [
      "Camera distance (m)",
      s.camera.distance === null || s.camera.distance === undefined
        ? "withheld"
        : s.camera.distance.toFixed(2),
    ],
    [
      "Scale along the shell (px/m)",
      s.camera.pxPerMetre === null || s.camera.pxPerMetre === undefined
        ? "withheld"
        : s.camera.pxPerMetre.toFixed(1),
    ],
    ["Seat spacing (m)", s.calibration?.seatSpacing ?? "not supplied"],
    [
      "Focal length (px)",
      s.camera.focalPx ?? s.calibration?.focalPx ?? "not supplied",
    ],
    ["Clip confirmed level", s.calibration?.level === true ? "yes" : "no"],
    ["Camera reason", s.camera.reason.replaceAll(",", ";")],
    ...(s.camera.cues ?? []).map((c) => [
      `Cue: ${c.name}`,
      `${Math.round(c.confidence * 100)}%`,
      c.detail.replaceAll(",", ";"),
    ]),
    [
      "3D lift",
      s.lift?.ran
        ? `${s.lift.placed ?? 0}/${s.lift.attempted ?? 0} poses placed; ${s.lift.model ?? ""}`
        : `not run: ${s.lift?.reason ?? "no lift on this session"}`.replaceAll(
            ",",
            ";",
          ),
    ],
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
/**
 * Per-frame normalized angles, one row per analyzed frame, aligned to the
 * session's frames.
 *
 * A withheld angle is written as "withheld", never as a blank and never as a
 * zero: a blank in a spreadsheet gets filled in by whoever opens it, and a zero
 * is a measurement. The reason column carries the refusal, so a coach filtering
 * the sheet can see which frames were dropped and why.
 */
export function exportAnglesCSV(s: Session) {
  const rows = s.angles ?? [];
  if (!rows.length) return;
  const lines = [
    [
      "Time (s)",
      "Seat",
      "Reprojection (px)",
      ...ANGLE_NAMES.map((n) => `${ANGLE_LABELS[n]} (deg)`),
      "Withheld because",
    ],
    ...rows.map((r) => {
      const reasons = [
        ...new Set(
          ANGLE_NAMES.map((n) => r.angles[n].reason).filter((x): x is string =>
            Boolean(x),
          ),
        ),
      ];
      return [
        r.time.toFixed(3),
        r.seat,
        r.reprojection === null ? "" : r.reprojection.toFixed(2),
        ...ANGLE_NAMES.map((n) =>
          r.angles[n].value === null
            ? "withheld"
            : r.angles[n].value!.toFixed(2),
        ),
        reasons.join(" | ").replaceAll(",", ";"),
      ];
    }),
  ];
  download(
    lines.map((row) => row.join(",")).join("\n"),
    "row-angles.csv",
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
    `Camera: ${s.camera.azimuth === null ? "azimuth withheld" : `${s.camera.azimuth.toFixed(1)} degrees off square, ${s.camera.azimuth > 0 ? "stern" : "bow"} end nearer`} | confidence ${Math.round(s.camera.confidence * 100)}%${
      s.camera.distance ? ` | ${s.camera.distance.toFixed(1)} m out` : ""
    }${s.camera.pxPerMetre ? ` | ${s.camera.pxPerMetre.toFixed(0)} px per metre along the shell` : ""}`,
  );
  text(s.camera.reason);
  for (const c of s.camera.cues ?? [])
    text(`Cue ${c.name} (${Math.round(c.confidence * 100)}%): ${c.detail}`, 9);
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
  // Normalized angles, when a lift produced them: the seat's own range over the
  // clip, and what was withheld, in the same place, so a reader cannot mistake
  // one for the other.
  const angleRows = s.angles ?? [];
  if (angleRows.length) {
    text("Normalized angles (boat frame)", 16);
    text(
      s.lift?.ran
        ? `${s.lift.placed ?? 0} of ${s.lift.attempted ?? 0} poses were placed against the solved camera${
            s.lift.refused ? `, ${s.lift.refused} refused` : ""
          }${
            s.lift.reprojection
              ? `, median reprojection ${s.lift.reprojection.toFixed(1)} px`
              : ""
          }. Model: ${s.lift.model ?? "unrecorded"}. These are inferences about depth from a learned lift, checked against the keypoints they were fitted to; they are not observations, and a joint the 2D pipeline discarded was never lifted.`
        : `No lift ran: ${s.lift?.reason ?? "unrecorded"}.`,
      10,
    );
    for (let seat = 1; seat <= s.boat.seats; seat++) {
      const seatRows = angleRows.filter((r) => r.seat === seat);
      if (!seatRows.length) continue;
      const parts: string[] = [];
      const withheldParts: string[] = [];
      for (const n of ANGLE_NAMES) {
        const range = angleRange(seatRows, seat, n);
        if (range.lo.value === null) withheldParts.push(ANGLE_LABELS[n]);
        else
          parts.push(
            `${ANGLE_LABELS[n]} ${range.lo.value.toFixed(0)}-${range.hi.value!.toFixed(0)} deg (${Math.round(range.observedFraction * 100)}% of frames)`,
          );
      }
      text(
        `Seat ${seat}: ${parts.length ? parts.join(", ") : "nothing lifted"}.`,
        9,
      );
      if (withheldParts.length)
        text(
          `Seat ${seat} withheld: ${withheldParts.join(", ")}. ${
            angleRange(seatRows, seat, withheldParts[0] as AngleName).lo
              .reason ?? ""
          }`,
          9,
        );
    }
  }
  text("Withheld measurements", 16);
  text(
    angleRows.some((r) => r.angles.layback.value !== null)
      ? "Trunk twist, sequencing faults, oar angles, release quality, handle crossover/height, boat set and split per 500 m require additional validated evidence: oar and handle tracking, lateral measures a lift's depth-axis convention cannot confirm, and calibrated speed rather than a panning video."
      : "Normalized shin/body/joint angles, layback, trunk twist, sequencing faults, oar angles, release quality, handle crossover/height, boat set and split per 500 m require additional validated evidence.",
  );
  text("Camera guidance", 16);
  text(
    "Film alongside the shell, near perpendicular with a slight offset to separate rowers. Keep hands, hips, knees and the whole slide visible; on the water the feet sit below the gunwale and are not needed. For individual technique, use a closer single-seat view. Avoid bow-on footage for layback assessment.",
  );
  doc.save("row-analysis-report.pdf");
}
/**
 * Lifted angle rows, checked the way frames are: an imported session is a
 * record of a run, and the lift cannot be recomputed without its weights, so
 * what can be checked is that every row is either an angle with a confidence or
 * a refusal with a reason — never a number with no provenance behind it.
 */
function validAngles(s: Session): boolean {
  const rows = s.angles;
  if (rows === undefined) return !s.lift?.ran;
  if (!Array.isArray(rows) || rows.length !== s.frames.length) return false;
  if (!s.lift?.ran) return false;
  return rows.every(
    (r, i) =>
      r &&
      r.time === s.frames[i]!.time &&
      r.seat === s.frames[i]!.seat &&
      (r.reprojection === null ||
        (Number.isFinite(r.reprojection) && r.reprojection >= 0)) &&
      !!r.angles &&
      ANGLE_NAMES.every((n) => {
        const m = r.angles[n];
        return (
          !!m &&
          typeof m.unit === "string" &&
          Number.isFinite(m.confidence) &&
          m.confidence >= 0 &&
          m.confidence <= 1 &&
          (m.value === null
            ? m.confidence === 0 && typeof m.reason === "string"
            : Number.isFinite(m.value) && m.confidence > 0)
        );
      }),
  );
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
    // Negative is a real answer: it means the bow end, not the stern end, was
    // the nearer one. Only the size has a range.
    (s.camera.azimuth !== null &&
      (!Number.isFinite(s.camera.azimuth) ||
        Math.abs(s.camera.azimuth) > 90)) ||
    typeof s.camera.reason !== "string" ||
    !validAngles(s)
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
