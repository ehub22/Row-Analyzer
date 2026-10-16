/**
 * Automatic horizon detection by gradient-weighted Hough voting.
 *
 * Only near-horizontal lines are considered (normal within ±maxTiltDeg of
 * vertical). Each edge pixel votes for the lines through it whose normal lies
 * within the edge's orientation, weighted by gradient magnitude. Votes from
 * several frames of a static camera can be summed before picking a peak, so
 * the horizon "locks" while people and oars, which move, do not accumulate.
 *
 * The result is an image line. Use `solveCameraNormalization` to turn it into
 * roll and pitch. Detection abstains (accepted = false) when the peak is weak
 * or not distinct from the next-best line.
 */
import { normalizeLine, type Line } from "./camera3d";

export interface GrayImage {
  data: Float32Array; // 0–1 luminance, row-major
  width: number;
  height: number;
}

/** Rec. 601 luminance from RGBA bytes (e.g. ImageData.data). */
export function grayFromRgba(
  rgba: ArrayLike<number>,
  width: number,
  height: number,
): GrayImage {
  const data = new Float32Array(width * height);
  for (let i = 0, p = 0; i < data.length; i++, p += 4)
    data[i] =
      (0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2]) / 255;
  return { data, width, height };
}

/** Box-downscale so the longest side is at most `maxSide` pixels. */
export function downscale(
  img: GrayImage,
  maxSide = 480,
): {
  image: GrayImage;
  factor: number;
} {
  const s = Math.max(img.width, img.height);
  const factor = s > maxSide ? s / maxSide : 1;
  if (factor === 1) return { image: img, factor: 1 };
  const w = Math.max(1, Math.floor(img.width / factor)),
    h = Math.max(1, Math.floor(img.height / factor));
  const data = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.floor(y * factor),
      y1 = Math.min(img.height, Math.max(y0 + 1, Math.floor((y + 1) * factor)));
    for (let x = 0; x < w; x++) {
      const x0 = Math.floor(x * factor),
        x1 = Math.min(
          img.width,
          Math.max(x0 + 1, Math.floor((x + 1) * factor)),
        );
      let sum = 0,
        n = 0;
      for (let yy = y0; yy < y1; yy++)
        for (let xx = x0; xx < x1; xx++) {
          sum += img.data[yy * img.width + xx];
          n++;
        }
      data[y * w + x] = sum / n;
    }
  }
  return { image: { data, width: w, height: h }, factor };
}

export interface EdgeSet {
  xs: Float32Array;
  ys: Float32Array;
  mags: Float32Array;
  /** Gradient direction, radians. */
  dirs: Float32Array;
  totalWeight: number;
  width: number;
  height: number;
  factor: number;
}

/** Sobel edges above the given magnitude percentile (default top 10%). */
export function extractEdges(
  img: GrayImage,
  factor: number,
  percentile = 0.9,
): EdgeSet {
  const { width: w, height: h, data } = img;
  const mag = new Float32Array(w * h),
    dir = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx =
        -data[i - w - 1] +
        data[i - w + 1] -
        2 * data[i - 1] +
        2 * data[i + 1] -
        data[i + w - 1] +
        data[i + w + 1];
      const gy =
        -data[i - w - 1] -
        2 * data[i - w] -
        data[i - w + 1] +
        data[i + w - 1] +
        2 * data[i + w] +
        data[i + w + 1];
      mag[i] = Math.hypot(gx, gy);
      dir[i] = Math.atan2(gy, gx);
    }
  const sorted = Float32Array.from(mag).sort();
  const threshold = Math.max(
    1e-4,
    sorted[Math.min(sorted.length - 1, Math.floor(percentile * sorted.length))],
  );
  const xs: number[] = [],
    ys: number[] = [],
    ms: number[] = [],
    ds: number[] = [];
  let totalWeight = 0;
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      if (mag[i] >= threshold) {
        xs.push(x);
        ys.push(y);
        ms.push(mag[i]);
        ds.push(dir[i]);
        totalWeight += mag[i];
      }
    }
  return {
    xs: Float32Array.from(xs),
    ys: Float32Array.from(ys),
    mags: Float32Array.from(ms),
    dirs: Float32Array.from(ds),
    totalWeight,
    width: w,
    height: h,
    factor,
  };
}

export interface HoughAccumulator {
  votes: Float64Array;
  nAngles: number;
  nRho: number;
  angles: Float64Array; // normal angle per row, radians
  rhoMax: number;
  width: number;
  height: number;
  factor: number;
  /** Total edge weight added, for normalizing support. */
  totalWeight: number;
}

/** Accumulator that also keeps the edges for refinement. */
export interface HoughState extends HoughAccumulator {
  edges?: EdgeSet;
}

const ANGLE_STEP_DEG = 0.25;

/** Accumulate edge votes for near-horizontal lines (summable across frames). */
export function houghHorizontal(
  edges: EdgeSet,
  maxTiltDeg = 25,
  acc?: HoughState,
): HoughState {
  const { width: w, height: h } = edges;
  const rhoMax = Math.ceil(Math.hypot(w, h)) + 1;
  if (!acc) {
    const nAngles = Math.round((2 * maxTiltDeg) / ANGLE_STEP_DEG) + 1;
    const angles = new Float64Array(nAngles);
    for (let k = 0; k < nAngles; k++)
      angles[k] = ((90 - maxTiltDeg + k * ANGLE_STEP_DEG) * Math.PI) / 180;
    acc = {
      votes: new Float64Array(nAngles * (2 * rhoMax + 1)),
      nAngles,
      nRho: 2 * rhoMax + 1,
      angles,
      rhoMax,
      width: w,
      height: h,
      factor: edges.factor,
      totalWeight: 0,
    };
  } else if (acc.width !== w || acc.height !== h)
    throw Error("Accumulator frame size mismatch.");
  const { nAngles, nRho, angles } = acc;
  const cosA = new Float64Array(nAngles),
    sinA = new Float64Array(nAngles);
  for (let k = 0; k < nAngles; k++) {
    cosA[k] = Math.cos(angles[k]);
    sinA[k] = Math.sin(angles[k]);
  }
  for (let e = 0; e < edges.xs.length; e++) {
    const x = edges.xs[e],
      y = edges.ys[e],
      m = edges.mags[e],
      g = edges.dirs[e];
    acc.totalWeight += m;
    for (let k = 0; k < nAngles; k++) {
      // Orientation consistency: a line's normal matches the gradient direction
      // up to sign, so weight by cos²(normal − gradient).
      const c = Math.cos(angles[k] - g);
      const wgt = m * c * c;
      if (wgt < 1e-6 * m) continue;
      const rho = x * cosA[k] + y * sinA[k];
      const idx = Math.round(rho) + rhoMax;
      acc.votes[k * nRho + idx] += wgt;
    }
  }
  return acc;
}

export interface HorizonDetection {
  /** Horizon in source-image pixels, unit-normalized. Null when no line found. */
  line: Line | null;
  /** Horizon tilt, degrees (0 = level, positive = right side lower). */
  tiltDeg: number;
  /** Horizon height at the image centre, pixels from the top. */
  yAtCenter: number;
  /** Fraction of total edge weight within 2 px of the line, 0–1. */
  support: number;
  /** Peak vote divided by the best vote from a clearly different line. */
  contrast: number;
  accepted: boolean;
  reason: string;
}

const MIN_SUPPORT = 0.04;
const MIN_CONTRAST = 1.5;

/**
 * Pick the strongest horizon from an accumulator, refine it by total least
 * squares on the supporting edge pixels, and decide whether to accept it.
 */
export function pickHorizon(acc: HoughState): HorizonDetection {
  const { nAngles, nRho, votes, rhoMax, factor, width } = acc;
  // 3×3 smoothing so a line split across two angle/offset bins still peaks.
  let best = -1,
    bestK = 0,
    bestR = 0;
  const at = (k: number, r: number) =>
    k < 0 || k >= nAngles || r < 0 || r >= nRho ? 0 : votes[k * nRho + r];
  const smoothed = new Float64Array(nAngles * nRho);
  for (let k = 0; k < nAngles; k++)
    for (let r = 0; r < nRho; r++) {
      const s =
        at(k - 1, r - 1) +
        at(k - 1, r) +
        at(k - 1, r + 1) +
        at(k, r - 1) +
        at(k, r) * 2 +
        at(k, r + 1) +
        at(k + 1, r - 1) +
        at(k + 1, r) +
        at(k + 1, r + 1);
      smoothed[k * nRho + r] = s / 10;
      if (s > best) {
        best = s;
        bestK = k;
        bestR = r;
      }
    }
  const empty: HorizonDetection = {
    line: null,
    tiltDeg: 0,
    yAtCenter: NaN,
    support: 0,
    contrast: 0,
    accepted: false,
    reason: "No edges voted for a near-horizontal line.",
  };
  if (best <= 0) return empty;

  // Second-best peak that is clearly a different line (≥ 3°, ≥ 12 px away).
  let second = 0;
  for (let k = 0; k < nAngles; k++)
    for (let r = 0; r < nRho; r++) {
      if (Math.abs(k - bestK) < 12 && Math.abs(r - bestR) < 12) continue;
      second = Math.max(second, smoothed[k * nRho + r]);
    }
  const peakMean = smoothed[bestK * nRho + bestR];
  const contrast = second > 0 ? peakMean / second : Infinity;

  const theta = acc.angles[bestK];
  const rho = bestR - rhoMax;
  let line = normalizeLine([Math.cos(theta), Math.sin(theta), -rho]);
  if (!line) return empty;

  // Refine: total least squares over edge pixels within 2 px of the line.
  const edges = acc.edges;
  let refined = line;
  let support = 0;
  if (edges) {
    const sw = { w: 0, x: 0, y: 0, xx: 0, yy: 0, xy: 0 };
    for (let e = 0; e < edges.xs.length; e++) {
      const d = Math.abs(
        refined[0] * edges.xs[e] + refined[1] * edges.ys[e] + refined[2],
      );
      if (d <= 2) {
        const m = edges.mags[e];
        sw.w += m;
        sw.x += m * edges.xs[e];
        sw.y += m * edges.ys[e];
        sw.xx += m * edges.xs[e] * edges.xs[e];
        sw.yy += m * edges.ys[e] * edges.ys[e];
        sw.xy += m * edges.xs[e] * edges.ys[e];
      }
    }
    if (sw.w > 0) {
      const mx = sw.x / sw.w,
        my = sw.y / sw.w;
      const cxx = sw.xx / sw.w - mx * mx,
        cyy = sw.yy / sw.w - my * my,
        cxy = sw.xy / sw.w - mx * my;
      // Normal = eigenvector of the smaller covariance eigenvalue.
      const tr = cxx + cyy,
        det = cxx * cyy - cxy * cxy,
        disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
      const lamSmall = tr / 2 - disc;
      let nx = cxy,
        ny = lamSmall - cxx;
      if (Math.abs(nx) + Math.abs(ny) < 1e-12) {
        nx = lamSmall - cyy;
        ny = cxy;
      }
      const cand = normalizeLine([nx, ny, -(nx * mx + ny * my)]);
      if (cand) refined = cand;
    }
    // Support is measured on the refined line.
    let inliers = 0;
    for (let e = 0; e < edges.xs.length; e++) {
      const d = Math.abs(
        refined[0] * edges.xs[e] + refined[1] * edges.ys[e] + refined[2],
      );
      if (d <= 2) inliers += edges.mags[e];
    }
    support = edges.totalWeight > 0 ? inliers / edges.totalWeight : 0;
  }

  // Convert back to source-image pixels: x_small = x_full / factor.
  const full: Line = normalizeLine([
    refined[0],
    refined[1],
    refined[2] * factor,
  ]) as Line;
  // A line has no direction: report the tilt in (-90°, 90°].
  let tiltDeg = (Math.atan2(-full[0], full[1]) * 180) / Math.PI;
  if (tiltDeg > 90) tiltDeg -= 180;
  else if (tiltDeg <= -90) tiltDeg += 180;
  const yAtCenter = -(full[0] * ((width * factor) / 2) + full[2]) / full[1];
  const accepted = support >= MIN_SUPPORT && contrast >= MIN_CONTRAST;
  let reason = accepted
    ? `Horizon locked: ${Math.round(support * 100)}% of edge weight on the line, contrast ${contrast.toFixed(1)}×.`
    : "Horizon not distinct enough to lock";
  if (!accepted && support < MIN_SUPPORT)
    reason += ` (support ${Math.round(support * 100)}%)`;
  else if (!accepted) reason += ` (contrast ${contrast.toFixed(2)}×)`;
  return {
    line: full,
    tiltDeg,
    yAtCenter,
    support,
    contrast,
    accepted,
    reason,
  };
}

/**
 * Detect the horizon in one frame. For several frames of a fixed camera, use
 * `lockHorizon`, which sums votes first.
 */
export function detectHorizon(
  img: GrayImage,
  options: { maxSide?: number; maxTiltDeg?: number } = {},
): HorizonDetection {
  const { image, factor } = downscale(img, options.maxSide ?? 480);
  const edges = extractEdges(image, factor);
  const acc = houghHorizontal(edges, options.maxTiltDeg ?? 25) as HoughState;
  acc.edges = edges;
  return pickHorizon(acc);
}

/**
 * Lock the horizon across frames of one static camera. Edge votes from every
 * frame are summed so a consistent horizon accumulates while moving
 * clutter does not. Frames must share a size.
 */
export function lockHorizon(
  frames: GrayImage[],
  options: { maxSide?: number; maxTiltDeg?: number } = {},
): HorizonDetection {
  if (!frames.length) throw Error("No frames to lock the horizon on.");
  let acc: HoughState | undefined;
  const allEdges: EdgeSet[] = [];
  for (const frame of frames) {
    const { image, factor } = downscale(frame, options.maxSide ?? 480);
    const edges = extractEdges(image, factor);
    allEdges.push(edges);
    acc = houghHorizontal(edges, options.maxTiltDeg ?? 25, acc) as HoughState;
  }
  // Refine on the pooled edges (all frames), so the least-squares fit uses
  // the same evidence the votes did.
  const pooled = pool(allEdges);
  acc!.edges = pooled;
  return pickHorizon(acc!);
}

function pool(sets: EdgeSet[]): EdgeSet {
  const n = sets.reduce((s, e) => s + e.xs.length, 0);
  const xs = new Float32Array(n),
    ys = new Float32Array(n),
    mags = new Float32Array(n),
    dirs = new Float32Array(n);
  let o = 0,
    total = 0;
  for (const e of sets) {
    xs.set(e.xs, o);
    ys.set(e.ys, o);
    mags.set(e.mags, o);
    dirs.set(e.dirs, o);
    o += e.xs.length;
    total += e.totalWeight;
  }
  return {
    xs,
    ys,
    mags,
    dirs,
    totalWeight: total,
    width: sets[0].width,
    height: sets[0].height,
    factor: sets[0].factor,
  };
}
