import { useEffect, useRef, useState } from "react";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Maximize2,
  ScanLine,
  Move,
  Upload,
  ArrowUpRight,
} from "lucide-react";
import type {
  Session,
  Correction,
  PoseFrame,
  LivePreview,
} from "../core/types";
import { colors } from "../core/types";
import { bones, jointNames } from "../core/keypoints";
export const clock = (s: number) =>
  `${Math.floor(s / 60)
    .toString()
    .padStart(2, "0")}:${Math.floor(s % 60)
    .toString()
    .padStart(2, "0")}.${Math.floor((s % 1) * 100)
    .toString()
    .padStart(2, "0")}`;
interface Props {
  analyzing?: boolean;
  livePreview?: LivePreview | null;
  url: string;
  session: Session | null;
  time: number;
  setTime: (t: number) => void;
  seat: number;
  crew: boolean;
  editing: boolean;
  joint: number;
  onCorrect: (c: Correction) => void;
  onUpload: () => void;
  onDemo: () => void;
  onDuration: (d: number, w: number, h: number) => void;
  duration: number;
  onError: (s: string) => void;
}
export function framesAt(session: Session, time: number) {
  let nearest = 0,
    distance = Infinity;
  for (const t of session.timestamps) {
    const d = Math.abs(t - time);
    if (d < distance) {
      nearest = t;
      distance = d;
    }
  }
  return session.frames.filter((f) => Math.abs(f.time - nearest) < 1e-5);
}
export default function Viewer(p: Props) {
  const video = useRef<HTMLVideoElement>(null),
    canvas = useRef<HTMLCanvasElement>(null),
    shell = useRef<HTMLDivElement>(null);
  const [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(0.5),
    [overlay, setOverlay] = useState(true),
    [drag, setDrag] = useState<{
      frame: PoseFrame;
      joint: number;
      x: number;
      y: number;
    } | null>(null);
  const [sourceSize, setSourceSize] = useState({ width: 1920, height: 1080 });
  const demo = p.session?.demo;
  const width = p.livePreview?.width || p.session?.width || sourceSize.width,
    height = p.livePreview?.height || p.session?.height || sourceSize.height;
  const duration = p.session?.duration || p.duration || 1;
  useEffect(() => {
    setPlaying(false);
  }, [p.url, p.session?.name]);
  useEffect(() => {
    if (p.analyzing) setPlaying(false);
  }, [p.analyzing]);
  useEffect(() => {
    if (video.current && !demo) {
      video.current.playbackRate = speed;
      if (playing) video.current.play().catch(() => setPlaying(false));
      else video.current.pause();
    }
  }, [playing, speed, demo]);
  useEffect(() => {
    const v = video.current;
    if (v && !demo && Math.abs(v.currentTime - p.time) > 0.12)
      v.currentTime = p.time;
  }, [p.time, demo]);
  useEffect(() => {
    if (!playing || !demo) return;
    let last = performance.now(),
      id = 0,
      t = p.time;
    function tick(now: number) {
      t += ((now - last) / 1000) * speed;
      last = now;
      if (t >= duration) {
        t = 0;
      }
      p.setTime(t);
      id = requestAnimationFrame(tick);
    }
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [playing, demo, speed, duration]);
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const ctx = c.getContext("2d")!;
    c.width = width;
    c.height = height;
    ctx.clearRect(0, 0, width, height);
    if (demo) {
      const gradient = ctx.createLinearGradient(0, 0, 0, height);
      gradient.addColorStop(0, "#b4cac8");
      gradient.addColorStop(0.4, "#90b3b1");
      gradient.addColorStop(1, "#608e8b");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "#c3d7d0";
      ctx.lineWidth = 1;
      for (let i = 0; i < 50; i++) {
        const y = 190 + i * 13,
          x = (i * 137) % width;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + 50 + ((i * 41) % 200), y);
        ctx.stroke();
      }
      ctx.fillStyle = "#68887f";
      ctx.beginPath();
      ctx.moveTo(0, 130);
      for (let x = 0; x <= width; x += 30)
        ctx.lineTo(x, 120 + Math.sin(x * 0.007) * 16 + Math.sin(x * 0.04) * 5);
      ctx.lineTo(width, 160);
      ctx.lineTo(0, 160);
      ctx.fill();
      ctx.fillStyle = "#c4d8cf";
      ctx.fillRect(0, 160, width, 2);
      ctx.fillStyle = "#efe8cf";
      ctx.beginPath();
      ctx.moveTo(50, 468);
      ctx.quadraticCurveTo(700, 515, 1390, 463);
      ctx.quadraticCurveTo(1280, 511, 140, 508);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = "#243f3d";
      ctx.lineWidth = 3;
      ctx.stroke();
    }
    if (p.livePreview) ctx.drawImage(p.livePreview.bitmap, 0, 0, width, height);
    if (!p.session && !p.livePreview) return;
    const frames = (
      p.livePreview?.frames ?? framesAt(p.session!, p.time)
    ).filter((f) => p.crew || f.seat === p.seat);
    for (const f of frames) {
      const color = colors[(f.seat - 1) % 8];
      if (demo) {
        const wrist = f.points[9];
        ctx.strokeStyle = "#294d49";
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.moveTo(wrist.x, wrist.y);
        ctx.lineTo(wrist.x + 100, 570);
        ctx.stroke();
        ctx.strokeStyle = "#eee2c9";
        ctx.lineWidth = 18;
        ctx.beginPath();
        ctx.moveTo(wrist.x + 98, 565);
        ctx.lineTo(wrist.x + 118, 606);
        ctx.stroke();
        ctx.lineCap = "round";
        ctx.strokeStyle = "#2d4949";
        ctx.lineWidth = 17;
        for (const [a, b] of bones.filter(
          ([a, b]) => a >= 5 && a < 17 && b < 17,
        )) {
          ctx.beginPath();
          ctx.moveTo(f.points[a].x, f.points[a].y);
          ctx.lineTo(f.points[b].x, f.points[b].y);
          ctx.stroke();
        }
        ctx.fillStyle = "#d2ae8b";
        ctx.beginPath();
        ctx.arc(f.points[0].x, f.points[0].y, 13, 0, Math.PI * 2);
        ctx.fill();
      }
      if (!overlay) continue;
      ctx.lineWidth = width / 600;
      for (const [a, b] of bones) {
        const start = f.points[a],
          end = f.points[b];
        if (start.origin === "missing" || end.origin === "missing") continue;
        ctx.strokeStyle = color;
        ctx.setLineDash(
          start.origin === "inferred" || end.origin === "inferred"
            ? [8, 7]
            : [],
        );
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      f.points.forEach((point, j) => {
        if (point.origin === "missing" || (j > 22 && j < 91)) return;
        const position =
          drag && drag.frame.seat === f.seat && drag.joint === j ? drag : point;
        ctx.fillStyle =
          point.origin === "manual"
            ? "#ffffff"
            : point.origin === "inferred"
              ? "#e5b461"
              : color;
        ctx.strokeStyle = "#173b37";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(
          position.x,
          position.y,
          ((p.editing && j === p.joint ? 7 : j < 23 ? 4 : 2) * width) / 1440,
          0,
          Math.PI * 2,
        );
        ctx.fill();
        ctx.stroke();
      });
      ctx.font = `600 ${width / 90}px sans-serif`;
      const label = `${f.seat}${f.seat === p.session?.boat.seats ? " · STROKE" : f.seat === 1 ? " · BOW" : ""}`;
      ctx.fillStyle = "#173b37";
      const bx = f.box.x + f.box.width / 2,
        by = f.box.y - 16;
      ctx.fillRect(bx - 12, by - 20, ctx.measureText(label).width + 22, 28);
      ctx.fillStyle = "#fafcf6";
      ctx.fillText(label, bx - 2, by);
    }
  }, [
    p.session,
    p.livePreview,
    p.time,
    p.seat,
    p.crew,
    p.editing,
    p.joint,
    overlay,
    drag,
    width,
    height,
    demo,
  ]);
  function step(dir: number) {
    setPlaying(false);
    const ts = p.session?.timestamps;
    const next = ts
      ? dir > 0
        ? ts.find((t) => t > p.time + 0.001)
        : [...ts].reverse().find((t) => t < p.time - 0.001)
      : p.time + dir / 30;
    p.setTime(Math.max(0, Math.min(duration, next ?? p.time)));
  }
  function pointer(e: React.PointerEvent<HTMLCanvasElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / r.width) * width,
      y: ((e.clientY - r.top) / r.height) * height,
    };
  }
  function start(e: React.PointerEvent<HTMLCanvasElement>) {
    if (p.analyzing || !p.editing || !p.session) return;
    setPlaying(false);
    const frame = framesAt(p.session, p.time).find((f) => f.seat === p.seat);
    if (!frame) return;
    const pos = pointer(e);
    const nearest = frame.points
      .map((v, j) => ({
        j,
        d:
          v.origin === "missing"
            ? Infinity
            : Math.hypot(pos.x - v.x, pos.y - v.y),
      }))
      .sort((a, b) => a.d - b.d)[0];
    const joint = nearest.d < width * 0.025 ? nearest.j : p.joint;
    e.currentTarget.setPointerCapture(e.pointerId);
    setDrag({ frame, joint, ...pos });
  }
  function end(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!drag) return;
    const pos = pointer(e);
    p.onCorrect({
      seat: drag.frame.seat,
      time: drag.frame.time,
      joint: drag.joint,
      before: drag.frame.points[drag.joint],
      after: { ...pos, confidence: 1, origin: "manual" },
    });
    setDrag(null);
  }
  return (
    <div className="viewer-card" ref={shell}>
      <div className="viewer-top">
        <span>
          <span className="live-dot" />{" "}
          {p.analyzing
            ? "Live inference preview"
            : p.session
              ? demo
                ? "Synthetic crew study"
                : "Source video"
              : p.url
                ? "Source video"
                : "Video workspace"}
        </span>
        <span className="viewer-source">
          {demo
            ? "ILLUSTRATIVE DATA"
            : p.url
              ? "LOCAL FILE · NEVER UPLOADED"
              : "ON-DEVICE BY DESIGN"}
        </span>
      </div>
      <div
        className={`video-stage ${p.editing ? "editing" : ""}`}
        style={{ aspectRatio: `${width}/${height}` }}
      >
        {p.url && !demo && (
          <video
            ref={video}
            src={p.url}
            playsInline
            muted
            onLoadedMetadata={(e) => {
              setSourceSize({
                width: e.currentTarget.videoWidth,
                height: e.currentTarget.videoHeight,
              });
              p.onDuration(
                e.currentTarget.duration,
                e.currentTarget.videoWidth,
                e.currentTarget.videoHeight,
              );
            }}
            onTimeUpdate={(e) => {
              if (playing) p.setTime(e.currentTarget.currentTime);
            }}
            onEnded={() => setPlaying(false)}
            onError={() =>
              p.onError(
                "Chrome could not play this video. Export H.264 MP4 locally and try again.",
              )
            }
          />
        )}
        {(p.session || p.url) && (
          <canvas
            ref={canvas}
            aria-label="Pose overlay; drag keypoints in correction mode"
            onPointerDown={start}
            onPointerMove={(e) => {
              if (drag) setDrag({ ...drag, ...pointer(e) });
            }}
            onPointerUp={end}
            onPointerCancel={() => setDrag(null)}
          />
        )}
        {!p.url && !p.session && (
          <div className="empty-stage">
            <svg
              viewBox="0 0 1000 440"
              className="water-art"
              aria-hidden="true"
            >
              <defs>
                <pattern
                  id="ripples"
                  width="170"
                  height="48"
                  patternUnits="userSpaceOnUse"
                >
                  <path
                    d="M0 28q30-8 70 0m25 16q24-6 60 0"
                    fill="none"
                    stroke="#91b6a6"
                    strokeOpacity=".14"
                  />
                </pattern>
              </defs>
              <rect width="1000" height="440" fill="url(#ripples)" />
              <g transform="translate(20 85) rotate(-9 500 200)" opacity=".5">
                <path
                  d="M100 220Q490 255 870 220Q700 280 160 245Z"
                  fill="#8faba0"
                />
                {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
                  <g key={i} transform={`translate(${205 + i * 75} 213)`}>
                    <path
                      d="M0 0l8-30 16 3 0 20 20 7"
                      fill="none"
                      stroke="#b6c5af"
                      strokeWidth="7"
                      strokeLinecap="round"
                    />
                    <circle cx="12" cy="-41" r="8" fill="#b6c5af" />
                    <path
                      d={i % 2 ? "M25 -12l45 -100" : "M25 -12l-20 100"}
                      stroke="#afc6b5"
                      strokeWidth="3"
                    />
                    <path
                      d={i % 2 ? "M70 -112l8 -20" : "M5 88l-6 22"}
                      stroke="#afc6b5"
                      strokeWidth="10"
                    />
                  </g>
                ))}
              </g>
            </svg>
            <div className="empty-message">
              <span className="upload-icon">
                <Upload size={25} />
              </span>
              <h2>
                Your next breakthrough
                <br />
                starts with a stroke.
              </h2>
              <p>Bring your footage. Find the details that matter.</p>
              <button className="button lime" onClick={p.onUpload}>
                <Upload size={16} /> Import rowing video
              </button>
              <button className="text-button light" onClick={p.onDemo}>
                Explore a synthetic demo <ArrowUpRight size={14} />
              </button>
            </div>
          </div>
        )}
        {p.analyzing && (
          <div className="stage-badges" role="status">
            <span>
              {p.livePreview
                ? `${p.livePreview.processed} FRAMES · ${clock(p.livePreview.time)}`
                : "WAITING FOR FIRST DETECTION"}
            </span>
            <span>LIVE · UNREVIEWED POSES</span>
          </div>
        )}
        {p.session && (
          <div className="stage-badges">
            <span>{p.crew ? "FULL CREW" : `SEAT ${p.seat}`}</span>
            <span className={demo ? "demo-badge" : ""}>
              {demo
                ? "SYNTHETIC DEMO · NOT REAL FOOTAGE"
                : "EXPERIMENTAL TRACKING"}
            </span>
          </div>
        )}
        {p.editing && p.session && (
          <div className="edit-hint">
            <Move size={13} /> Drag a visible joint, or click to place{" "}
            {jointNames[p.joint].toLowerCase()}
          </div>
        )}
      </div>
      <div className="timeline">
        <input
          aria-label="Video timeline"
          type="range"
          min="0"
          max={duration}
          step="0.001"
          value={Math.min(p.livePreview?.time ?? p.time, duration)}
          disabled={p.analyzing || (!p.session && !p.url)}
          onChange={(e) => {
            setPlaying(false);
            p.setTime(+e.target.value);
          }}
        />
        <div className="timeline-ticks">
          {[0, 0.25, 0.5, 0.75, 1].map((n) => (
            <span key={n}>{clock(duration * n).slice(0, 5)}</span>
          ))}
        </div>
      </div>
      <div className="playback">
        <div>
          <button
            title="Previous frame"
            aria-label="Previous frame"
            onClick={() => step(-1)}
            disabled={p.analyzing || (!p.url && !p.session)}
          >
            <SkipBack size={16} />
          </button>
          <button
            className="play-button"
            title="Play or pause"
            aria-label="Play or pause"
            disabled={p.analyzing || (!p.url && !p.session)}
            onClick={() => setPlaying(!playing)}
          >
            {playing ? <Pause size={16} /> : <Play size={16} />}
          </button>
          <button
            title="Next frame"
            aria-label="Next frame"
            onClick={() => step(1)}
            disabled={p.analyzing || (!p.url && !p.session)}
          >
            <SkipForward size={16} />
          </button>
          <span className="timecode">
            {clock(p.livePreview?.time ?? p.time)}{" "}
            <em>
              / {clock(duration === 1 && !p.session && !p.url ? 0 : duration)}
            </em>
          </span>
        </div>
        <div>
          <select
            aria-label="Playback speed"
            value={speed}
            onChange={(e) => setSpeed(+e.target.value)}
          >
            {[0.25, 0.5, 1].map((v) => (
              <option value={v} key={v}>
                {v}×
              </option>
            ))}
          </select>
          <button
            className={overlay ? "active-tool" : ""}
            aria-label="Toggle skeleton"
            title="Toggle skeleton overlay"
            onClick={() => setOverlay(!overlay)}
          >
            <ScanLine size={17} />
            <span>Skeleton</span>
          </button>
          <button
            title="Full screen"
            aria-label="Full screen"
            onClick={() => shell.current?.requestFullscreen()}
          >
            <Maximize2 size={16} />
          </button>
        </div>
      </div>
      <div className="viewer-legend">
        <span>
          <i className="observed" />
          Observed
        </span>
        <span>
          <i className="inferred" />
          Inferred
        </span>
        <span>
          <i className="manual" />
          Manually corrected
        </span>
        <small>
          {p.session
            ? "Exact decoded timestamps"
            : p.url
              ? "Frame step is approximate until analysis"
              : "Frame-by-frame review"}
        </small>
      </div>
    </div>
  );
}
