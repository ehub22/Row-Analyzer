import { useMemo, useState } from "react";
import type { Session } from "../core/types";
import { framesAt } from "./Viewer";
import { biomechanics } from "../core/biomechanics";
import { bones } from "../core/keypoints";
import { rotateY } from "../core/geometry";
import { exportBiomechanicsCSV } from "../core/export";
export default function AdvancedPanel({
  session,
  seat,
  time,
}: {
  session: Session | null;
  seat: number;
  time: number;
}) {
  const [yaw, setYaw] = useState(30);
  const rows = useMemo(() => (session ? biomechanics(session) : []), [session]);
  if (!session)
    return (
      <section className="advanced-panel">
        <h2>Geometry, with its assumptions visible.</h2>
        <p>
          Analyze a video or load the synthetic demo to review kinematic 3D,
          viewpoint evidence and image-plane biomechanics.
        </p>
      </section>
    );
  const frame = framesAt(session, time).find((f) => f.seat === seat),
    lifted = frame?.lifted;
  const projected = lifted?.points.map((p) => (p ? rotateY(p, yaw) : null));
  const xy = (p: { x: number; y: number }) => ({
    x: 220 + p.x * 100,
    y: 220 - p.y * 100,
  });
  const count = projected?.filter(Boolean).length || 0;
  const stroke = session.strokes.find(
    (s) => s.seat === seat && time >= s.catch && time <= s.nextCatch,
  );
  const current = rows.filter(
    (r) => r.seat === seat && (stroke ? r.stroke === stroke.index : true),
  );
  return (
    <section className="advanced-panel">
      <div className="section-title">
        <div>
          <div className="eyebrow">EXPERIMENTAL · NOT CALIBRATED</div>
          <h2>Geometry, with its assumptions visible.</h2>
          <p>
            Seat {seat} · {time.toFixed(3)} s ·{" "}
            {session.demo ? "synthetic demonstration" : "recorded observations"}
          </p>
        </div>
        <button
          className="button"
          disabled={!rows.length}
          onClick={() => exportBiomechanicsCSV(session)}
        >
          Export biomechanics CSV
        </button>
      </div>
      <div className="advanced-grid">
        <article>
          <h3>Constrained 3D preview</h3>
          <p>Ambiguous depth · torso-length units · not a learned 3D model</p>
          <svg
            viewBox="0 0 440 380"
            role="img"
            aria-label="Experimental 3D skeleton"
            className="lifted-view"
          >
            <path
              d="M30 330H410 M220 25V355"
              stroke="#dae4dc"
              strokeDasharray="4 5"
            />
            {projected &&
              bones
                .filter(([a, b]) => a < 17 && b < 17)
                .map(([a, b]) => {
                  const pa = projected[a],
                    pb = projected[b];
                  if (!pa || !pb) return null;
                  const p = xy(pa),
                    q = xy(pb);
                  return (
                    <line
                      key={`${a}-${b}`}
                      x1={p.x}
                      y1={p.y}
                      x2={q.x}
                      y2={q.y}
                      stroke="#39796b"
                      strokeWidth="4"
                    />
                  );
                })}
            {projected?.map((p, i) =>
              p ? (
                <circle
                  key={i}
                  {...{ cx: xy(p).x, cy: xy(p).y }}
                  r="5"
                  fill="#d68a53"
                />
              ) : null,
            )}
            {count < 4 && (
              <text x="220" y="190" textAnchor="middle" fill="#51655b">
                Insufficient reliable joints for lifting
              </text>
            )}
          </svg>
          <label>
            Orbit preview (not camera calibration): {yaw}°
            <input
              aria-label="Orbit 3D preview"
              type="range"
              min="-180"
              max="180"
              value={yaw}
              onChange={(e) => setYaw(+e.target.value)}
            />
          </label>
          <small>
            {count} / 17 body joints shown. Bone-length priors choose one of
            multiple depth solutions; temporal continuity chooses a nearby
            branch. Missing observations and impossible projected bone lengths
            are withheld. No inferred 3D angle is used as a coaching
            measurement.
          </small>
        </article>
        <article>
          <h3>Automatic viewpoint estimate</h3>
          <div className="camera-reading">
            {session.camera.azimuth === null
              ? "Unresolved"
              : `${session.camera.azimuth.toFixed(0)}°`}
            {session.camera.uncertaintyDegrees && (
              <small>
                {" "}
                ± at least {session.camera.uncertaintyDegrees.toFixed(0)}°
              </small>
            )}
          </div>
          <p>
            {session.camera.source} · evidence score{" "}
            {Math.round(session.camera.confidence * 100)}% (not a probability)
          </p>
          <p>{session.camera.reason}</p>
          <h3>Oar-line tracking</h3>
          <p>
            {frame?.oars?.length
              ? `${frame.oars.length} candidate line(s) near observed wrists; ${frame.oars.filter((o) => o.age >= 3).length} supported across at least 3 consecutive frames.`
              : "No supported oar candidates at this frame. Enable on-water oar tracking before analysis if appropriate."}
          </p>
          <small>
            Dashed gold overlays mark image-supported line candidates. Water,
            riggers, erg rails and forearms can cause false positives. No blade
            identity, water contact, force, or release quality is claimed.
          </small>
        </article>
      </div>
      <h3>Observed image-plane kinematics</h3>
      <p>
        Projected angles depend on camera viewpoint. Wrist motion is not direct
        handle tracking. Timing is from body-motion stroke proxies.
        Inferred-only joints never unlock measurements.
      </p>
      {!current.length ? (
        <p className="note">
          No complete supported cycles for this seat. Missing evidence is not a
          zero measurement.
        </p>
      ) : (
        current.map((row) => (
          <div className="biomechanics-table" key={row.stroke}>
            <h4>
              Seat {row.seat} · stroke {row.stroke}
            </h4>
            <table>
              <thead>
                <tr>
                  <th>Measure</th>
                  <th>Value</th>
                  <th>Evidence / limitation</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(row.metrics).map(([name, m]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>
                      {m.value === null
                        ? "Withheld"
                        : `${m.value.toFixed(1)} ${m.unit}`}
                    </td>
                    <td>{m.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))
      )}
      <p className="note">
        True camera calibration, validated depth, force/power, trunk twist,
        blade entry/release, and split per 500 m require additional ground
        truth, camera geometry, sensors or trained models. They remain
        unavailable rather than being inferred from these previews.
      </p>
    </section>
  );
}
