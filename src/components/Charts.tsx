import type { Session } from "../core/types";
import { colors } from "../core/types";
import { crewTiming, mean } from "../core/strokes";
export function MotionChart({
  session,
  seat,
  onSeek,
}: {
  session: Session | null;
  seat: number;
  onSeek: (t: number) => void;
}) {
  const frames = session?.frames.filter((f) => f.seat === seat) || [];
  const values = frames.map((f) => {
    const p = f.points[9],
      hip = f.points[11];
    return {
      time: f.time,
      value:
        p.origin !== "missing" && hip.origin !== "missing" ? p.x - hip.x : null,
    };
  });
  const valid = values.filter((v) => v.value !== null);
  const min = Math.min(...valid.map((v) => v.value!), 0),
    max = Math.max(...valid.map((v) => v.value!), 1);
  let path = "",
    open = false;
  for (const v of values) {
    if (v.value === null) {
      open = false;
      continue;
    }
    path += `${open ? "L" : "M"}${30 + (v.time / (session?.duration || 1)) * 730},${140 - ((v.value - min) / (max - min)) * 110} `;
    open = true;
  }
  return (
    <div className="motion-chart">
      <div className="panel-heading">
        <div>
          <h3>Stroke rhythm</h3>
          <p>Wrist relative to hip · image-space pixels</p>
        </div>
        <span className="chart-key">
          <i /> Seat {seat}
        </span>
      </div>
      {!frames.length ? (
        <div className="chart-empty">
          <div className="faint-wave" />
          <span>Your stroke pattern will appear here after analysis</span>
        </div>
      ) : (
        <svg
          viewBox="0 0 800 180"
          role="img"
          aria-label="Wrist motion over time"
          onClick={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            onSeek(
              Math.max(
                0,
                Math.min(
                  session!.duration,
                  ((((e.clientX - r.left) / r.width) * 800 - 30) / 730) *
                    session!.duration,
                ),
              ),
            );
          }}
        >
          {[30, 65, 100, 140].map((y) => (
            <line key={y} x1="30" x2="760" y1={y} y2={y} stroke="#edf0eb" />
          ))}
          {session?.strokes
            .filter((s) => s.seat === seat)
            .map((s) => (
              <g key={s.index}>
                <rect
                  x={30 + (s.catch / session.duration) * 730}
                  y="20"
                  width={((s.finish - s.catch) / session.duration) * 730}
                  height="120"
                  fill="#eef4e7"
                />
                <line
                  x1={30 + (s.catch / session.duration) * 730}
                  x2={30 + (s.catch / session.duration) * 730}
                  y1="20"
                  y2="140"
                  stroke="#a6bf82"
                  strokeDasharray="3 4"
                />
              </g>
            ))}
          <path d={path} fill="none" stroke="#376f61" strokeWidth="2.5" />
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <text
              key={v}
              x={30 + v * 730}
              y="166"
              fill="#849089"
              fontSize="11"
              textAnchor="middle"
            >
              {((session?.duration || 0) * v).toFixed(1)} s
            </text>
          ))}
        </svg>
      )}
      <div className="chart-footer">
        <span>
          <i />
          Drive proxy
        </span>
        <span>Catch → drive → finish → recovery</span>
      </div>
    </div>
  );
}
export function CrewChart({ session }: { session: Session | null }) {
  const timing = session ? crewTiming(session.strokes, session.boat.seats) : [];
  const seats = session?.boat.seats || 8;
  return (
    <div className="crew-chart">
      <div className="panel-heading">
        <div>
          <h3>In time. Together.</h3>
          <p>Catch timing relative to stroke seat</p>
        </div>
        <span className="tag">MILLISECONDS</span>
      </div>
      <div className="sync-axis">
        <span>Early</span>
        <span>Stroke seat</span>
        <span>Late</span>
      </div>
      {Array.from({ length: seats }, (_, i) => {
        const seat = i + 1,
          rows = timing.filter((t) => t.seat === seat),
          lag =
            seat === seats && session ? 0 : mean(rows.map((t) => t.catchMs));
        return (
          <div className="sync-row" key={seat}>
            <span>
              <i style={{ background: colors[i] }} />
              {seat === 1 ? "Bow" : seat === seats ? "Stroke" : `Seat ${seat}`}
            </span>
            <div className="sync-track">
              <i className="zero" />
              {lag !== null && (
                <b
                  style={{
                    left: `${50 + Math.max(-45, Math.min(45, lag / 4))}%`,
                    background: colors[i],
                  }}
                />
              )}
            </div>
            <strong>
              {lag === null
                ? "—"
                : `${lag > 0 ? "+" : ""}${Math.round(lag)} ms`}
            </strong>
          </div>
        );
      })}
      <div className="note">
        {session
          ? "Cycle proxies, not blade-entry measurements. Confidence and timing detail are shown below."
          : "Analyze a crew clip to compare seat-to-seat timing."}
      </div>
    </div>
  );
}
