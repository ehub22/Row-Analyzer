import type { ModelsController } from "../hooks/useModels";
export default function ModelDownloads({
  models: m,
}: {
  models: ModelsController;
}) {
  return (
    <section className="model-downloads" aria-label="Model downloads">
      <strong>YOLOX-L + DWPose · 351 MB first download</strong>
      <p>
        Automatic, hash-verified download from this site. Reference weights are
        cached in this browser; video never leaves your device. Local imports
        work if downloads are blocked.
      </p>
      {m.progress && (
        <div role="status">
          <span>
            {m.progress.kind === "detector" ? "YOLOX-L" : "DWPose"} ·{" "}
            {m.progress.phase} · {(m.progress.loaded / 1e6).toFixed(1)} /{" "}
            {(m.progress.total / 1e6).toFixed(1)} MB
          </span>
          <progress
            aria-label="Model download progress"
            value={m.progress.loaded}
            max={m.progress.total}
          />
        </div>
      )}
      {m.message && <p role="status">{m.message}</p>}
      {m.error && (
        <p role="alert" className="model-error">
          {m.error}
        </p>
      )}
      <div className="model-actions">
        <button
          className="button"
          disabled={m.loading}
          onClick={() => void m.ensure().catch(() => {})}
        >
          {m.detector && m.pose
            ? "Check ready models"
            : "Download / restore models"}
        </button>
        {m.loading ? (
          <button className="button" onClick={m.cancel}>
            Cancel download
          </button>
        ) : (
          <button className="text-button" onClick={() => void m.clear()}>
            Clear model cache
          </button>
        )}
      </div>
      <small>
        Storage is best-effort and may be evicted. Custom exports are
        runtime-checked, not treated as verified reference weights.{" "}
        <a href="./licenses/DWPose.txt" target="_blank" rel="noreferrer">
          DWPose license
        </a>{" "}
        ·{" "}
        <a href="./licenses/YOLOX.txt" target="_blank" rel="noreferrer">
          YOLOX license
        </a>
      </small>
    </section>
  );
}
