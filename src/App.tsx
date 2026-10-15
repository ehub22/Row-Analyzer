import type { LivePreview } from "./core/types";
import { useRef, useState, useEffect } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowLeftRight,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Clock3,
  Cpu,
  Download,
  FileText,
  Film,
  Flag,
  FolderOpen,
  Layers,
  LockKeyhole,
  Move,
  Plus,
  Settings2,
  ShieldCheck,
  Sparkles,
  Upload,
  Users,
  Waves,
  X,
  AlertTriangle,
  RotateCcw,
  Focus,
  ExternalLink,
  Gauge,
} from "lucide-react";
import Viewer, { framesAt } from "./components/Viewer";
import { MotionChart, CrewChart } from "./components/Charts";
import type { Boat, Session, ModelSettings, Correction } from "./core/types";
import { boatClasses, defaultBoat, defaultModels, colors } from "./core/types";
import { createDemo } from "./core/demo";
import { mean, segmentStrokes, crewTiming } from "./core/strokes";
import { applyCorrection, HIDDEN_FEET_FLAG } from "./core/occlusion";
import {
  modelSpecs,
  downloadModel,
  getCachedModel,
  isModelCached,
  clearModelCache,
  ModelLoadError,
  type ModelId,
} from "./core/modelCache";
type ModelState = {
  state: "missing" | "checking" | "cached" | "downloading" | "ready" | "error";
  received: number;
  total: number;
  source?: "cache" | "download";
  error?: string;
};
const initialModelState = (): Record<ModelId, ModelState> => ({
  detector: { state: "checking", received: 0, total: 0 },
  pose: { state: "checking", received: 0, total: 0 },
});
const mb = (n: number) => `${(n / 1048576).toFixed(1)} MB`;
import {
  exportJSON,
  exportCSV,
  exportPDF,
  exportLabels,
  validateSession,
} from "./core/export";
import { jointNames } from "./core/keypoints";
import {
  planParallelism,
  readParallelEnv,
  summarizePlan,
} from "./core/parallel";
import { describeTta } from "./core/tta";
const tabs = ["Overview", "Crew timing", "Compare", "Keyframe labels"];
const metricNames = [
  "Catch shin angle",
  "Catch body angle",
  "Arm extension",
  "Catch oar angle",
  "Compression",
  "Drive sequencing",
  "Handle path / height",
  "Layback / finish angle",
  "Finish oar angle",
  "Release quality",
  "Recovery rush",
  "Head position",
  "Shoulder level",
  "Lateral lean / boat set",
  "Trunk twist (sweep)",
  "Handle crossover (scull)",
  "Left/right handle height",
  "Split per 500 m",
];
function App() {
  const [boat, setBoat] = useState<Boat>(structuredClone(defaultBoat)),
    [session, setSession] = useState<Session | null>(null),
    [compare, setCompare] = useState<Session | null>(null),
    [videoFile, setVideoFile] = useState<File | null>(null),
    [url, setUrl] = useState(""),
    [duration, setDuration] = useState(0),
    [time, setTime] = useState(0),
    [seat, setSeat] = useState(8),
    [crew, setCrew] = useState(true),
    [tab, setTab] = useState("Overview"),
    [modal, setModal] = useState<"models" | "guide" | null>(null),
    [toast, setToast] = useState(""),
    [error, setError] = useState(""),
    [exportOpen, setExportOpen] = useState(false),
    [settings, setSettings] = useState<ModelSettings>(defaultModels),
    [detector, setDetector] = useState<File | null>(null),
    [pose, setPose] = useState<File | null>(null),
    [modelState, setModelState] =
      useState<Record<ModelId, ModelState>>(initialModelState),
    [progress, setProgress] = useState<number | null>(null),
    [livePreview, setLivePreview] = useState<LivePreview | null>(null),
    [status, setStatus] = useState(""),
    [joint, setJoint] = useState(9),
    [baseline, setBaseline] = useState<Session | null>(null);
  const resolvedModels = useRef<Partial<Record<ModelId, File>>>({}),
    downloadAbort = useRef<AbortController | null>(null),
    runToken = useRef(0);
  const upload = useRef<HTMLInputElement>(null),
    importSession = useRef<HTMLInputElement>(null),
    importCompare = useRef<HTMLInputElement>(null),
    worker = useRef<Worker | null>(null),
    poseWorkers = useRef<Worker[]>([]);
  const previewBitmap = useRef<ImageBitmap | null>(null);
  /** Terminates the coordinator and every pose-pool worker. */
  function stopWorkers() {
    previewBitmap.current?.close();
    previewBitmap.current = null;
    setLivePreview(null);
    worker.current?.terminate();
    worker.current = null;
    for (const w of poseWorkers.current) w.terminate();
    poseWorkers.current = [];
  }
  const busy = progress !== null;
  const currentBoat = session?.boat || boat;
  const rows = session?.strokes.filter((s) => crew || s.seat === seat) || [];
  const rate = mean(rows.map((s) => s.rate.value)),
    ratio = mean(rows.map((s) => s.ratio.value)),
    confidence = mean(rows.map((s) => s.confidence));
  useEffect(() => () => stopWorkers(), []);
  useEffect(
    () => () => {
      if (url) URL.revokeObjectURL(url);
    },
    [url],
  );
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(""), 4500);
    return () => clearTimeout(t);
  }, [toast]);
  useEffect(() => {
    let alive = true;
    (async () => {
      for (const id of ["detector", "pose"] as const) {
        const spec = modelSpecs[id];
        let next: ModelState;
        if (await isModelCached(spec)) {
          const f = await getCachedModel(spec);
          next = f
            ? {
                state: "cached",
                received: f.size,
                total: f.size,
                source: "cache",
              }
            : { state: "missing", received: 0, total: spec.bytes };
        } else {
          next = { state: "missing", received: 0, total: spec.bytes };
        }
        if (alive) setModelState((m) => ({ ...m, [id]: next }));
      }
    })();
    return () => {
      alive = false;
    };
  }, []);
  useEffect(() => {
    if (!modal) return;
    const previous = document.activeElement as HTMLElement | null;
    const dialog = document.querySelector<HTMLElement>('[role="dialog"]');
    dialog?.querySelector<HTMLButtonElement>("button")?.focus();
    const keyboard = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setModal(null);
        return;
      }
      if (e.key !== "Tab" || !dialog) return;
      const nodes = [
        ...dialog.querySelectorAll<HTMLElement>(
          "button:not(:disabled), a[href], input, select",
        ),
      ].filter((n) => n.getClientRects().length);
      const first = nodes[0],
        last = nodes.at(-1);
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last?.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", keyboard);
    return () => {
      document.removeEventListener("keydown", keyboard);
      previous?.focus();
    };
  }, [modal]);

  function loadVideo(file?: File) {
    if (!file) return;
    if (file.size > 300 * 1024 * 1024) {
      setError("Please trim your video to 300 MB or smaller.");
      return;
    }
    stopWorkers();
    setProgress(null);
    setVideoFile(file);
    setUrl(URL.createObjectURL(file));
    if (session && !session.demo && session.name === file.name) {
      setToast(
        "Matching source attached. Existing analysis and corrections preserved.",
      );
      return;
    }
    setSession(null);
    setBaseline(null);
    setTime(0);
    setDuration(0);
    setError("");
    setToast(
      "Video loaded locally. Configure the boat and models, then analyze.",
    );
  }
  function demo() {
    stopWorkers();
    setProgress(null);
    const s = createDemo(boat);
    setSession(s);
    setBaseline(structuredClone(s));
    setSeat(boat.seats);
    setTime(0);
    setError("");
    setToast(
      "Synthetic demo loaded. These are generated poses, not real athlete data.",
    );
  }
  function configure(name: string) {
    const def = boatClasses[name];
    setBoat({
      ...boat,
      name,
      seats: def.seats,
      discipline: def.discipline,
      bestSeat: def.seats,
      rigging: Array.from({ length: def.seats }, (_, i) =>
        i % 2 ? "starboard" : "port",
      ),
    });
    setSeat(def.seats);
  }
  /** Local file (user choice) > Cache Storage > streamed download. */
  /** Local file (user choice) > Cache Storage > streamed download. */
  async function resolveModel(id: ModelId, token: number): Promise<File> {
    const local = id === "detector" ? detector : pose;
    if (local) return local;
    const known = resolvedModels.current[id];
    if (known) return known;
    return downloadOrCache(id, modelSpecs[id], token, true);
  }
  /**
   * Single place where model state changes: cache hit -> ready, otherwise a
   * streamed download with progress -> cached -> ready, or an error state.
   * `inRun` also drives the analysis progress bar and status line.
   */
  async function downloadOrCache(
    id: ModelId,
    spec: (typeof modelSpecs)[ModelId],
    token: number,
    inRun: boolean,
  ): Promise<File> {
    const setState = (st: ModelState) =>
      setModelState((m) => ({ ...m, [id]: st }));
    try {
      const cached = await getCachedModel(spec);
      if (cached) {
        resolvedModels.current[id] = cached;
        setState({
          state: "ready",
          received: cached.size,
          total: cached.size,
          source: "cache",
        });
        if (inRun) setStatus(`${spec.fileName} loaded from cache.`);
        return cached;
      }
      if (!spec.url)
        throw new ModelLoadError(
          `${spec.fileName} is not cached and this build has no download URL. Import it under Models & processing.`,
        );
      let lastPct = -1;
      setState({ state: "downloading", received: 0, total: spec.bytes });
      if (inRun) setStatus(`Downloading ${spec.fileName}…`);
      downloadAbort.current = new AbortController();
      const file = await downloadModel(
        spec,
        (received, total) => {
          const pct = total ? Math.floor((received / total) * 100) : 0;
          if (pct === lastPct) return;
          lastPct = pct;
          setState({ state: "downloading", received, total });
          if (inRun && token === runToken.current) {
            setStatus(
              `Downloading ${spec.fileName} · ${pct}% (${mb(received)} of ${mb(total)})`,
            );
            setProgress(pct / 1000); // download occupies the first 10% of the bar
          }
        },
        { signal: downloadAbort.current.signal },
      );
      if (token !== runToken.current && inRun)
        throw new ModelLoadError("Canceled.");
      resolvedModels.current[id] = file;
      setState({
        state: "ready",
        received: file.size,
        total: file.size,
        source: "download",
      });
      if (inRun) setStatus(`${spec.fileName} downloaded and cached.`);
      return file;
    } catch (e) {
      if (!(e instanceof ModelLoadError && e.message === "Canceled."))
        setState({
          state: "error",
          received: 0,
          total: spec.bytes,
          error: e instanceof Error ? e.message : String(e),
        });
      throw e;
    }
  }
  async function prepareModels(token: number) {
    try {
      const d = await resolveModel("detector", token);
      const p = await resolveModel("pose", token);
      return { d, p };
    } catch (e) {
      if (e instanceof ModelLoadError && e.message === "Canceled.") return null;
      if (token !== runToken.current) return null; // user canceled mid-download
      setError(
        `${e instanceof Error ? e.message : String(e)} Analysis did not start.`,
      );
      setProgress(null);
      setModal("models");
      return null;
    }
  }
  async function run() {
    if (!videoFile) {
      upload.current?.click();
      return;
    }
    if (
      settings.region.left >= settings.region.right ||
      settings.region.top >= settings.region.bottom
    ) {
      setError("Detection region must have left < right and top < bottom.");
      return;
    }
    setError("");
    setSession(null);
    setBaseline(null);
    setProgress(0);
    setStatus("Preparing models…");
    const token = ++runToken.current;
    const models = await prepareModels(token);
    if (!models || token !== runToken.current) return;
    setStatus("Starting private analysis…");
    stopWorkers();
    const w = new Worker(
      new URL("./workers/analysis.worker.ts", import.meta.url),
      { type: "module" },
    );
    worker.current = w;
    // Spawn the pose-worker pool on the main thread (nested workers are not
    // available in every browser) and connect each pose worker directly to
    // the coordinator with a MessageChannel, so tensors never relay through
    // this UI thread. The coordinator initialises the pool lazily — if
    // WebGPU works it is never used — and falls back to a local session if
    // pool startup fails for any reason.
    const plan = planParallelism(readParallelEnv(), settings.workers);
    try {
      for (let i = 0; i < plan.poseWorkers; i++) {
        const pw = new Worker(
          new URL("./workers/pose.worker.ts", import.meta.url),
          { type: "module" },
        );
        const channel = new MessageChannel();
        w.postMessage({ type: "pool-port" }, [channel.port1]);
        pw.postMessage({ type: "connect" }, [channel.port2]);
        pw.onerror = (e) => {
          try {
            w.postMessage({
              type: "pool-error",
              index: i,
              error: e.message || "pose worker error",
            });
          } catch {
            /* coordinator already gone */
          }
        };
        poseWorkers.current.push(pw);
      }
    } catch {
      // Worker construction itself failed (very old browser / CSP): the
      // coordinator runs pose inference on its local session instead.
    }
    w.onmessage = (e) => {
      const { type, payload } = e.data;
      if (worker.current !== w || token !== runToken.current) {
        if (type === "preview") payload.bitmap.close();
        return;
      }
      if (type === "preview") {
        previewBitmap.current?.close();
        previewBitmap.current = payload.bitmap;
        setLivePreview(payload);
      }
      if (type === "progress") setProgress(payload);
      if (type === "status") setStatus(payload);
      if (type === "result") {
        setSession(payload);
        setBaseline(structuredClone(payload));
        setSeat(boat.bestSeat);
        setProgress(null);
        setTime(0);
        setToast(
          "Analysis complete. Review seat identity and evidence before interpreting results.",
        );
        stopWorkers();
      }
      if (type === "error") {
        setError(payload);
        setProgress(null);
        stopWorkers();
      }
    };
    w.onerror = (e) => {
      setError(`Analysis worker failed: ${e.message}`);
      setProgress(null);
      stopWorkers();
    };
    w.postMessage({
      file: videoFile,
      detector: models.d,
      pose: models.p,
      settings,
      boat,
      parallel: plan,
    });
  }
  function correct(c: Correction) {
    if (!session) return;
    const frames = applyCorrection(session.frames, c);
    setSession({
      ...session,
      frames,
      strokes: segmentStrokes(frames),
      corrections: [...session.corrections, c],
    });
    setToast(
      `${jointNames[c.joint]} corrected for seat ${c.seat}. Nearby refits are labeled inferred.`,
    );
  }
  async function loadJSON(file: File | undefined, comparison = false) {
    if (!file) return;
    try {
      if (file.size > 100 * 1024 * 1024)
        throw Error("Session JSON is limited to 100 MB.");
      const s = validateSession(JSON.parse(await file.text()));
      if (comparison) {
        setCompare(s);
        setTab("Compare");
      } else {
        setSession(s);
        setBaseline(structuredClone(s));
        setBoat(s.boat);
        setSeat(s.boat.bestSeat);
        setTime(0);
        setUrl("");
        setVideoFile(null);
      }
      setToast(
        comparison
          ? "Comparison session loaded."
          : "Session loaded. Import the original video with the same filename to reattach footage.",
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }
  const activeFrames = session ? framesAt(session, time) : [];
  const selectedFrame = activeFrames.find((f) => f.seat === seat);
  const observed =
    selectedFrame?.points.filter(
      (p) => p.origin === "observed" || p.origin === "manual",
    ).length || 0;
  // Joints dropped because the session declares the feet hidden cannot be
  // observed, so they do not count against the frame.
  const observable =
    selectedFrame?.points.filter((p) => !p.flags?.includes(HIDDEN_FEET_FLAG))
      .length ?? 133;
  return (
    <div className="app-shell">
      <input
        className="hidden"
        ref={upload}
        type="file"
        accept="video/mp4,video/quicktime,.mov,.mp4"
        onChange={(e) => {
          loadVideo(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      <input
        className="hidden"
        ref={importSession}
        type="file"
        accept="application/json,.json"
        onChange={(e) => {
          loadJSON(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      <input
        className="hidden"
        ref={importCompare}
        type="file"
        accept="application/json,.json"
        onChange={(e) => {
          loadJSON(e.target.files?.[0], true);
          e.target.value = "";
        }}
      />
      <aside className="sidebar">
        <a className="brand" href="./" aria-label="Row Analyzer home">
          <span className="brand-mark">
            <Waves size={23} />
          </span>
          <span>
            row<span className="brand-light">analyzer</span>
            <small>SEE THE DIFFERENCE.</small>
          </span>
        </a>
        <div className="workspace-label">YOUR WORKSPACE</div>
        <button
          className="nav-item selected"
          onClick={() => setTab("Overview")}
        >
          <Activity size={18} /> Analysis studio <span>01</span>
        </button>
        <button className="nav-item" onClick={() => setTab("Compare")}>
          <ArrowLeftRight size={18} /> Compare sessions
        </button>
        <button
          className="nav-item"
          onClick={() => importSession.current?.click()}
        >
          <FolderOpen size={18} /> Open saved session
        </button>
        <div className="nav-rule" />
        <div className="workspace-label">THIS SESSION</div>
        <div className="session-nav">
          <span className="session-dot" />
          <div>
            {session?.demo
              ? "Demo crew study"
              : videoFile?.name || session?.name || "Untitled session"}
            <small>
              {session ? "Review in progress" : "Ready for your footage"}
            </small>
          </div>
        </div>
        <button
          className="new-session"
          onClick={() => {
            stopWorkers();
            setProgress(null);
            setSession(null);
            setBaseline(null);
            setVideoFile(null);
            setUrl("");
            setDuration(0);
            setTime(0);
            setCompare(null);
            setToast(
              "New session ready. Export results before leaving a session to keep them.",
            );
          }}
        >
          <Plus size={15} /> New session
        </button>
        <div className="sidebar-bottom">
          <div className="coach-note">
            <span>
              <Sparkles size={15} /> A better point of view
            </span>
            <p>
              Good footage makes
              <br />
              great feedback possible.
            </p>
            <button onClick={() => setModal("guide")}>
              Camera setup guide <ArrowUpRight size={14} />
            </button>
          </div>
          <button className="nav-item" onClick={() => setModal("models")}>
            <Settings2 size={17} /> Models & settings
          </button>
          <button className="nav-item" onClick={() => setModal("guide")}>
            <CircleHelp size={17} /> Help & methodology
          </button>
          <div className="private-status">
            <span className="live-dot" /> Local-first. Athlete-first.
            <LockKeyhole size={12} />
          </div>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div>
            Workspace <ChevronRight size={13} />{" "}
            <strong>Analysis studio</strong>
          </div>
          <div>
            <span className="privacy-pill">
              <ShieldCheck size={14} /> Your footage stays yours
            </span>
            <span className="avatar">RA</span>
          </div>
        </header>
        <div className="main-content">
          <section className="page-heading">
            <div>
              <div className="eyebrow">THE ROWING TECHNIQUE WORKBENCH</div>
              <h1>
                Every stroke, understood<span>.</span>
              </h1>
              <p>A closer look at your technique. A clearer way forward.</p>
            </div>
            <div className="heading-actions">
              <button
                className="button"
                onClick={() => upload.current?.click()}
                disabled={busy}
              >
                <Upload size={16} /> Import video
              </button>
              <div className="export-wrap">
                <button
                  className="button dark"
                  disabled={!session}
                  onClick={() => setExportOpen(!exportOpen)}
                >
                  <ArrowDownToLine size={16} /> Export <ChevronDown size={13} />
                </button>
                {exportOpen && session && (
                  <div className="export-menu">
                    {[
                      ["PDF coaching report", () => exportPDF(session)],
                      ["CSV stroke table", () => exportCSV(session)],
                      ["JSON session", () => exportJSON(session)],
                      ["Training corrections", () => exportLabels(session)],
                    ].map(([label, fn]) => (
                      <button
                        key={label as string}
                        onClick={() => {
                          (fn as () => void)();
                          setExportOpen(false);
                        }}
                      >
                        <FileText size={15} />
                        {label as string}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </section>
          <div className="workspace-toolbar">
            <nav className="tabs">
              {tabs.map((t) => (
                <button
                  key={t}
                  className={tab === t ? "active" : ""}
                  onClick={() => setTab(t)}
                >
                  {t === "Overview" ? (
                    <Layers size={15} />
                  ) : t === "Crew timing" ? (
                    <Users size={15} />
                  ) : t === "Compare" ? (
                    <ArrowLeftRight size={15} />
                  ) : (
                    <Move size={15} />
                  )}{" "}
                  {t}
                  {t === "Keyframe labels" && !!session?.corrections.length && (
                    <b>{session.corrections.length}</b>
                  )}
                </button>
              ))}
            </nav>
            <span className="version-tag">
              RESEARCH PREVIEW <span>v0.1</span>
            </span>
          </div>
          {error && (
            <div className="alert error" role="alert">
              <AlertTriangle size={18} />
              <div>
                <strong>We couldn’t complete that step.</strong>
                <p>{error}</p>
              </div>
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          {session?.demo && (
            <div className="demo-notice">
              <Sparkles size={15} />
              <span>
                <strong>Synthetic demonstration.</strong> Explore the tools with
                generated stroke data. This is not a tracking accuracy
                benchmark.
              </span>
              <button onClick={() => upload.current?.click()}>
                Use my footage <ArrowRight size={14} />
              </button>
            </div>
          )}
          {session && !session.demo && (
            <details className="analysis-notes">
              <summary>
                <AlertTriangle size={15} /> Analysis notes · experimental
                evidence <ChevronDown size={14} />
              </summary>
              {session.performance && (
                <p className="note performance-line">
                  <Gauge size={14} /> {session.performance.summary}
                </p>
              )}
              <ul>
                {session.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </details>
          )}
          {tab === "Compare" ? (
            <section className="compare-page">
              <div className="section-title">
                <div>
                  <h2>A different session. A new perspective.</h2>
                  <p>Compare observed timing, not uncalibrated body angles.</p>
                </div>
                <button
                  className="button"
                  onClick={() => importCompare.current?.click()}
                >
                  <Plus size={16} /> Add comparison
                </button>
              </div>
              <div className="comparison-grid">
                {[session, compare].map((s, i) => (
                  <div className="comparison-card" key={i}>
                    <div className="eyebrow">
                      {i ? "COMPARISON SESSION" : "CURRENT SESSION"}
                    </div>
                    <h2>{s?.name || "A fresh point of comparison"}</h2>
                    <p>
                      {s
                        ? `${s.boat.name} · ${s.boat.discipline} · ${s.demo ? "Synthetic" : "Experimental measurements"}`
                        : "Import an exported Row Analyzer JSON session."}
                    </p>
                    <div className="comparison-stat">
                      <span>
                        {mean(
                          s?.strokes.map((v) => v.rate.value) || [],
                        )?.toFixed(1) || "—"}
                      </span>{" "}
                      strokes per minute
                    </div>
                    <MotionChart
                      session={s}
                      seat={s?.boat.bestSeat || 1}
                      onSeek={() => {}}
                    />
                    {!s && (
                      <button
                        className="button"
                        onClick={() =>
                          i
                            ? importCompare.current?.click()
                            : upload.current?.click()
                        }
                      >
                        {i ? "Open comparison JSON" : "Import source video"}
                      </button>
                    )}
                  </div>
                ))}
              </div>
              {session && compare && (
                <div className="alert">
                  <ArrowLeftRight size={20} />
                  <div>
                    <strong>
                      {session.demo !== compare.demo
                        ? "Not comparable: synthetic vs. recorded data"
                        : session.boat.discipline !== compare.boat.discipline
                          ? "Different disciplines: do not compare technique metrics directly"
                          : "Review matched conditions before drawing conclusions"}
                    </strong>
                    <p>
                      Rate difference:{" "}
                      {session.demo === compare.demo &&
                      session.boat.discipline === compare.boat.discipline &&
                      mean(session.strokes.map((s) => s.rate.value)) !== null &&
                      mean(compare.strokes.map((s) => s.rate.value)) !== null
                        ? `${(mean(session.strokes.map((s) => s.rate.value))! - mean(compare.strokes.map((s) => s.rate.value))!).toFixed(1)} spm (current − comparison)`
                        : "withheld"}
                      . Different camera angles and evidence coverage can change
                      timing quality. Phase-normalized 3D/model-stroke
                      comparison is not available.
                    </p>
                  </div>
                </div>
              )}
            </section>
          ) : (
            <>
              <div className="analysis-grid">
                <section className="analysis-main">
                  <div className="section-heading">
                    <div className="mode-toggle">
                      <button
                        className={crew ? "active" : ""}
                        onClick={() => setCrew(true)}
                      >
                        <Users size={14} /> Full crew
                      </button>
                      <button
                        className={!crew ? "active" : ""}
                        onClick={() => setCrew(false)}
                      >
                        <Focus size={14} /> Single seat
                      </button>
                    </div>
                    <span className="file-name">
                      <Film size={13} />
                      {session?.demo
                        ? "Synthetic side view"
                        : videoFile?.name || "No video selected"}
                    </span>
                  </div>
                  <Viewer
                    analyzing={busy}
                    livePreview={livePreview}
                    url={url}
                    session={session}
                    time={time}
                    setTime={setTime}
                    seat={seat}
                    crew={tab === "Keyframe labels" ? false : crew}
                    editing={tab === "Keyframe labels"}
                    joint={joint}
                    onCorrect={correct}
                    onUpload={() => upload.current?.click()}
                    onDemo={demo}
                    onDuration={(d) => setDuration(d)}
                    duration={duration}
                    onError={setError}
                  />
                  <div className="seat-strip">
                    <span>SEAT SELECTOR</span>
                    <div>
                      {Array.from({ length: currentBoat.seats }, (_, i) => (
                        <button
                          key={i}
                          className={seat === i + 1 ? "active" : ""}
                          onClick={() => {
                            setSeat(i + 1);
                            setCrew(false);
                          }}
                          title={`Seat ${i + 1}${i === 0 ? " · bow seat" : i === currentBoat.seats - 1 ? " · stroke seat" : ""}`}
                        >
                          <i style={{ background: colors[i % 8] }} />
                          {i + 1}
                          <small>
                            {i === 0
                              ? "BOW"
                              : i === currentBoat.seats - 1
                                ? "STROKE"
                                : ""}
                          </small>
                        </button>
                      ))}
                    </div>
                    <span className="seat-best">
                      <Flag size={12} /> Seat {currentBoat.bestSeat} best
                      visible
                    </span>
                  </div>
                </section>
                <aside className="setup-panel">
                  <div className="panel-heading">
                    <h3>Session setup</h3>
                    <Settings2 size={16} />
                  </div>
                  <div className="setup-step">
                    <span>01</span>
                    <strong>Choose your shell</strong>
                  </div>
                  <label>
                    Boat class
                    <select
                      disabled={busy || !!session}
                      value={currentBoat.name}
                      onChange={(e) => configure(e.target.value)}
                    >
                      {Object.keys(boatClasses).map((n) => (
                        <option key={n}>{n}</option>
                      ))}
                    </select>
                  </label>
                  {boat.name === "Custom" && !session && (
                    <label>
                      Seat count
                      <input
                        type="number"
                        min="1"
                        max="8"
                        value={boat.seats}
                        onChange={(e) => {
                          const seats = Math.max(
                            1,
                            Math.min(8, +e.target.value),
                          );
                          setBoat({
                            ...boat,
                            seats,
                            bestSeat: seats,
                            rigging: Array.from({ length: seats }, (_, i) =>
                              i % 2 ? "starboard" : "port",
                            ),
                          });
                          setSeat(seats);
                        }}
                      />
                    </label>
                  )}
                  <div className="discipline">
                    <button
                      disabled={!!session || busy || boat.name !== "Custom"}
                      className={
                        currentBoat.discipline === "sweep" ? "active" : ""
                      }
                      onClick={() => setBoat({ ...boat, discipline: "sweep" })}
                    >
                      Sweep <span>1 oar</span>
                    </button>
                    <button
                      disabled={!!session || busy || boat.name !== "Custom"}
                      className={
                        currentBoat.discipline === "scull" ? "active" : ""
                      }
                      onClick={() => setBoat({ ...boat, discipline: "scull" })}
                    >
                      Scull <span>2 oars</span>
                    </button>
                  </div>
                  <div className="setup-step">
                    <span>02</span>
                    <strong>Set your seats</strong>
                  </div>
                  <div className="two-fields">
                    <label>
                      Best-visible seat
                      <select
                        disabled={busy || !!session}
                        value={currentBoat.bestSeat}
                        onChange={(e) =>
                          setBoat({ ...boat, bestSeat: +e.target.value })
                        }
                      >
                        {Array.from({ length: currentBoat.seats }, (_, i) => (
                          <option key={i} value={i + 1}>
                            Seat {i + 1}
                            {i + 1 === currentBoat.seats ? " · stroke" : ""}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Bow in image
                      <select
                        disabled={busy || !!session}
                        value={currentBoat.bowLeft ? "left" : "right"}
                        onChange={(e) =>
                          setBoat({
                            ...boat,
                            bowLeft: e.target.value === "left",
                          })
                        }
                      >
                        <option value="left">On the left</option>
                        <option value="right">On the right</option>
                      </select>
                    </label>
                  </div>
                  <label className="feet-field">
                    Feet in view
                    <select
                      disabled={busy || !!session}
                      value={
                        currentBoat.feetHidden === false ? "visible" : "hidden"
                      }
                      onChange={(e) =>
                        setBoat({
                          ...boat,
                          feetHidden: e.target.value === "hidden",
                        })
                      }
                    >
                      <option value="hidden">
                        Hidden by the hull (on-water)
                      </option>
                      <option value="visible">Visible (erg or overhead)</option>
                    </select>
                    <small>
                      {currentBoat.feetHidden === false
                        ? "Ankle and foot keypoints are tracked like any other joint."
                        : "Feet on the stretcher sit below the gunwale, so ankle and foot keypoints would only be model guesses. They are not estimated; stroke timing uses hips and knees."}
                    </small>
                  </label>
                  {currentBoat.discipline === "sweep" && (
                    <details className="rigging">
                      <summary>
                        Port / starboard rigging <ChevronDown size={12} />
                      </summary>
                      <div>
                        {currentBoat.rigging
                          .slice(0, currentBoat.seats)
                          .map((r, i) => (
                            <label key={i}>
                              Seat {i + 1}
                              <select
                                disabled={busy || !!session}
                                value={r}
                                onChange={(e) =>
                                  setBoat({
                                    ...boat,
                                    rigging: boat.rigging.map((v, j) =>
                                      i === j
                                        ? (e.target.value as
                                            "port" | "starboard")
                                        : v,
                                    ),
                                  })
                                }
                              >
                                <option value="port">Port</option>
                                <option value="starboard">Starboard</option>
                              </select>
                            </label>
                          ))}
                      </div>
                    </details>
                  )}
                  <div className="setup-step">
                    <span>03</span>
                    <strong>Accuracy over speed</strong>
                  </div>
                  <button
                    className="model-summary"
                    onClick={() => setModal("models")}
                    disabled={busy}
                  >
                    <span className="model-icon">
                      <Cpu size={18} />
                    </span>
                    <span>
                      <strong>
                        {(detector ||
                          modelState.detector.state === "cached" ||
                          modelState.detector.state === "ready") &&
                        (pose ||
                          modelState.pose.state === "cached" ||
                          modelState.pose.state === "ready")
                          ? "Whole-body models ready"
                          : "Models download on first analysis"}
                      </strong>
                      <small>
                        {`133 joints · ${describeTta(settings.tta)}`}
                      </small>
                    </span>
                    <ChevronRight size={15} />
                  </button>
                  <div className="camera-status">
                    <div>
                      <Focus size={15} />
                      <strong>Camera azimuth</strong>
                      <span>
                        {session?.camera.source === "synthetic"
                          ? "0° · demo"
                          : "Unresolved"}
                      </span>
                    </div>
                    <p>
                      {session?.camera.source === "synthetic"
                        ? "Known generated view · 100% synthetic certainty."
                        : `Confidence ${Math.round((session?.camera.confidence || 0) * 100)}%. 3D angles stay withheld until validated calibration is available.`}
                    </p>
                  </div>
                  {busy ? (
                    <div className="progress-block">
                      <div>
                        <strong>Analyzing every frame</strong>
                        <span>{Math.round(progress! * 100)}%</span>
                      </div>
                      <progress value={progress!} max="1" />
                      <p>{status}</p>
                      <button
                        className="text-button"
                        onClick={() => {
                          runToken.current++;
                          downloadAbort.current?.abort();
                          stopWorkers();
                          setProgress(null);
                          setStatus("");
                          setToast(
                            "Analysis canceled. No partial metrics reported.",
                          );
                        }}
                      >
                        Cancel analysis
                      </button>
                    </div>
                  ) : (
                    <button className="button analyze" onClick={run}>
                      <Sparkles size={16} />
                      {videoFile ? "Analyze video" : "Import & analyze"}
                      <ArrowRight size={16} />
                    </button>
                  )}
                  <p className="setup-footnote">
                    <LockKeyhole size={11} /> Processed on your device. Never
                    uploaded.
                  </p>
                  {session && !session.demo && (
                    <p className="runtime-line">
                      <Cpu size={11} /> Runtime: {session.provider}
                    </p>
                  )}
                </aside>
              </div>
              {tab === "Keyframe labels" ? (
                <section className="labels-panel">
                  <div className="section-title">
                    <div>
                      <h2>Make the invisible explicit.</h2>
                      <p>
                        Correct only joints you can see. Neighboring refits are
                        never training ground truth.
                      </p>
                    </div>
                    <button
                      className="button"
                      disabled={!session?.corrections.length}
                      onClick={() => session && exportLabels(session)}
                    >
                      <Download size={15} /> Export labels
                    </button>
                  </div>
                  <div className="label-controls">
                    <label>
                      Joint to place
                      <select
                        value={joint}
                        onChange={(e) => setJoint(+e.target.value)}
                      >
                        {jointNames.map((name, i) => (
                          <option value={i} key={i}>
                            {name}
                          </option>
                        ))}
                      </select>
                    </label>
                    <div>
                      <strong>
                        {observed} / {observable}
                      </strong>
                      <span>
                        {" "}
                        observed or manually corrected
                        {observable < 133 ? " · feet hidden by hull" : ""}
                      </span>
                    </div>
                    <button
                      className="button"
                      disabled={!session?.corrections.length}
                      onClick={() => {
                        if (baseline) setSession(structuredClone(baseline));
                      }}
                    >
                      <RotateCcw size={14} /> Reset corrections
                    </button>
                  </div>
                  <p className="note">
                    Select a seat and frame, then drag a keypoint. To add a
                    missing point, choose the joint above and click its visible
                    location. Corrections refit ±0.15 sec, preserving other
                    manual anchors. Face points remain in exports but are hidden
                    in the overlay.
                  </p>
                </section>
              ) : (
                <>
                  <div className="metrics-row">
                    <MetricCard
                      label="STROKE RATE"
                      value={rate?.toFixed(1) || "—"}
                      unit="spm"
                      sub={
                        rate
                          ? "Average of complete cycles"
                          : "Waiting for observed cycles"
                      }
                      confidence={confidence}
                      icon={<Activity size={17} />}
                    />
                    <MetricCard
                      label="DRIVE : RECOVERY"
                      value={ratio ? `${ratio.toFixed(2)} : 1` : "—"}
                      unit=""
                      sub="Timing from body-motion proxies"
                      confidence={confidence}
                      icon={<Clock3 size={17} />}
                    />
                    <MetricCard
                      label="COMPLETE CYCLES"
                      value={session ? `${rows.length}` : "—"}
                      unit={crew ? "seat-cycles" : "strokes"}
                      sub={
                        crew
                          ? "Total across all selected seats"
                          : "For the selected seat"
                      }
                      confidence={confidence}
                      icon={<Waves size={17} />}
                    />
                    <MetricCard
                      label="EVIDENCE COVERAGE"
                      value={
                        session
                          ? `${Math.round((mean(rows.map((s) => s.observedFraction)) || 0) * 100)}`
                          : "—"
                      }
                      unit="%"
                      sub="Observed joints in timing metrics"
                      confidence={confidence}
                      icon={<ShieldCheck size={17} />}
                    />
                  </div>
                  <div className="charts-grid">
                    {tab === "Crew timing" ? (
                      <CrewChart session={session} />
                    ) : (
                      <MotionChart
                        session={session}
                        seat={seat}
                        onSeek={setTime}
                      />
                    )}
                    <div className="insight-panel">
                      <div className="panel-heading">
                        <h3>
                          <Sparkles size={16} /> Coach’s notebook
                        </h3>
                        <span className="tag">EVIDENCE FIRST</span>
                      </div>
                      {session ? (
                        <>
                          <h4>
                            {session.demo
                              ? "A glimpse of the bigger picture."
                              : rows.length
                                ? "Start with the rhythm."
                                : "More evidence, better feedback."}
                          </h4>
                          <p>
                            {rows.length
                              ? `The ${crew ? "crew" : "selected seat"} has ${rows.length} complete seat-cycles available for review. ${rate ? `Average stroke rate is ${rate.toFixed(1)} strokes per minute.` : "Stroke rate is withheld because observed evidence is insufficient."}`
                              : "No complete, sufficiently visible cycles were found. Review hip and knee tracking, or use a longer, closer side-view clip."}
                          </p>
                          <div className="insight-tip">
                            <Focus size={18} />
                            <span>
                              <strong>Angles need context.</strong> Layback,
                              compression, trunk twist, and oar angles are
                              withheld without reliable 3D calibration.
                            </span>
                          </div>
                        </>
                      ) : (
                        <>
                          <h4>
                            Good coaching starts
                            <br />
                            with a clear view.
                          </h4>
                          <p>
                            Import a clip to explore your stroke rhythm, review
                            each seat, and see where the crew comes together.
                          </p>
                          <button
                            className="text-button"
                            onClick={() => setModal("guide")}
                          >
                            Get your camera position right{" "}
                            <ArrowUpRight size={14} />
                          </button>
                        </>
                      )}
                      <div className="insight-footer">
                        <ShieldCheck size={13} /> Confidence is shown.
                        Uncertainty is never hidden.
                      </div>
                    </div>
                  </div>
                  {session && (
                    <section className="results-panel">
                      <div className="section-title">
                        <div>
                          <h2>
                            {tab === "Crew timing"
                              ? "Seat-to-seat timing"
                              : "The details, stroke by stroke"}
                          </h2>
                          <p>
                            {tab === "Crew timing"
                              ? "Positive values mean later than stroke seat. Timing resolution is bounded by decoded frames."
                              : "Cycle events are body-motion proxies. Check the video before interpreting them as catch or release."}
                          </p>
                        </div>
                        <button
                          className="button"
                          onClick={() => exportCSV(session)}
                        >
                          <Download size={14} /> CSV
                        </button>
                      </div>
                      <div className="table-scroll">
                        {tab === "Crew timing" ? (
                          <table>
                            <thead>
                              <tr>
                                <th>Seat</th>
                                <th>Stroke</th>
                                <th>Catch offset</th>
                                <th>Finish offset</th>
                                <th>Phase offset</th>
                                <th>Confidence</th>
                              </tr>
                            </thead>
                            <tbody>
                              {crewTiming(
                                session.strokes,
                                currentBoat.seats,
                              ).map((t) => (
                                <tr key={`${t.seat}-${t.index}`}>
                                  <td>Seat {t.seat}</td>
                                  <td>{t.index}</td>
                                  <td>{t.catchMs.toFixed(0)} ms</td>
                                  <td>{t.finishMs.toFixed(0)} ms</td>
                                  <td>{t.phase.toFixed(1)}°</td>
                                  <td>{Math.round(t.confidence * 100)}%</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        ) : (
                          <table>
                            <thead>
                              <tr>
                                <th>Seat / stroke</th>
                                <th>Catch proxy</th>
                                <th>Finish proxy</th>
                                <th>Rate</th>
                                <th>Drive : recovery</th>
                                <th>Confidence</th>
                              </tr>
                            </thead>
                            <tbody>
                              {rows.map((s) => (
                                <tr
                                  key={`${s.seat}-${s.index}`}
                                  onClick={() => setTime(s.catch)}
                                  tabIndex={0}
                                  onKeyDown={(e) => {
                                    if (e.key === "Enter") setTime(s.catch);
                                  }}
                                >
                                  <td>
                                    <i
                                      style={{
                                        background: colors[(s.seat - 1) % 8],
                                      }}
                                    />
                                    Seat {s.seat} / {s.index}
                                  </td>
                                  <td>{s.catch.toFixed(3)} s</td>
                                  <td>{s.finish.toFixed(3)} s</td>
                                  <td>
                                    {s.rate.value?.toFixed(1) ?? "Withheld"}{" "}
                                    {s.rate.value ? "spm" : ""}
                                  </td>
                                  <td>
                                    {s.ratio.value?.toFixed(2) ?? "Withheld"}
                                  </td>
                                  <td>
                                    <span className="confidence-badge">
                                      {Math.round(s.confidence * 100)}%
                                    </span>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        )}
                      </div>
                      {!rows.length && (
                        <p className="note">
                          No complete cycles to report. Missing evidence is not
                          a zero measurement.
                        </p>
                      )}
                      <details className="metric-gates">
                        <summary>
                          <ShieldCheck size={15} /> Biomechanics & observability{" "}
                          <span>18 evidence-gated measures</span>
                          <ChevronDown size={15} />
                        </summary>
                        <div>
                          {metricNames.map((m) => (
                            <div key={m}>
                              <span>{m}</span>
                              <strong>Withheld · 0%</strong>
                              <small>
                                {m === "Split per 500 m"
                                  ? "Needs calibrated speed / distance, not a panning video."
                                  : "Needs validated 3D, handle/oar tracking, or phase-specific evidence."}
                              </small>
                            </div>
                          ))}
                        </div>
                      </details>
                    </section>
                  )}
                </>
              )}
            </>
          )}
          <footer className="page-footer">
            <span>
              <Waves size={16} /> Built for the details. Designed for the crew.
            </span>
            <span>
              American rowing terminology <i /> Private by default
            </span>
          </footer>
        </div>
      </main>
      {toast && (
        <div className="toast" role="status">
          <Check size={17} />
          {toast}
          <button
            onClick={() => setToast("")}
            aria-label="Dismiss notification"
          >
            <X size={15} />
          </button>
        </div>
      )}
      {modal && (
        <div className="modal-backdrop" onClick={() => setModal(null)}>
          <section
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-label={
              modal === "models"
                ? "Models and settings"
                : "Camera guide and methodology"
            }
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="modal-close"
              aria-label="Close dialog"
              onClick={() => setModal(null)}
            >
              <X size={20} />
            </button>
            {modal === "models" ? (
              <>
                <div className="eyebrow">ACCURACY IS A PROCESS</div>
                <h2>Models & processing</h2>
                <p>
                  Load authentic ONNX models from your device. No lightweight
                  pose model is silently substituted. Model weights are not
                  bundled with this research preview.
                </p>
                <div className="model-load">
                  <label>
                    <Cpu size={18} />
                    <strong>Person detector</strong>
                    <span>
                      {detector?.name || "YOLOX-L ONNX · recommended"}
                    </span>
                    <input
                      aria-label="Detector model"
                      type="file"
                      accept=".onnx"
                      onChange={(e) => setDetector(e.target.files?.[0] || null)}
                    />
                  </label>
                  <label>
                    <Activity size={18} />
                    <strong>Whole-body pose</strong>
                    <span>
                      {pose?.name || "DWPose / RTMW · COCO WholeBody 133"}
                    </span>
                    <input
                      aria-label="Pose model"
                      type="file"
                      accept=".onnx"
                      onChange={(e) => setPose(e.target.files?.[0] || null)}
                    />
                  </label>
                </div>
                <div className="model-cache" aria-label="Automatic model cache">
                  <div className="model-cache-head">
                    <strong>Automatic download & cache</strong>
                    <small>
                      Models download once from the configured source, then load
                      from this browser’s cache. A local file above always takes
                      precedence.
                    </small>
                  </div>
                  {(["detector", "pose"] as const).map((id) => {
                    const spec = modelSpecs[id];
                    const st = modelState[id];
                    const local = id === "detector" ? detector : pose;
                    const pct =
                      st.total > 0
                        ? Math.round((st.received / st.total) * 100)
                        : 0;
                    const label = local
                      ? `Using local file · ${local.name}`
                      : st.state === "cached"
                        ? `Cached · ${mb(st.total)}`
                        : st.state === "ready"
                          ? `Downloaded and cached · ${mb(st.total)}`
                          : st.state === "downloading"
                            ? `Downloading · ${pct}% of ${mb(st.total)}`
                            : st.state === "checking"
                              ? "Checking cache…"
                              : spec.url
                                ? "Not cached · downloads on first analysis"
                                : "No download URL in this build · import locally";
                    return (
                      <div className="model-cache-row" key={id}>
                        <div>
                          <strong>{spec.label}</strong>
                          <small>{label}</small>
                          {st.state === "error" && !local && (
                            <small className="model-cache-error" role="alert">
                              {st.error}
                            </small>
                          )}
                        </div>
                        {st.state === "downloading" && (
                          <progress
                            aria-label={`${spec.fileName} download progress`}
                            value={st.received}
                            max={st.total || 1}
                          />
                        )}
                        <div className="model-cache-actions">
                          {spec.url &&
                            !local &&
                            st.state !== "cached" &&
                            st.state !== "ready" &&
                            st.state !== "downloading" && (
                              <button
                                className="button small"
                                onClick={() => {
                                  downloadOrCache(
                                    id,
                                    spec,
                                    runToken.current,
                                    false,
                                  ).catch(() => {});
                                }}
                              >
                                <Download size={13} /> Download now
                              </button>
                            )}
                        </div>
                      </div>
                    );
                  })}
                  <button
                    className="text-button"
                    onClick={async () => {
                      await clearModelCache();
                      resolvedModels.current = {};
                      setModelState({
                        detector: {
                          state: "missing",
                          received: 0,
                          total: modelSpecs.detector.bytes,
                        },
                        pose: {
                          state: "missing",
                          received: 0,
                          total: modelSpecs.pose.bytes,
                        },
                      });
                      setToast("Cached model files removed.");
                    }}
                  >
                    Clear cached models
                  </button>
                </div>
                <div className="two-fields">
                  <label>
                    Detector export
                    <select
                      value={settings.detector}
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          detector: e.target.value as ModelSettings["detector"],
                        })
                      }
                    >
                      <option value="yolox">YOLOX · raw grid</option>
                      <option value="yolo">YOLOv8 / 11 · [1,84,N]</option>
                    </select>
                  </label>
                  <label>
                    Pose output
                    <select
                      value={settings.format}
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          format: e.target.value as ModelSettings["format"],
                        })
                      }
                    >
                      <option value="simcc">SimCC · DWPose / RTMW</option>
                      <option value="heatmap">Heatmaps · WholeBody 133</option>
                    </select>
                  </label>
                </div>
                <div className="two-fields">
                  <label>
                    Input width
                    <input
                      type="number"
                      min="128"
                      max="1024"
                      step="32"
                      value={settings.width}
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          width: Math.max(128, Math.min(1024, +e.target.value)),
                        })
                      }
                    />
                  </label>
                  <label>
                    Input height
                    <input
                      type="number"
                      min="128"
                      max="1024"
                      step="32"
                      value={settings.height}
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          height: Math.max(
                            128,
                            Math.min(1024, +e.target.value),
                          ),
                        })
                      }
                    />
                  </label>
                </div>
                <div className="two-fields">
                  <label>
                    Parallel pose workers
                    <select
                      aria-label="Parallel pose workers"
                      value={settings.workers}
                      onChange={(e) =>
                        setSettings({ ...settings, workers: +e.target.value })
                      }
                    >
                      <option value={0}>Auto (recommended)</option>
                      {[1, 2, 3, 4, 6, 8].map((n) => (
                        <option key={n} value={n}>
                          {n} worker{n === 1 ? "" : "s"}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Your machine
                    <input
                      aria-label="Parallel processing plan"
                      readOnly
                      value={summarizePlan(
                        planParallelism(readParallelEnv(), settings.workers),
                      )}
                    />
                  </label>
                </div>
                <p className="note">
                  Pose inference — the slow part — runs in parallel across these
                  workers, one model session each, so it scales with your CPU
                  cores even where WebGPU is unavailable. More workers are
                  faster but each loads its own copy of the pose model (~134 MB
                  for DWPose-l), so Auto leaves a core free for decoding and
                  caps the pool at 4. Threaded WASM per worker additionally
                  needs cross-origin isolation (COOP/COEP); the deployed site
                  enables that with a service worker after one reload.
                </p>
                <details className="region-settings">
                  <summary>
                    Detection region · exclude the coxswain / bystanders
                  </summary>
                  <p className="note">
                    Keep detection centers inside this image region. Percentages
                    are measured from the top-left. This is a static assist, not
                    boat-motion compensation.
                  </p>
                  <div className="two-fields">
                    {(["left", "top", "right", "bottom"] as const).map(
                      (key) => (
                        <label key={key}>
                          {key} (%)
                          <input
                            type="number"
                            min="0"
                            max="100"
                            value={settings.region[key]}
                            onChange={(e) =>
                              setSettings({
                                ...settings,
                                region: {
                                  ...settings.region,
                                  [key]: Math.max(
                                    0,
                                    Math.min(100, +e.target.value),
                                  ),
                                },
                              })
                            }
                          />
                        </label>
                      ),
                    )}
                  </div>
                </details>
                <label>
                  Test-time augmentation
                  <select
                    aria-label="Test-time augmentation"
                    value={settings.tta}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        tta: e.target.value as ModelSettings["tta"],
                      })
                    }
                  >
                    <option value="off">Off · one pass per crop</option>
                    <option value="adaptive">
                      Adaptive · extra passes only when the first is weak
                    </option>
                    <option value="always">
                      Always · four passes per crop
                    </option>
                  </select>
                </label>
                <p className="note">
                  Pose inference is the slow part of a run, and each pass of a
                  crop is a full whole-body inference. <strong>Adaptive</strong>{" "}
                  runs the single pass first and spends the other three (second
                  scale + both flips) only where that pass left a body joint
                  below 0.3 — the same four passes fuse exactly as in Always.{" "}
                  <strong>Always</strong> pays four passes on every crop;{" "}
                  <strong>Off</strong> pays one and reports no disagreement
                  flag. Crops are always distributed across the pose workers.
                </p>
                <label className="check-label">
                  <input
                    type="checkbox"
                    checked={settings.wasm}
                    onChange={(e) =>
                      setSettings({ ...settings, wasm: e.target.checked })
                    }
                  />
                  <span>
                    <strong>Force WASM compatibility mode</strong>
                    <small>
                      Otherwise a GPU adapter is probed and every model session
                      must pass a warm-up run before decoding. If the browser
                      has no adapter or a session fails — including a GPU lost
                      mid-analysis — the run continues on the parallel WASM
                      worker pool and says why in the analysis notes. The
                      runtime used is shown under the Analyze button.
                    </small>
                  </span>
                </label>
                <div className="alert">
                  <AlertTriangle size={17} />
                  <p>
                    Automatic download needs the model URLs configured for this
                    build. If a download fails, import the files locally to
                    continue. A working adapter is not an accuracy validation.
                  </p>
                </div>
                <div className="model-links">
                  <a
                    href="https://github.com/IDEA-Research/DWPose"
                    target="_blank"
                    rel="noreferrer"
                  >
                    DWPose model sources <ExternalLink size={13} />
                  </a>
                  <a
                    href="https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose"
                    target="_blank"
                    rel="noreferrer"
                  >
                    RTMPose / RTMW sources <ExternalLink size={13} />
                  </a>
                </div>
                <p className="note">
                  DWPose uses 288 × 384 RGB ImageNet-normalized input, with two
                  SimCC outputs. Check the export contract before loading other
                  models. ViTPose/Sapiens whole-body models need compatible
                  exports and substantially more memory. See the repository’s
                  model guide.
                </p>
                <button
                  className="button dark full"
                  onClick={() => setModal(null)}
                >
                  Save processing settings <Check size={15} />
                </button>
              </>
            ) : (
              <>
                <div className="eyebrow">A BETTER POINT OF VIEW</div>
                <h2>Good footage. Better feedback.</h2>
                <p>
                  Keep the entire rower visible throughout the catch, drive,
                  finish, and recovery.
                </p>
                <div className="guide-illustration">
                  <Waves size={45} />
                  <div className="guide-shell" />
                  <Focus size={26} />
                  <span>Launch alongside · near perpendicular</span>
                </div>
                <div className="guide-points">
                  <div>
                    <span>01</span>
                    <div>
                      <h3>Get alongside the shell</h3>
                      <p>
                        A slight offset can separate overlapping rowers. Avoid
                        bow-on or stern-on views for body angle and layback
                        review.
                      </p>
                    </div>
                  </div>
                  <div>
                    <span>02</span>
                    <div>
                      <h3>Give each seat some space</h3>
                      <p>
                        For an individual, use a closer view that includes
                        hands, knees, and the full slide. Exclude the coxswain
                        and bystanders or initialization will stop for review.
                      </p>
                    </div>
                  </div>
                  <div>
                    <span>03</span>
                    <div>
                      <h3>Keep the evidence honest</h3>
                      <p>
                        Solid joints are observed, dashed joints inferred.
                        Spray, riggers, reflections, and hidden far-side limbs
                        can defeat tracking. Manual corrections are local, not
                        proof of depth.
                      </p>
                    </div>
                  </div>
                </div>
                <div className="alert">
                  <ShieldCheck size={19} />
                  <p>
                    <strong>
                      Research preview, not validated biomechanics.
                    </strong>{" "}
                    Automatic camera azimuth, 3D lifting, oar tracking, and
                    sequencing diagnoses are not implemented. Unreliable metrics
                    are withheld. Whole-body inference requires separately
                    loaded model files.
                  </p>
                </div>
                <p className="note">
                  Seat 1 is bow seat; the highest number is stroke seat. Port /
                  starboard refers to the shell facing the bow. Bowside /
                  strokeside terminology varies with rigging; configure actual
                  sides rather than assuming them.
                </p>
                <a
                  className="button"
                  href="https://github.com/ehub22/Row-Analyzer/blob/arena/d1e71cca-row-analyzer/ROADMAP.md"
                  target="_blank"
                  rel="noreferrer"
                >
                  <BookOpen size={15} /> Read the architecture & roadmap{" "}
                  <ArrowUpRight size={14} />
                </a>
              </>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
function MetricCard({
  label,
  value,
  unit,
  sub,
  confidence,
  icon,
}: {
  label: string;
  value: string;
  unit: string;
  sub: string;
  confidence: number | null;
  icon: React.ReactNode;
}) {
  return (
    <div className="metric-card">
      <div className="metric-label">
        {label}
        {icon}
      </div>
      <div className="metric-value">
        {value}
        <span>{unit}</span>
      </div>
      <p>{sub}</p>
      <div className="metric-confidence">
        <span className={confidence === null ? "" : "available"} />
        {confidence === null
          ? "Awaiting analysis"
          : `${Math.round(confidence * 100)}% evidence confidence`}
      </div>
    </div>
  );
}
export default App;
