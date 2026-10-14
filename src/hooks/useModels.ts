import { useEffect, useRef, useState } from "react";
import {
  clearModels,
  importModel,
  loadModel,
  type ModelKind,
  type ModelProgress,
} from "../core/models";
export function useModels() {
  const [detector, setDetector] = useState<File | null>(null);
  const [pose, setPose] = useState<File | null>(null);
  const [progress, setProgress] = useState<ModelProgress | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  async function ensure() {
    if (controller.current)
      throw Error("Model preparation is already running.");
    const c = new AbortController();
    controller.current = c;
    setLoading(true);
    setError("");
    setMessage("");
    try {
      const d = detector
        ? { file: detector }
        : await loadModel("detector", setProgress, c.signal);
      setDetector(d.file);
      const p = pose
        ? { file: pose }
        : await loadModel("pose", setProgress, c.signal);
      setPose(p.file);
      setMessage(
        [d.warning, p.warning].filter(Boolean).join(" ") ||
          "Weights ready. Runtime input/output checks run before video analysis.",
      );
      return { detector: d.file, pose: p.file };
    } catch (e) {
      const text = c.signal.aborted
        ? "Download canceled. Local-file import is still available."
        : e instanceof Error
          ? e.message
          : String(e);
      setError(text);
      throw Error(text);
    } finally {
      controller.current = null;
      setLoading(false);
      setProgress(null);
    }
  }
  async function local(kind: ModelKind, file?: File) {
    if (!file || controller.current) return;
    const c = new AbortController();
    controller.current = c;
    setLoading(true);
    setError("");
    setMessage(`Checking ${file.name}…`);
    try {
      const result = await importModel(kind, file);
      c.signal.throwIfAborted();
      (kind === "detector" ? setDetector : setPose)(result.file);
      setMessage(
        result.warning || "Reference model verified and cached on this device.",
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setMessage("");
    } finally {
      controller.current = null;
      setLoading(false);
    }
  }
  async function clear() {
    if (controller.current) return;
    try {
      await clearModels();
      setDetector(null);
      setPose(null);
      setMessage("Cached weights removed from this browser.");
      setError("");
    } catch {
      setError(
        "Could not clear the model cache. Clear site data in browser settings.",
      );
    }
  }
  return {
    detector,
    pose,
    progress,
    loading,
    message,
    error,
    ensure,
    local,
    clear,
    cancel: () => controller.current?.abort(),
  };
}
export type ModelsController = ReturnType<typeof useModels>;
