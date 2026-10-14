import type { Session } from "./types";
import { automaticCamera } from "./calibration";
import { liftSequence } from "./lifting";
export function deriveAdvanced(session: Session): Session {
  return {
    ...session,
    camera: session.demo
      ? session.camera
      : automaticCamera(session.frames, session.boat),
    frames: liftSequence(session.frames),
  };
}
