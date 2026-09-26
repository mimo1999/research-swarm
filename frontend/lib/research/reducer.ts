import type { Critique, Finding, FinalReport, NodeUpdate, RunState } from "./types";

export type RunAction =
  | { type: "SUBMITTED"; sessionId: string }
  | { type: "NODE_UPDATE"; payload: NodeUpdate }
  | { type: "INTERRUPTED"; findings: Finding[]; critiques: Critique[] }
  | { type: "FINAL_REPORT"; report: FinalReport }
  | { type: "RESUMED" }
  | { type: "DISCARDED" }
  | { type: "STREAM_FAILED" }
  | { type: "RESET" };

export function runReducer(state: RunState, action: RunAction): RunState {
  switch (action.type) {
    case "SUBMITTED":
      return { status: "running", sessionId: action.sessionId, trace: [] };

    case "NODE_UPDATE":
      if (state.status !== "running" && state.status !== "interrupted") return state;
      return { ...state, trace: [...state.trace, action.payload] };

    case "INTERRUPTED":
      if (state.status !== "running") return state;
      return {
        status: "interrupted",
        sessionId: state.sessionId,
        trace: state.trace,
        findings: action.findings,
        critiques: action.critiques,
      };

    case "FINAL_REPORT":
      if (state.status !== "running" && state.status !== "interrupted") return state;
      return { status: "done", sessionId: state.sessionId, trace: state.trace, report: action.report };

    case "RESUMED":
      if (state.status !== "interrupted") return state;
      return { status: "running", sessionId: state.sessionId, trace: state.trace };

    case "STREAM_FAILED":
      // The backend SSE stream ended (closed the generator) without ever
      // reaching final_report or an HITL interrupt -- an unhandled node
      // error, not the graceful supervisor-fallback path. Surface it as a
      // terminal state instead of leaving the UI stuck on "running".
      if (state.status !== "running" && state.status !== "interrupted") return state;
      return { status: "failed", sessionId: state.sessionId, trace: state.trace };

    case "DISCARDED":
    case "RESET":
      return { status: "idle" };

    default:
      return state;
  }
}

export const initialRunState: RunState = { status: "idle" };
