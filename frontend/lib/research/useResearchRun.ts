"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { runReducer, initialRunState } from "./reducer";
import type { RunError } from "./types";

export type ResearchSubmitBody = {
  topic: string;
  audience: string;
  depth: string;
  max_sources: number;
  provider: string;
  model: string;
  ollama_url?: string | null;
  ollama_deployment?: string | null;
  hitl_enabled: boolean;
};

async function postJson(url: string, body: unknown) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res;
}

// The research run lives on the server, not in this connection. Remembering
// the session id lets a reloaded tab rejoin it and replay the event log
// instead of stranding a run that is still burning tokens.
const SESSION_KEY = "research-swarm:session-id";

function rememberSession(sessionId: string) {
  try {
    sessionStorage.setItem(SESSION_KEY, sessionId);
  } catch {
    // Private mode / storage disabled — rejoin is a nicety, not a requirement.
  }
}

function forgetSession() {
  try {
    sessionStorage.removeItem(SESSION_KEY);
  } catch {
    /* ignore */
  }
}

export function useResearchRun() {
  const [state, dispatch] = useReducer(runReducer, initialRunState);
  const [error, setError] = useState<RunError>(null);
  const [reconnecting, setReconnecting] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  const connect = useCallback((sessionId: string, from?: number) => {
    esRef.current?.close();
    // `from` replays the log from a known point (a rejoin after reload).
    // Without it the browser's own reconnect sends Last-Event-ID and the
    // server resumes exactly where this client left off.
    const query = from === undefined ? "" : `?from=${from}`;
    const es = new EventSource(`/api/research/${sessionId}/stream${query}`);

    const close = () => {
      setReconnecting(false);
      es.close();
    };

    es.onopen = () => setReconnecting(false);

    // A run's event log carries a "done" after EVERY segment, not just the
    // final one: ResearchRun._drive emits interrupted+done when it pauses
    // for HITL, then -- once resumed -- a second, separate _drive() call
    // streams the rest and emits its own final done. A single replay (e.g.
    // reconnecting after a page reload) reads the whole log in one
    // continuous EventSource, so the FIRST "done" it sees is only "this
    // segment is over," not "nothing more will ever arrive." Closing
    // unconditionally on every "done" (the old behavior) tore the stream
    // down right after that first segment, stranding a reload on the stale
    // "interrupted" panel even though the run had actually finished. Track
    // whether we're mid-interrupt so "done" only closes the connection when
    // it's the true end.
    let awaitingResume = false;

    es.addEventListener("node_update", (e: MessageEvent) => {
      awaitingResume = false;
      const { node, update } = JSON.parse(e.data);
      dispatch({ type: "NODE_UPDATE", payload: { node, update, ts: Date.now() } });
    });
    es.addEventListener("interrupted", (e: MessageEvent) => {
      awaitingResume = true;
      const { findings, critiques } = JSON.parse(e.data);
      dispatch({ type: "INTERRUPTED", findings, critiques });
    });
    es.addEventListener("final_report", (e: MessageEvent) => {
      awaitingResume = false;
      dispatch({ type: "FINAL_REPORT", report: JSON.parse(e.data) });
    });
    es.addEventListener("error", (e: MessageEvent) => {
      // This listener catches two different things: a named "error" event the
      // server sent (has .data), and a native transport failure (no .data).
      if (e.data) {
        try {
          setError({ message: JSON.parse(e.data).message });
        } catch {
          setError({ message: "Research failed" });
        }
        dispatch({ type: "STREAM_FAILED" });
        close();
        return;
      }
      // Transport failure. readyState CONNECTING means the browser is already
      // retrying — the run is untouched on the server and will replay from
      // Last-Event-ID, so leave it alone rather than killing the UI.
      if (es.readyState === EventSource.CONNECTING) {
        setReconnecting(true);
        return;
      }
      setError({ message: "Connection lost" });
      dispatch({ type: "STREAM_FAILED" });
      close();
    });
    es.addEventListener("done", () => {
      // "done" right after "interrupted" just ends that segment -- more may
      // still be in the log (or arrive later, once a human resumes), so
      // keep reading instead of tearing the connection down. Reset the flag
      // either way: if this really was the last event, the stream idles
      // harmlessly; the next real reconnect (a resume, or a future replay)
      // starts its own EventSource regardless.
      if (awaitingResume) {
        awaitingResume = false;
        return;
      }
      close();
    });

    esRef.current = es;
  }, []);

  // On mount, rejoin a run left behind by a reload. Replaying from 0 rebuilds
  // the trace, findings, and report from the server's event log.
  useEffect(() => {
    let cancelled = false;
    const stored = (() => {
      try {
        return sessionStorage.getItem(SESSION_KEY);
      } catch {
        return null;
      }
    })();
    if (!stored) return;

    (async () => {
      const res = await fetch(`/api/research/${stored}/status`);
      if (cancelled) return;
      if (!res.ok) {
        forgetSession();
        return;
      }
      dispatch({ type: "SUBMITTED", sessionId: stored });
      connect(stored, 0);
    })();

    return () => {
      cancelled = true;
    };
  }, [connect]);

  const submit = useCallback(
    async (body: ResearchSubmitBody, files: File[] = [], urls: string[] = []) => {
      setError(null);
      const res = await postJson("/api/research", body);
      if (!res.ok) {
        setError({ message: await res.text() });
        return;
      }
      const { session_id } = await res.json();

      if (files.length || urls.length) {
        const form = new FormData();
        for (const f of files) form.append("files", f);
        form.append("urls", urls.join("\n"));
        const docRes = await fetch(`/api/sessions/${session_id}/documents`, {
          method: "POST",
          body: form,
        });
        if (!docRes.ok) {
          setError({ message: `Document ingestion failed: ${await docRes.text()}` });
          return;
        }
      }

      rememberSession(session_id);
      dispatch({ type: "SUBMITTED", sessionId: session_id });
      connect(session_id);
    },
    [connect]
  );

  const resume = useCallback(
    async (action: "approve" | "edit" | "discard", feedback?: string) => {
      if (state.status !== "interrupted") return;
      const res = await postJson(`/api/research/${state.sessionId}/resume`, { action, feedback });
      if (!res.ok) {
        setError({ message: await res.text() });
        return;
      }
      if (action === "discard") {
        forgetSession();
        dispatch({ type: "DISCARDED" });
        return;
      }
      // Event ids continue across the pause, so resume from the server's
      // cursor rather than replaying the events we already rendered.
      const { resume_from } = await res.json();
      dispatch({ type: "RESUMED" });
      connect(state.sessionId, resume_from);
    },
    [state, connect]
  );

  const reset = useCallback(() => {
    esRef.current?.close();
    forgetSession();
    setReconnecting(false);
    dispatch({ type: "RESET" });
    setError(null);
  }, []);

  useEffect(() => {
    return () => {
      esRef.current?.close();
    };
  }, []);

  return {
    state,
    error,
    reconnecting,
    submit,
    resume,
    reset,
    clearError: () => setError(null),
  };
}
