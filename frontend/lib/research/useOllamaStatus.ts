"use client";

import { useEffect, useState } from "react";
import type { OllamaStatus } from "./types";

export function useOllamaStatus({
  url,
  model,
  deployment,
  enabled,
}: {
  url?: string;
  model?: string;
  deployment?: string;
  enabled: boolean;
}) {
  const [status, setStatus] = useState<OllamaStatus | null>(null);

  useEffect(() => {
    if (!enabled || !url || !model) {
      setStatus(null);
      return;
    }
    let cancelled = false;
    const params = new URLSearchParams({ url, model, deployment: deployment ?? "local" });
    // Debounce so status doesn't refire on every keystroke while typing the URL/model.
    const t = setTimeout(() => {
      fetch(`/api/config/ollama/status?${params}`)
        .then((r) => r.json())
        .then((data) => {
          if (!cancelled) setStatus(data);
        })
        .catch(() => {
          if (!cancelled) setStatus({ reachable: false, model_pulled: false, logged_in: null, message: "Status check failed" });
        });
    }, 400);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [url, model, deployment, enabled]);

  return status;
}
