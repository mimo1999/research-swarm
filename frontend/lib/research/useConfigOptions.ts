"use client";

import { useEffect, useState } from "react";
import type { ConfigOptions } from "./types";

export function useConfigOptions() {
  const [options, setOptions] = useState<ConfigOptions | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/config/options")
      .then((r) => r.json())
      .then((data) => {
        if (!cancelled) setOptions(data);
      })
      .catch(() => {
        /* left null — QueryForm renders its loading state */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return options;
}
