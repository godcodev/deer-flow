// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { useEffect, useState } from "react";

import { fetchReplayTitle } from "./chat";

export function useReplayMetadata() {
  const [title, setTitle] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<boolean>(false);
  useEffect(() => {
    if (title || isLoading) {
      return;
    }
    setIsLoading(true);
    fetchReplayTitle()
      .then((title) => {
        setError(false);
        setTitle(title ?? null);
        if (title) {
          document.title = `${title} - DeerFlow`;
        }
      })
      .catch(() => {
        setError(true);
        setTitle("Error: the replay is not available.");
        document.title = "DeerFlow";
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, [isLoading, title]);
  return { title, isLoading, hasError: error };
}
