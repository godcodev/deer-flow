// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { useMemo } from "react";

import { extractReplayIdFromURL } from "./get-replay-id";

export function useReplay() {
  const replayId = useMemo(() => {
    return extractReplayIdFromURL();
  }, []);
  return { isReplay: replayId != null, replayId };
}
