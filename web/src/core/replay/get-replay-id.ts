// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

export function extractReplayIdFromURL() {
  if (typeof window === "undefined") {
    return null;
  }
  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.has("replay")) {
    return urlParams.get("replay");
  }
  return null;
}
