/**
 * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially useful
 * for Docker builds.
 */
// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import "./src/env.js";

/** @type {import("next").NextConfig} */
const config = {
  rewrites: async () => [
    {
      source: "/api/podcast/:path*",
      destination: "http://localhost:8000/api/podcast/:path*",
    },
  ],
};

export default config;
