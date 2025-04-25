// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

"use client";

import dynamic from "next/dynamic";
import { Suspense } from "react";

const App = dynamic(() => import("./app"), { ssr: false });

export default function HomePage() {
  return (
    <Suspense fallback={<div>Loading DeerFlow...</div>}>
      <App />
    </Suspense>
  );
}
