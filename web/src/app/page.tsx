// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

"use client";

import { redirect } from "next/navigation";

export default function HomePage() {
  redirect("/chat");
}
