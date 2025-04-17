// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { useState } from "react";

import { Markdown } from "./markdown";

export function Logo() {
  const [text, setText] = useState("🦌 DeerFlow");
  return (
    <a
      className="text-sm opacity-70 transition-opacity duration-300 hover:opacity-100"
      target="_blank"
      href="https://github.com/bytedance/deer-flow"
      onMouseEnter={() =>
        setText("🦌 **D**eep **E**xploration and **E**fficient **R**esearch")
      }
      onMouseLeave={() => setText("🦌 DeerFlow")}
    >
      <Markdown animate>{text}</Markdown>
    </a>
  );
}
