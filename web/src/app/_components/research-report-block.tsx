// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { useRef } from "react";

import { useMessage } from "~/core/store";
import { cn } from "~/lib/utils";

import { LoadingAnimation } from "./loading-animation";
import { Markdown } from "./markdown";

export function ResearchReportBlock({
  className,
  messageId,
}: {
  className?: string;
  researchId: string;
  messageId: string;
}) {
  const message = useMessage(messageId);
  const contentRef = useRef<HTMLDivElement>(null);
  return (
    <div
      ref={contentRef}
      className={cn("relative flex flex-col pb-8", className)}
    >
      <Markdown animate>{message?.content}</Markdown>
      {message?.isStreaming && <LoadingAnimation className="my-12" />}
    </div>
  );
}
