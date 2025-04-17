// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { useMessage } from "~/core/store";
import { cn } from "~/lib/utils";

import { LoadingAnimation } from "./loading-animation";
import { Markdown } from "./markdown";

export function ResearchReportBlock({
  className,
  messageId,
}: {
  className?: string;
  messageId: string;
}) {
  const message = useMessage(messageId);
  return (
    <div className={cn("flex flex-col pb-8", className)}>
      <Markdown animate>{message?.content}</Markdown>
      {message?.isStreaming && <LoadingAnimation className="my-12" />}
    </div>
  );
}
