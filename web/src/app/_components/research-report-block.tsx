// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { SoundOutlined } from "@ant-design/icons";
import { useCallback, useRef, useState } from "react";

import { Button } from "~/components/ui/button";
import { listenToPodcast, useMessage } from "~/core/store";
import { cn } from "~/lib/utils";

import { LoadingAnimation } from "./loading-animation";
import { Markdown } from "./markdown";
import { Tooltip } from "./tooltip";

export function ResearchReportBlock({
  className,
  researchId,
  messageId,
}: {
  className?: string;
  researchId: string;
  messageId: string;
}) {
  const message = useMessage(messageId);
  const contentRef = useRef<HTMLDivElement>(null);
  const [isGenerated, setGenerated] = useState(false);
  const handleListenToReport = useCallback(async () => {
    if (isGenerated) {
      return;
    }
    setGenerated(true);
    await listenToPodcast(researchId);
  }, [isGenerated, researchId]);
  return (
    <div
      ref={contentRef}
      className={cn("relative flex flex-col pb-8", className)}
    >
      <div className="absolute top-2 right-6">
        {message?.content && !message.isStreaming && (
          <Tooltip
            title={
              isGenerated ? "The podcast is generated" : "Generate podcast"
            }
          >
            <Button
              variant="outline"
              size="icon"
              disabled={isGenerated}
              onClick={() => {
                void handleListenToReport();
              }}
            >
              <SoundOutlined />
            </Button>
          </Tooltip>
        )}
      </div>
      <Markdown animate>{message?.content}</Markdown>
      {message?.isStreaming && <LoadingAnimation className="my-12" />}
    </div>
  );
}
