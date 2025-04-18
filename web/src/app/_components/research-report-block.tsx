// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { PauseCircleOutlined, SoundOutlined } from "@ant-design/icons";
import { useCallback, useRef, useState } from "react";

import { Button } from "~/components/ui/button";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "~/components/ui/tooltip";
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
  const contentRef = useRef<HTMLDivElement>(null);
  const [isTTS, setIsTTS] = useState(false);
  const handleTTS = useCallback(() => {
    if (contentRef.current) {
      if (isTTS) {
        window.speechSynthesis.cancel();
        setIsTTS(false);
      } else {
        const text = contentRef.current.textContent;
        if (text) {
          const utterance = new SpeechSynthesisUtterance(text);
          setIsTTS(true);
          window.speechSynthesis.speak(utterance);
        }
      }
    }
  }, [isTTS]);
  return (
    <div
      ref={contentRef}
      className={cn("relative flex flex-col pb-8", className)}
    >
      <div className="absolute top-2 right-6">
        {message?.content && !message.isStreaming && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                onClick={() => {
                  handleTTS();
                }}
              >
                {isTTS ? <PauseCircleOutlined /> : <SoundOutlined />}
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>{isTTS ? "Pause" : "Listen to the report"}</p>
            </TooltipContent>
          </Tooltip>
        )}
      </div>
      <Markdown animate>{message?.content}</Markdown>
      {message?.isStreaming && <LoadingAnimation className="my-12" />}
    </div>
  );
}
