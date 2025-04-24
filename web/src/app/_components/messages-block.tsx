// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { FastForward } from "lucide-react";
import { useCallback, useRef, useState } from "react";

import { Button } from "~/components/ui/button";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "~/components/ui/card";
import { fastForwardReplay } from "~/core/api";
import type { Option } from "~/core/messages";
import { useReplay } from "~/core/replay";
import { sendMessage, useStore } from "~/core/store";
import { cn } from "~/lib/utils";

import { ConversationStarter } from "./conversation-starter";
import { InputBox } from "./input-box";
import { MessageListView } from "./message-list-view";
import { RainbowText } from "./rainbow-text";

export function MessagesBlock({ className }: { className?: string }) {
  const messageCount = useStore((state) => state.messageIds.length);
  const responding = useStore((state) => state.responding);
  const { isReplay } = useReplay();
  const abortControllerRef = useRef<AbortController | null>(null);
  const [feedback, setFeedback] = useState<{ option: Option } | null>(null);
  const handleSend = useCallback(
    async (message: string, options?: { interruptFeedback?: string }) => {
      const abortController = new AbortController();
      abortControllerRef.current = abortController;
      try {
        await sendMessage(
          message,
          {
            interruptFeedback:
              options?.interruptFeedback ?? feedback?.option.value,
          },
          {
            abortSignal: abortController.signal,
          },
        );
      } catch {}
    },
    [feedback],
  );
  const handleCancel = useCallback(() => {
    console.info("cancel");
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
  }, []);
  const handleFeedback = useCallback(
    (feedback: { option: Option }) => {
      setFeedback(feedback);
    },
    [setFeedback],
  );
  const handleRemoveFeedback = useCallback(() => {
    setFeedback(null);
  }, [setFeedback]);
  const [fastForwarding, setFastForwarding] = useState(false);
  const handleFastForwardReplay = useCallback(() => {
    setFastForwarding(!fastForwarding);
    fastForwardReplay(!fastForwarding);
  }, [fastForwarding]);
  return (
    <div className={cn("flex h-full flex-col", className)}>
      <MessageListView
        className="flex flex-grow"
        onFeedback={handleFeedback}
        onSendMessage={handleSend}
      />
      {!isReplay ? (
        <div className="relative flex h-42 shrink-0 pb-4">
          {!responding && messageCount === 0 && (
            <ConversationStarter
              className="absolute top-[-218px] left-0"
              onSend={handleSend}
            />
          )}
          <InputBox
            className="h-full w-full"
            responding={responding}
            feedback={feedback}
            onSend={handleSend}
            onCancel={handleCancel}
            onRemoveFeedback={handleRemoveFeedback}
          />
        </div>
      ) : (
        <div className="flex h-42 w-full items-center justify-center">
          <Card className="w-full">
            <div className="flex items-center justify-between">
              <div className="flex-grow">
                <CardHeader>
                  <CardTitle>
                    <RainbowText animated={responding}>Replay Mode</RainbowText>
                  </CardTitle>
                  <CardDescription>
                    <RainbowText animated={responding}>
                      DeerFlow is now replaying the conversation...
                    </RainbowText>
                  </CardDescription>
                </CardHeader>
              </div>
              <div className="pr-4">
                {responding && (
                  <Button
                    className={cn(fastForwarding && "animate-pulse")}
                    variant={fastForwarding ? "default" : "outline"}
                    onClick={handleFastForwardReplay}
                  >
                    Fast Forward
                    <FastForward size={16} />
                  </Button>
                )}
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
