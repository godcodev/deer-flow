// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { parse } from "best-effort-json-parser";
import { motion } from "framer-motion";
import { useCallback, useMemo } from "react";

import { Button } from "~/components/ui/button";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "~/components/ui/card";
import type { Message, Option } from "~/core/messages";
import {
  openResearch,
  sendMessage,
  useMessage,
  useResearchTitle,
  useStore,
} from "~/core/store";
import { cn } from "~/lib/utils";

import { LoadingAnimation } from "./loading-animation";
import { Markdown } from "./markdown";
import { RainbowText } from "./rainbow-text";
import { RollingText } from "./rolling-text";
import { ScrollContainer } from "./scroll-container";

export function MessageListView({
  className,
  onFeedback,
}: {
  className?: string;
  onFeedback?: (feedback: { option: Option }) => void;
}) {
  const messageIds = useStore((state) => state.messageIds);
  const interruptMessage = useStore((state) => {
    if (messageIds.length >= 2) {
      const lastMessage = state.messages.get(
        messageIds[messageIds.length - 1]!,
      );
      return lastMessage?.finishReason === "interrupt" ? lastMessage : null;
    }
    return null;
  });
  const waitingForFeedbackMessageId = useStore((state) => {
    if (messageIds.length >= 2) {
      const lastMessage = state.messages.get(
        messageIds[messageIds.length - 1]!,
      );
      if (lastMessage && lastMessage.finishReason === "interrupt") {
        return state.messageIds[state.messageIds.length - 2];
      }
    }
    return null;
  });
  const responding = useStore((state) => state.responding);
  const noOngoingResearch = useStore(
    (state) => state.ongoingResearchId === null,
  );
  const ongoingResearchIsOpen = useStore(
    (state) => state.ongoingResearchId === state.openResearchId,
  );

  return (
    <ScrollContainer
      className={cn(
        "flex h-full w-full flex-col overflow-y-auto pt-4",
        className,
      )}
      scrollShadowColor="#f7f5f3"
    >
      <ul className="flex flex-col">
        {messageIds.map((messageId) => (
          <MessageListItem
            key={messageId}
            messageId={messageId}
            waitForFeedback={waitingForFeedbackMessageId === messageId}
            interruptMessage={interruptMessage}
            onFeedback={onFeedback}
          />
        ))}
        <div className="flex h-8 w-full shrink-0"></div>
      </ul>
      {responding && (noOngoingResearch || !ongoingResearchIsOpen) && (
        <LoadingAnimation className="ml-4" />
      )}
    </ScrollContainer>
  );
}

function MessageListItem({
  className,
  messageId,
  waitForFeedback,
  onFeedback,
  interruptMessage,
}: {
  className?: string;
  messageId: string;
  waitForFeedback?: boolean;
  onFeedback?: (feedback: { option: Option }) => void;
  interruptMessage?: Message | null;
}) {
  const message = useMessage(messageId);
  const startOfResearch = useStore((state) =>
    state.researchIds.includes(messageId),
  );
  if (message) {
    if (
      message.role === "user" ||
      message.agent === "coordinator" ||
      message.agent === "planner" ||
      startOfResearch
    ) {
      let content: React.ReactNode;
      if (message.agent === "planner") {
        content = (
          <div className="w-full px-4">
            <PlanCard
              message={message}
              waitForFeedback={waitForFeedback}
              interruptMessage={interruptMessage}
              onFeedback={onFeedback}
            />
          </div>
        );
      } else if (startOfResearch) {
        content = (
          <div className="w-full px-4">
            <ResearchCard researchId={message.id} />
          </div>
        );
      } else {
        content = message.content ? (
          <div
            className={cn(
              "flex w-full px-4",
              message.role === "user" && "justify-end",
              className,
            )}
          >
            <MessageBubble message={message}>
              <div className="flex w-full flex-col">
                <Markdown
                  animate={message.role !== "user" && message.isStreaming}
                >
                  {message?.content}
                </Markdown>
              </div>
            </MessageBubble>
          </div>
        ) : null;
      }
      if (content) {
        return (
          <motion.li
            className="mt-10"
            key={messageId}
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            style={{ transition: "all 0.2s ease-out" }}
            transition={{
              duration: 0.2,
              ease: "easeOut",
            }}
          >
            {content}
          </motion.li>
        );
      }
    }
    return null;
  }

  function MessageBubble({
    className,
    message,
    children,
  }: {
    className?: string;
    message: Message;
    children: React.ReactNode;
  }) {
    return (
      <div
        className={cn(
          `flex w-fit max-w-[85%] flex-col rounded-2xl px-4 py-3 shadow-xs`,
          message.role === "user" &&
            "text-primary-foreground rounded-ee-none bg-[#007aff]",
          message.role === "assistant" && "rounded-es-none bg-white",
          className,
        )}
      >
        {children}
      </div>
    );
  }

  function ResearchCard({
    className,
    researchId,
  }: {
    className?: string;
    researchId: string;
  }) {
    const reportId = useStore((state) =>
      state.researchReportIds.get(researchId),
    );
    const hasReport = useStore((state) =>
      state.researchReportIds.has(researchId),
    );
    const reportGenerating = useStore(
      (state) => hasReport && state.messages.get(reportId!)!.isStreaming,
    );
    const openResearchId = useStore((state) => state.openResearchId);
    const state = useMemo(() => {
      if (hasReport) {
        return reportGenerating ? "Generating report..." : "Report generated";
      }
      return "Researching...";
    }, [hasReport, reportGenerating]);
    const title = useResearchTitle(researchId);
    const handleOpen = useCallback(() => {
      if (openResearchId === researchId) {
        openResearch(null);
      } else {
        openResearch(researchId);
      }
    }, [openResearchId, researchId]);
    return (
      <Card className={cn("w-full bg-white", className)}>
        <CardHeader>
          <CardTitle>
            <RainbowText animated={state !== "Report generated"}>
              {title !== undefined && title !== "" ? title : "Deep Research"}
            </RainbowText>
          </CardTitle>
        </CardHeader>
        <CardFooter>
          <div className="flex w-full">
            <RollingText className="flex-grow text-sm opacity-50">
              {state}
            </RollingText>
            <Button onClick={handleOpen}>
              {!openResearchId ? "Open" : "Close"}
            </Button>
          </div>
        </CardFooter>
      </Card>
    );
  }
}

const GREETINGS = ["Cool", "Sounds great", "Looks good", "Great", "Awesome"];
function PlanCard({
  className,
  message,
  interruptMessage,
  onFeedback,
  waitForFeedback,
}: {
  className?: string;
  message: Message;
  interruptMessage?: Message | null;
  onFeedback?: (feedback: { option: Option }) => void;
  waitForFeedback?: boolean;
}) {
  const plan = useMemo<{
    title?: string;
    thought?: string;
    steps?: { title?: string; description?: string }[];
  }>(() => {
    return parse(message.content ?? "");
  }, [message.content]);
  const handleAccept = useCallback(async () => {
    await sendMessage(
      `${GREETINGS[Math.floor(Math.random() * GREETINGS.length)]}! ${Math.random() > 0.5 ? "Let's get started." : "Let's start."}`,
      {
        interruptFeedback: "accepted",
      },
    );
  }, []);
  return (
    <Card className={cn("w-full bg-white", className)}>
      <CardHeader>
        <CardTitle>
          <h1 className="text-xl font-medium">
            <Markdown animate>
              {plan.title !== undefined && plan.title !== ""
                ? plan.title
                : "Deep Research"}
            </Markdown>
          </h1>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Markdown className="opacity-80" animate>
          {plan.thought}
        </Markdown>
        {plan.steps && (
          <ul className="my-2 flex list-decimal flex-col gap-4 border-l-[2px] pl-8">
            {plan.steps.map((step, i) => (
              <li key={`step-${i}`}>
                <h3 className="mb text-lg font-medium">
                  <Markdown animate>{step.title}</Markdown>
                </h3>
                <div className="text-muted-foreground text-sm">
                  <Markdown animate>{step.description}</Markdown>
                </div>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
      <CardFooter className="flex justify-end">
        {!message.isStreaming && interruptMessage?.options?.length && (
          <motion.div
            className="flex gap-2"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.3 }}
          >
            {interruptMessage?.options.map((option) => (
              <Button
                key={option.value}
                variant={option.value === "accepted" ? "default" : "outline"}
                disabled={!waitForFeedback}
                onClick={() => {
                  if (option.value === "accepted") {
                    void handleAccept();
                  } else {
                    onFeedback?.({
                      option,
                    });
                  }
                }}
              >
                {option.text}
              </Button>
            ))}
          </motion.div>
        )}
      </CardFooter>
    </Card>
  );
}
