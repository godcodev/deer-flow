// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { parse } from "best-effort-json-parser";
import { nanoid } from "nanoid";
import { create } from "zustand";

import { chatStream } from "../api";
import type { Message } from "../messages";
import { mergeMessage } from "../messages";

const THREAD_ID = nanoid();

export const useStore = create<{
  responding: boolean;
  threadId: string | undefined;
  messageIds: string[];
  messages: Map<string, Message>;
  researchIds: string[];
  researchPlanIds: Map<string, string>;
  researchReportIds: Map<string, string>;
  researchActivityIds: Map<string, string[]>;
  ongoingResearchId: string | null;
  openResearchId: string | null;
}>(() => ({
  responding: false,
  threadId: THREAD_ID,
  messageIds: [],
  messages: new Map<string, Message>(),
  researchIds: [],
  researchPlanIds: new Map<string, string>(),
  researchReportIds: new Map<string, string>(),
  researchActivityIds: new Map<string, string[]>(),
  ongoingResearchId: null,
  openResearchId: null,
}));

export async function sendMessage(
  content: string,
  {
    maxPlanIterations = 1,
    maxStepNum = 3,
    interruptFeedback,
  }: {
    maxPlanIterations?: number;
    maxStepNum?: number;
    interruptFeedback?: string;
  } = {},
  options: { abortSignal?: AbortSignal } = {},
) {
  appendMessage({
    id: nanoid(),
    threadId: THREAD_ID,
    role: "user",
    content: content,
    contentChunks: [content],
  });

  setResponding(true);
  try {
    const stream = chatStream(
      content,
      {
        thread_id: THREAD_ID,
        max_plan_iterations: maxPlanIterations,
        max_step_num: maxStepNum,
        interrupt_feedback: interruptFeedback,
      },
      options,
    );

    for await (const event of stream) {
      const { type, data } = event;
      const messageId = data.id;
      let message: Message | undefined;
      if (type === "tool_call_result") {
        message = findMessageByToolCallId(data.tool_call_id);
      } else if (!existsMessage(messageId)) {
        message = {
          id: messageId,
          threadId: data.thread_id,
          agent: data.agent,
          role: data.role,
          content: "",
          contentChunks: [],
          isStreaming: true,
          interruptFeedback,
        };
        appendMessage(message);
      }
      message ??= findMessage(messageId);
      if (message) {
        message = mergeMessage(message, event);
        updateMessage(message);
      }
    }
  } finally {
    setResponding(false);
  }
}

function setResponding(value: boolean) {
  useStore.setState({ responding: value });
}

function existsMessage(id: string) {
  return useStore.getState().messageIds.includes(id);
}

function findMessage(id: string) {
  return useStore.getState().messages.get(id);
}

function findMessageByToolCallId(toolCallId: string) {
  return Array.from(useStore.getState().messages.values())
    .reverse()
    .find((message) => {
      if (message.toolCalls) {
        return message.toolCalls.some((toolCall) => toolCall.id === toolCallId);
      }
      return false;
    });
}

function appendMessage(message: Message) {
  if (
    message.agent === "coder" ||
    message.agent === "reporter" ||
    message.agent === "researcher"
  ) {
    appendResearchActivity(message);
  }
  useStore.setState({
    messageIds: [...useStore.getState().messageIds, message.id],
    messages: new Map(useStore.getState().messages).set(message.id, message),
  });
}

function updateMessage(message: Message) {
  if (
    message.agent === "researcher" ||
    message.agent === "coder" ||
    message.agent === "reporter"
  ) {
    const id = message.id;
    if (!getOngoingResearchId()) {
      appendResearch(id);
      openResearch(id);
    }
  }

  if (
    getOngoingResearchId() &&
    message.agent === "reporter" &&
    !message.isStreaming
  ) {
    setOngoingResearchId(null);
  }
  useStore.setState({
    messages: new Map(useStore.getState().messages).set(message.id, message),
  });
}

function getOngoingResearchId() {
  return useStore.getState().ongoingResearchId;
}

function setOngoingResearchId(value: string | null) {
  return useStore.setState({
    ongoingResearchId: value,
  });
}

function appendResearch(researchId: string) {
  let planMessage: Message | undefined;
  const reversedMessageIds = [...useStore.getState().messageIds].reverse();
  for (const messageId of reversedMessageIds) {
    const message = findMessage(messageId);
    if (message?.agent === "planner") {
      planMessage = message;
      break;
    }
  }
  const messageIds = [researchId];
  messageIds.unshift(planMessage!.id);
  useStore.setState({
    ongoingResearchId: researchId,
    researchIds: [...useStore.getState().researchIds, researchId],
    researchPlanIds: new Map(useStore.getState().researchPlanIds).set(
      researchId,
      planMessage!.id,
    ),
    researchActivityIds: new Map(useStore.getState().researchActivityIds).set(
      researchId,
      messageIds,
    ),
  });
}

function appendResearchActivity(message: Message) {
  const researchId = getOngoingResearchId();
  if (researchId) {
    const researchActivityIds = useStore.getState().researchActivityIds;
    useStore.setState({
      researchActivityIds: new Map(researchActivityIds).set(researchId, [
        ...researchActivityIds.get(researchId)!,
        message.id,
      ]),
    });
    if (message.agent === "reporter") {
      useStore.setState({
        researchReportIds: new Map(useStore.getState().researchReportIds).set(
          researchId,
          message.id,
        ),
      });
    }
  }
}

export function openResearch(researchId: string | null) {
  useStore.setState({
    openResearchId: researchId,
  });
}

export function useResearchTitle(researchId: string) {
  const planMessage = useMessage(
    useStore.getState().researchPlanIds.get(researchId),
  );
  return planMessage ? parse(planMessage.content).title : undefined;
}

export function useMessage(messageId: string | null | undefined) {
  return useStore((state) =>
    messageId ? state.messages.get(messageId) : undefined,
  );
}

// void sendMessage(
//   "How many times taller is the Eiffel Tower than the tallest building in the world?",
// );
