// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { nanoid } from "nanoid";
import { create } from "zustand";

import { chatStream, generatePodcast } from "../api";
import type { MCPServerMetadata, SimpleMCPServerMetadata } from "../mcp";
import type { Message } from "../messages";
import { mergeMessage } from "../messages";
import { parseJSON } from "../utils";

import { useSettingsStore } from "./settings-store";

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

  setOngoingResearchId: (value: string | null) => void;
}>((set) => ({
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

  setOngoingResearchId(value: string | null) {
    set({ ongoingResearchId: value });
  }
}));

export async function sendMessage(
  content?: string,
  {
    interruptFeedback,
  }: {
    interruptFeedback?: string;
  } = {},
  options: { abortSignal?: AbortSignal } = {},
) {
  if (content !== undefined) {
    appendMessage({
      id: nanoid(),
      threadId: THREAD_ID,
      role: "user",
      content: content,
      contentChunks: [content],
    });
  }

  setResponding(true);
  try {
    const settings = useSettingsStore.getState();
    const generalSettings = settings.general;
    const mcpServers = settings.mcp.servers.filter((server) => server.enabled);
    let mcpSettings:
      | {
        servers: Record<
          string,
          MCPServerMetadata & {
            enabled_tools: string[];
            add_to_agents: string[];
          }
        >;
      }
      | undefined = undefined;
    if (mcpServers.length > 0) {
      mcpSettings = {
        servers: mcpServers.reduce((acc, cur) => {
          const { transport, env } = cur;
          let server: SimpleMCPServerMetadata;
          if (transport === "stdio") {
            server = {
              name: cur.name,
              transport,
              env,
              command: cur.command,
              args: cur.args,
            };
          } else {
            server = {
              name: cur.name,
              transport,
              env,
              url: cur.url,
            };
          }
          return {
            ...acc,
            [cur.name]: {
              ...server,
              enabled_tools: cur.tools.map((tool) => tool.name),
              add_to_agents: ["researcher"],
            },
          };
        }, {}),
      };
    }
    const stream = chatStream(
      content ?? "[REPLAY]",
      {
        thread_id: THREAD_ID,
        auto_accepted_plan: generalSettings.autoAcceptedPlan,
        max_plan_iterations: generalSettings.maxPlanIterations,
        max_step_num: generalSettings.maxStepNum,
        interrupt_feedback: interruptFeedback,
        mcp_settings: mcpSettings,
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
      message ??= getMessage(messageId);
      if (message) {
        message = mergeMessage(message, event);
        updateMessage(message);
      }
    }
  } catch {
    setOngoingResearchId(null);
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

function getMessage(id: string) {
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
  useStore.getState().setOngoingResearchId(value);
}

function appendResearch(researchId: string) {
  let planMessage: Message | undefined;
  const reversedMessageIds = [...useStore.getState().messageIds].reverse();
  for (const messageId of reversedMessageIds) {
    const message = getMessage(messageId);
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

export async function listenToPodcast(researchId: string) {
  const planMessageId = useStore.getState().researchPlanIds.get(researchId);
  const reportMessageId = useStore.getState().researchReportIds.get(researchId);
  if (planMessageId && reportMessageId) {
    const planMessage = getMessage(planMessageId)!;
    const title = parseJSON(planMessage.content, { title: "Untitled" }).title;
    const reportMessage = getMessage(reportMessageId);
    if (reportMessage?.content) {
      appendMessage({
        id: nanoid(),
        threadId: THREAD_ID,
        role: "user",
        content: "Please generate a podcast for the above research.",
        contentChunks: [],
      });
      const podCastMessageId = nanoid();
      const podcastObject = { title, researchId };
      const podcastMessage: Message = {
        id: podCastMessageId,
        threadId: THREAD_ID,
        role: "assistant",
        agent: "podcast",
        content: JSON.stringify(podcastObject),
        contentChunks: [],
        isStreaming: true,
      };
      appendMessage(podcastMessage);
      // Generating podcast...
      const audioUrl = await generatePodcast(reportMessage.content);
      useStore.setState((state) => ({
        messages: new Map(useStore.getState().messages).set(podCastMessageId, {
          ...state.messages.get(podCastMessageId)!,
          content: JSON.stringify({ ...podcastObject, audioUrl }),
          isStreaming: false,
        }),
      }));
    }
  }
}

export function useResearchTitle(researchId: string) {
  const planMessage = useMessage(
    useStore.getState().researchPlanIds.get(researchId),
  );
  return planMessage
    ? parseJSON(planMessage.content, { title: "" }).title
    : undefined;
}

export function useMessage(messageId: string | null | undefined) {
  return useStore((state) =>
    messageId ? state.messages.get(messageId) : undefined,
  );
}
