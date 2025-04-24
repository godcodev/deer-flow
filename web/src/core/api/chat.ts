// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import type { MCPServerMetadata } from "../mcp";
import { extractReplayIdFromSearchParams } from "../replay/get-replay-id";
import { fetchStream } from "../sse";
import { sleep } from "../utils";

import { resolveServiceURL } from "./resolve-service-url";
import type { ChatEvent } from "./types";

export function chatStream(
  userMessage: string,
  params: {
    thread_id: string;
    auto_accepted_plan: boolean;
    max_plan_iterations: number;
    max_step_num: number;
    interrupt_feedback?: string;
    mcp_settings?: {
      servers: Record<
        string,
        MCPServerMetadata & {
          enabled_tools: string[];
          add_to_agents: string[];
        }
      >;
    };
  },
  options: { abortSignal?: AbortSignal } = {},
) {
  if (location.search.includes("mock") || location.search.includes("replay=")) {
    return chatReplayStream(userMessage, params, options);
  }
  return fetchStream<ChatEvent>(resolveServiceURL("chat/stream"), {
    body: JSON.stringify({
      messages: [{ role: "user", content: userMessage }],
      ...params,
    }),
    signal: options.abortSignal,
  });
}

async function* chatReplayStream(
  userMessage: string,
  params: {
    thread_id: string;
    auto_accepted_plan: boolean;
    max_plan_iterations: number;
    max_step_num: number;
    interrupt_feedback?: string;
  } = {
    thread_id: "__mock__",
    auto_accepted_plan: false,
    max_plan_iterations: 3,
    max_step_num: 1,
    interrupt_feedback: undefined,
  },
  options: { abortSignal?: AbortSignal } = {},
): AsyncIterable<ChatEvent> {
  const urlParams = new URLSearchParams(window.location.search);
  let replayFilePath = "";
  if (urlParams.has("mock")) {
    replayFilePath =
      params.interrupt_feedback === "accepted"
        ? "/mock/before-interrupt.txt"
        : "/mock/after-interrupt.txt";
  } else {
    const replayId = extractReplayIdFromSearchParams(window.location.search);
    if (replayId) {
      replayFilePath = `/replay/${replayId}.txt`;
    } else {
      replayFilePath = "/mock/before-interrupt.txt";
    }
  }
  const res = await fetch(replayFilePath, {
    signal: options.abortSignal,
  });
  const text = await res.text();
  const chunks = text.split("\n\n");
  for (const chunk of chunks) {
    const [eventRaw, dataRaw] = chunk.split("\n") as [string, string];
    const [, event] = eventRaw.split("event: ", 2) as [string, string];
    const [, data] = dataRaw.split("data: ", 2) as [string, string];

    try {
      const chatEvent = {
        type: event,
        data: JSON.parse(data),
      } as ChatEvent;
      if (chatEvent.type === "message_chunk") {
        if (!chatEvent.data.finish_reason) {
          await sleepInReplay(250 + Math.random() * 250);
        }
      } else if (chatEvent.type === "tool_call_result") {
        await sleepInReplay(1500);
      }
      yield chatEvent;
      if (chatEvent.type === "tool_call_result") {
        await sleepInReplay(800);
      } else if (chatEvent.type === "message_chunk") {
        if (chatEvent.data.role === "user") {
          await sleepInReplay(1000);
        }
      }
    } catch (e) {
      console.error(e);
    }
  }
}

export async function sleepInReplay(ms: number) {
  if (fastForwardReplaying) {
    await sleep(0);
  } else {
    await sleep(ms);
  }
}

let fastForwardReplaying = false;
export function fastForwardReplay(value: boolean) {
  fastForwardReplaying = value;
}
