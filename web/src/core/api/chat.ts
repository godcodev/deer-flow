// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import type { MCPServerMetadata } from "../mcp";
import { fetchStream } from "../sse";
import { sleep } from "../utils";

import { resolveServiceURL } from "./resolve-service-url";
import type { ChatEvent } from "./types";

export function chatStream(
  userMessage: string,
  params: {
    thread_id: string;
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
  if (location.search.includes("mock")) {
    return chatStreamMock(userMessage, params, options);
  }
  return fetchStream<ChatEvent>(resolveServiceURL("chat/stream"), {
    body: JSON.stringify({
      messages: [{ role: "user", content: userMessage }],
      auto_accepted_plan: false,
      ...params,
    }),
    signal: options.abortSignal,
  });
}

async function* chatStreamMock(
  userMessage: string,
  params: {
    thread_id: string;
    max_plan_iterations: number;
    max_step_num: number;
    interrupt_feedback?: string;
  } = {
    thread_id: "__mock__",
    max_plan_iterations: 3,
    max_step_num: 1,
    interrupt_feedback: undefined,
  },
  options: { abortSignal?: AbortSignal } = {},
): AsyncIterable<ChatEvent> {
  const mockFile =
    params.interrupt_feedback === "accepted"
      ? "/mock-before-interrupt.txt"
      : "/mock-after-interrupt.txt";
  const res = await fetch(mockFile, {
    signal: options.abortSignal,
  });
  await sleep(500);
  const text = await res.text();
  const chunks = text.split("\n\n");
  for (const chunk of chunks) {
    const [eventRaw, dataRaw] = chunk.split("\n") as [string, string];
    const [, event] = eventRaw.split("event: ", 2) as [string, string];
    const [, data] = dataRaw.split("data: ", 2) as [string, string];
    if (event === "message_chunk") {
      await sleep(100);
    } else if (event === "tool_call_result") {
      await sleep(2000);
    }
    try {
      yield {
        type: event,
        data: JSON.parse(data),
      } as ChatEvent;
    } catch (e) {
      console.error(e);
    }
  }
}
