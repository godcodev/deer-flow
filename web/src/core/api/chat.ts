// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { env } from "~/env";

import { fetchStream } from "../sse";
import { sleep } from "../utils";

import type { ChatEvent } from "./types";

export function chatStream(
  userMessage: string,
  params: {
    thread_id: string;
    max_plan_iterations: number;
    max_step_num: number;
    interrupt_feedback?: string;
  },
  options: { abortSignal?: AbortSignal } = {},
) {
  if (location.search.includes("mock")) {
    return chatStreamMock(userMessage, params, options);
  }
  return fetchStream<ChatEvent>(
    (env.NEXT_PUBLIC_API_URL ?? "/api") + "/chat/stream",
    {
      body: JSON.stringify({
        messages: [{ role: "user", content: userMessage }],
        auto_accepted_plan: false,
        ...params,
      }),
      signal: options.abortSignal,
    },
  );
}

async function* chatStreamMock(
  userMessage: string,
  _: {
    thread_id: string;
    max_plan_iterations: number;
    max_step_num: number;
  } = {
    thread_id: "__mock__",
    max_plan_iterations: 3,
    max_step_num: 1,
  },
  options: { abortSignal?: AbortSignal } = {},
): AsyncIterable<ChatEvent> {
  const res = await fetch("/mock.txt", {
    signal: options.abortSignal,
  });
  await sleep(800);
  const text = await res.text();
  const chunks = text.split("\n\n");
  for (const chunk of chunks) {
    const [eventRaw, dataRaw] = chunk.split("\n") as [string, string];
    const [, event] = eventRaw.split("event: ", 2) as [string, string];
    const [, data] = dataRaw.split("data: ", 2) as [string, string];
    if (event === "message_chunk") {
      await sleep(0);
    } else if (event === "tool_call_result") {
      await sleep(1500);
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
