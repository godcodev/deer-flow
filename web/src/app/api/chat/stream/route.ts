import { NextResponse, type NextRequest } from "next/server";

import { env } from "~/env";

export async function POST(request: NextRequest) {
  const requestBody = await request.json();

  const stream = new ReadableStream({
    async start(controller) {
      try {
        const response = await fetch(
          (env.NEXT_PUBLIC_API_URL
            ? env.NEXT_PUBLIC_API_URL
            : "http://localhost:8000/api") + "/chat/stream",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(requestBody),
          },
        );
        if (!response.ok) {
          const errorBody = await response.text();
          console.error("API error message:", errorBody);
          controller.enqueue(
            createErrorEvent(`API responded with status ${response.status}`),
          );
          controller.close();
          return;
        }
        const reader = response.body
          ?.pipeThrough(new TextDecoderStream())
          .getReader();
        if (!reader) {
          controller.enqueue(
            createErrorEvent("No data received from /api/chat/stream"),
          );
          controller.close();
          return;
        }

        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            break;
          }
          controller.enqueue(value);
        }

        controller.close();
        reader.releaseLock();
      } catch (error) {
        console.error("Stream error:", error);
        controller.enqueue(createErrorEvent("Stream interrupted"));
        controller.close();
      }
    },
  });

  return new NextResponse(stream, {
    headers: {
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "Content-Type": "text/event-stream",
    },
    status: 200,
  });
}

function createErrorEvent(message: string) {
  return `event: error\ndata: ${message}\n\n`;
}
