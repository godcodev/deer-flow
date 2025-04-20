// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { useStickToBottom } from "use-stick-to-bottom";

import { ScrollArea } from "~/components/ui/scroll-area";
import { cn } from "~/lib/utils";

export function ScrollContainer({
  className,
  children,
  scrollShadow = true,
  scrollShadowColor = "var(--background)",
}: {
  className?: string;
  children: React.ReactNode;
  scrollShadow?: boolean;
  scrollShadowColor?: string;
}) {
  const { scrollRef, contentRef } = useStickToBottom({
    initial: "instant",
  });
  return (
    <div className={cn("relative", className)}>
      {scrollShadow && (
        <>
          <div
            className={cn(
              "absolute top-0 right-0 left-0 z-10 h-10 bg-gradient-to-b",
              `from-[var(--scroll-shadow-color)] to-transparent`,
            )}
            style={
              {
                "--scroll-shadow-color": scrollShadowColor,
              } as React.CSSProperties
            }
          ></div>
          <div
            className={cn(
              "absolute right-0 bottom-0 left-0 z-10 h-10 bg-gradient-to-b",
              `from-transparent to-[var(--scroll-shadow-color)]`,
            )}
            style={
              {
                "--scroll-shadow-color": scrollShadowColor,
              } as React.CSSProperties
            }
          ></div>
        </>
      )}
      <ScrollArea ref={scrollRef} className="h-full w-full">
        <div className="h-fit w-full" ref={contentRef}>
          {children}
        </div>
      </ScrollArea>
    </div>
  );
}
