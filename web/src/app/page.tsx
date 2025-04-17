"use client";

import { GithubOutlined } from "@ant-design/icons";
import Link from "next/link";
import { useMemo } from "react";

import { Button } from "~/components/ui/button";
import { useStore } from "~/core/store";
import { cn } from "~/lib/utils";

import { Logo } from "./_components/logo";
import { MessagesBlock } from "./_components/messages-block";
import { ResearchBlock } from "./_components/research-block";

export default function HomePage() {
  const openResearchId = useStore((state) => state.openResearchId);
  const doubleColumnMode = useMemo(
    () => openResearchId !== null,
    [openResearchId],
  );
  return (
    <div className="flex h-full w-full justify-center">
      <header className="fixed top-0 left-0 flex h-12 w-full w-screen items-center justify-between px-4">
        <Logo />
        <Button
          className="opacity-70 transition-opacity duration-300 hover:opacity-100"
          variant="ghost"
          size="icon"
          asChild
        >
          <Link href="https://github.com/bytedance/deer" target="_blank">
            <GithubOutlined />
          </Link>
        </Button>
      </header>
      <div
        className={cn(
          "flex h-full w-full justify-center px-4 pt-12",
          doubleColumnMode && "gap-8",
        )}
      >
        <MessagesBlock
          className={cn(
            "shrink-0 transition-all duration-300 ease-out",
            !doubleColumnMode &&
              `w-[768px] translate-x-[min(calc((100vw-538px)*0.75/2),960px/2)]`,
            doubleColumnMode && `w-[538px]`,
          )}
        />
        <ResearchBlock
          className={cn(
            "w-[min(calc((100vw-538px)*0.75),960px)] pb-4 transition-all duration-300 ease-out",
            !doubleColumnMode && "scale-0",
            doubleColumnMode && "",
          )}
          researchId={openResearchId}
        />
      </div>
    </div>
  );
}
