// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

"use client";

import { GithubOutlined } from "@ant-design/icons";
import dynamic from "next/dynamic";
import Link from "next/link";
import { Suspense } from "react";

import { Button } from "~/components/ui/button";
import { useReplay } from "~/core/replay";

import { Logo } from "../_components/logo";
import { ThemeToggle } from "../_components/theme-toggle";
import { Tooltip } from "../_components/tooltip";
import { SettingsDialog } from "../_settings/dialogs/settings-dialog";

const Main = dynamic(() => import("./main"), { ssr: false });

export default function HomePage() {
  const { isReplay } = useReplay();
  return (
    <div className="flex h-screen w-screen justify-center overscroll-none">
      <header className="fixed top-0 left-0 flex h-12 w-full items-center justify-between px-4">
        <Logo />
        <div className="flex items-center">
          <Tooltip title="Star DeerFlow on GitHub">
            <Button variant="ghost" size="icon" asChild>
              <Link
                href="https://github.com/bytedance/deer-flow"
                target="_blank"
              >
                <GithubOutlined />
              </Link>
            </Button>
          </Tooltip>
          <ThemeToggle />
          {!isReplay && <SettingsDialog />}
        </div>
      </header>
      <Suspense fallback={<div>Loading DeerFlow...</div>}>
        <Main />
      </Suspense>
    </div>
  );
}
