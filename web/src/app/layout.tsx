// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import "~/styles/globals.css";

import { type Metadata } from "next";
import { Geist } from "next/font/google";

import { ThemeProviderWrapper } from "~/app/_components/theme-provider-wrapper";
import { TooltipProvider } from "~/components/ui/tooltip";

import { Toaster } from "./_components/toaster";

export const metadata: Metadata = {
  title: "🦌 DeerFlow",
  description:
    "Deep Exploration and Efficient Research, an AI tool that combines language models with specialized tools for research tasks.",
  icons: [{ rel: "icon", url: "/favicon.ico" }],
};

const geist = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans",
});

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${geist.variable}`} suppressHydrationWarning>
      <body className="bg-app">
        <ThemeProviderWrapper>
          <TooltipProvider>{children}</TooltipProvider>
        </ThemeProviderWrapper>
        <Toaster />
      </body>
    </html>
  );
}
