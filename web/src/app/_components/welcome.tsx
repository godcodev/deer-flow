// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { motion } from "framer-motion";

import { cn } from "~/lib/utils";

export function Welcome({ className }: { className?: string }) {
  return (
    <motion.div
      className={cn("flex flex-col", className)}
      style={{ transition: "all 0.2s ease-out" }}
      initial={{ opacity: 0, scale: 0.85 }}
      animate={{ opacity: 1, scale: 1 }}
    >
      <h3 className="mb-2 text-center text-3xl font-medium">
        👋 Hello, there!
      </h3>
      <div className="px-4 text-center text-lg text-gray-400">
        Welcome to{" "}
        <a
          href="https://github.com/bytedance/deer"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:underline"
        >
          🦌 Deer
        </a>
        , a research tool built on cutting-edge language models, helps you
        search on web, browse information, and handle complex tasks.
      </div>
    </motion.div>
  );
}
