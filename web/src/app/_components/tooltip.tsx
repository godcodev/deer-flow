// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import {
  Tooltip as ShadcnTooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";

export function Tooltip({
  className,
  children,
  title,
  open,
  side,
  sideOffset,
}: {
  className?: string;
  children: React.ReactNode;
  title?: React.ReactNode;
  open?: boolean;
  side?: "left" | "right" | "top" | "bottom";
  sideOffset?: number;
}) {
  return (
    <ShadcnTooltip delayDuration={750} open={open}>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent side={side} sideOffset={sideOffset} className={className}>
        {title}
      </TooltipContent>
    </ShadcnTooltip>
  );
}
