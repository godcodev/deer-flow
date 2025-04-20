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
}: {
  className?: string;
  children: React.ReactNode;
  title: React.ReactNode;
}) {
  return (
    <ShadcnTooltip>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent className={className}>{title}</TooltipContent>
    </ShadcnTooltip>
  );
}
