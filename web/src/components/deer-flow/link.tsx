import { useEffect, useMemo } from "react";
import { useToolCalls } from "~/core/store";
import { cn } from "~/lib/utils";
import { Tooltip } from "./tooltip";

export const Link = ({
  href,
  children,
}: {
  href: string | undefined;
  children: React.ReactNode;
}) => {
  const toolCalls = useToolCalls();
  const credibleLinks = useMemo(() => {
    const links = new Set<string>();
    (toolCalls || []).forEach((call) => {
      if (call && call.name === "web_search" && call.result) {
        const result = JSON.parse(call.result) as Array<{ url: string }>;
        result.forEach((r) => {
          links.add(r.url);
        });
      }
    });
    return links;
  }, [toolCalls]);
  const isCredible = useMemo(() => {
    return href ? credibleLinks.has(href) : true;
  }, [credibleLinks, href]);

  if (!isCredible) {
    return (
      <Tooltip title="This link might be a hallucination from AI model and may not be reliable.">
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className={cn(isCredible && "after:ml-0.5 after:content-['⚠️']")}
        >
          {children}
        </a>
      </Tooltip>
    );
  }

  return (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  );
};
