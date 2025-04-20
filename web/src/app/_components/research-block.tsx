// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { X } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "~/components/ui/button";
import { Card } from "~/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "~/components/ui/tabs";
import { openResearch, useStore } from "~/core/store";
import { cn } from "~/lib/utils";

import { ResearchActivitiesBlock } from "./research-activities-block";
import { ResearchReportBlock } from "./research-report-block";
import { ScrollContainer } from "./scroll-container";
import { Tooltip } from "./tooltip";

export function ResearchBlock({
  className,
  researchId = null,
}: {
  className?: string;
  researchId: string | null;
}) {
  const reportId = useStore((state) =>
    researchId ? state.researchReportIds.get(researchId) : undefined,
  );
  const [activeTab, setActiveTab] = useState("activities");
  const hasReport = useStore((state) =>
    researchId ? state.researchReportIds.has(researchId) : false,
  );
  useEffect(() => {
    if (hasReport) {
      setActiveTab("report");
    }
  }, [hasReport]);

  return (
    <div className={cn("h-full w-full", className)}>
      <Card className={cn("relative h-full w-full pt-4", className)}>
        <div className="absolute right-4 flex h-9 items-center">
          <Tooltip title="Close">
            <Button
              className="text-gray-400"
              size="sm"
              variant="ghost"
              onClick={() => {
                openResearch(null);
              }}
            >
              <X />
            </Button>
          </Tooltip>
        </div>
        <Tabs
          className="flex h-full w-full flex-col"
          value={activeTab}
          onValueChange={(value) => setActiveTab(value)}
        >
          <div className="flex w-full justify-center">
            <TabsList className="">
              <TabsTrigger
                className="px-8"
                value="report"
                disabled={!hasReport}
              >
                Report
              </TabsTrigger>
              <TabsTrigger className="px-8" value="activities">
                Activities
              </TabsTrigger>
            </TabsList>
          </div>
          <TabsContent className="h-full min-h-0 flex-grow px-8" value="report">
            <ScrollContainer
              className="px-5pb-20 h-full"
              scrollShadowColor="var(--card)"
            >
              {reportId && researchId && (
                <ResearchReportBlock
                  className="mt-4"
                  researchId={researchId}
                  messageId={reportId}
                />
              )}
            </ScrollContainer>
          </TabsContent>
          <TabsContent
            className="h-full min-h-0 flex-grow px-8"
            value="activities"
          >
            <ScrollContainer className="h-full" scrollShadowColor="var(--card)">
              {researchId && (
                <ResearchActivitiesBlock
                  className="mt-4"
                  researchId={researchId}
                />
              )}
            </ScrollContainer>
          </TabsContent>
        </Tabs>
      </Card>
    </div>
  );
}
