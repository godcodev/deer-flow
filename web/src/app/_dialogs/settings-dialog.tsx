import { BadgeInfo, Blocks, Settings } from "lucide-react";
import { useState } from "react";

import { Button } from "~/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "~/components/ui/dialog";
import { Tabs, TabsContent } from "~/components/ui/tabs";
import { cn } from "~/lib/utils";

import { Markdown } from "../_components/markdown";
import { Tooltip } from "../_components/tooltip";

import about from "./about.md";

const SETTINGS_TABS = [
  {
    id: "general",
    label: "General",
    icon: Settings,
  },
  {
    id: "mcp",
    label: "MCP",
    icon: Blocks,
  },
  {
    id: "about",
    label: "About",
    icon: BadgeInfo,
  },
];
export function SettingsDialog() {
  const [activeTabId, setActiveTabId] = useState(SETTINGS_TABS[0]!.id);
  return (
    <Dialog>
      <Tooltip title="Settings">
        <DialogTrigger asChild>
          <Button variant="ghost" size="icon">
            <Settings />
          </Button>
        </DialogTrigger>
      </Tooltip>
      <DialogContent className="sm:max-w-[800px]">
        <DialogHeader>
          <DialogTitle>DeerFlow Settings</DialogTitle>
          <DialogDescription>
            Manage your DeerFlow settings here.
          </DialogDescription>
        </DialogHeader>
        <Tabs value={activeTabId}>
          <div className="flex h-100 w-full overflow-auto border-y">
            <ul className="flex w-60 shrink-0 border-r p-1">
              <div className="size-full">
                {SETTINGS_TABS.map((tab) => (
                  <li
                    key={tab.id}
                    className={cn(
                      "hover:accent-foreground hover:bg-accent mb-1 flex h-8 w-full cursor-pointer items-center gap-1.5 rounded px-2",
                      activeTabId === tab.id &&
                        "!bg-primary !text-primary-foreground",
                    )}
                    onClick={() => setActiveTabId(tab.id)}
                  >
                    <tab.icon size={16} />
                    <span>{tab.label}</span>
                  </li>
                ))}
              </div>
            </ul>
            <div className="min-w-0 flex-grow">
              <div className="size-full overflow-auto p-4">
                <TabsContent value="general">
                  <div>General</div>
                </TabsContent>
                <TabsContent value="mcp">
                  <div>Coming soon...</div>
                </TabsContent>
                <TabsContent value="about">
                  <Markdown>{about}</Markdown>
                </TabsContent>
              </div>
            </div>
          </div>
        </Tabs>
        <DialogFooter>
          <Button type="submit">Save Settings</Button>
          <Button variant="outline">Cancel</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
