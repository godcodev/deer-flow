import { BadgeInfo, Blocks, Settings } from "lucide-react";

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

import { ScrollContainer } from "../_components/scroll-container";
import { Tooltip } from "../_components/tooltip";

export function SettingsDialog() {
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
        <div className="flex h-100 w-full border-y">
          <div className="flex w-60 border-r">
            <ScrollContainer className="size-full p-2" scrollShadow={false}>
              <ul className="flex flex-col gap-2">
                <li className="flex h-8 items-center gap-1.5">
                  <Settings size={16} />
                  <span>General</span>
                </li>
                <li className="flex items-center gap-1.5">
                  <Blocks size={16} />
                  <span>MCP</span>
                </li>
                <li className="flex items-center gap-1.5">
                  <BadgeInfo size={16} />
                  <span>About</span>
                </li>
              </ul>
            </ScrollContainer>
          </div>
          <div className="min-h-0 flex-grow">
            <ScrollContainer
              className="size-full"
              scrollShadow={false}
            ></ScrollContainer>
          </div>
        </div>
        <DialogFooter>
          <Button type="submit">Save Settings</Button>
          <Button variant="outline">Cancel</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
