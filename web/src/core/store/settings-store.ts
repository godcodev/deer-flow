// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { create } from "zustand";

import type { MCPServerMetadata } from "../mcp";

const SETTINGS_KEY = "deerflow.settings";

const DEFAULT_SETTINGS: SettingsState = {
  general: {
    autoAcceptedPlan: false,
    maxPlanIterations: 1,
    maxStepNum: 3,
  },
  mcp: {
    servers: [
      {
        enabled: true,
        name: "Zapier",
        transport: "sse",
        url: "https://actions.zapier.com/mcp/sk-ak-OnJ4kVKzxcLjpvpLChkT7RCYuh/sse",
        env: { API_KEY: "123" },
        createdAt: new Date("2025-04-20").valueOf(),
        updatedAt: new Date("2025-04-20").valueOf(),
        tools: [
          {
            name: "youtube_get_report",
            description:
              "Creates a report on specified data from your owned and managed channels.",
          },
          {
            name: "edit_actions",
            description: "Edit your existing MCP provider actions",
          },
          {
            name: "add_actions",
            description: "Add new actions to your MCP provider",
          },
        ],
      },
    ],
  },
};

export type SettingsState = {
  general: {
    autoAcceptedPlan: boolean;
    maxPlanIterations: number;
    maxStepNum: number;
  };
  mcp: {
    servers: MCPServerMetadata[];
  };
};

export const useSettingsStore = create<SettingsState>(() => ({
  ...DEFAULT_SETTINGS,
}));

export const useSettings = (key: keyof SettingsState) => {
  return useSettingsStore((state) => state[key]);
};

export const changeSettings = (settings: SettingsState) => {
  useSettingsStore.setState(settings);
};

export const loadSettings = () => {
  if (typeof window === "undefined") {
    return;
  }
  const json = localStorage.getItem(SETTINGS_KEY);
  if (json) {
    const settings = JSON.parse(json);
    for (const key in DEFAULT_SETTINGS) {
      if (!(key in settings)) {
        settings[key] = DEFAULT_SETTINGS[key as keyof SettingsState];
      }
    }

    try {
      useSettingsStore.setState(settings);
    } catch (error) {
      console.error(error);
    }
  }
};

export const saveSettings = () => {
  const latestSettings = useSettingsStore.getState();
  const json = JSON.stringify(latestSettings);
  localStorage.setItem(SETTINGS_KEY, json);
};

loadSettings();
