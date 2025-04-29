// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

"use client";
import {
  ReactFlow,
  Background,
  useNodesState,
  useEdgesState,
  Handle,
  Position,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  Brain,
  FilePen,
  MessageSquareQuote,
  Microscope,
  SquareTerminal,
  UserCheck,
  Users,
  type LucideIcon,
} from "lucide-react";

import { ShineBorder } from "~/components/magicui/shine-border";

const WORKFLOW_STEPS = [
  {
    description:
      "The Coordinator is responsible for engaging with the user to understand their problem and requirements.",
    activeNodes: ["Start", "Coordinator"],
    activeEdges: ["Start->Coordinator"],
  },
  {
    description:
      "If the user's problem is clearly defined, the Coordinator will hand it over to the Planner.",
    activeNodes: ["Coordinator", "Planner"],
    activeEdges: ["Coordinator->Planner"],
  },
  {
    description: "Awaiting human feedback to refine the plan.",
    activeNodes: ["Planner", "HumanFeedback"],
    activeEdges: ["Planner->HumanFeedback"],
  },
  {
    description: "Updating the plan based on human feedback.",
    activeNodes: ["Planner", "HumanFeedback"],
    activeEdges: ["HumanFeedback->Planner"],
  },
  {
    description:
      "The Research Team is responsible for conducting the core research tasks.",
    activeNodes: ["HumanFeedback", "ResearchTeam"],
    activeEdges: ["HumanFeedback->ResearchTeam", "ResearchTeam->HumanFeedback"],
  },
  {
    description:
      "The Researcher is responsible for gathering information using search and crawling tools.",
    activeNodes: ["ResearchTeam", "Researcher"],
    activeEdges: ["ResearchTeam->Researcher", "Researcher->ResearchTeam"],
  },
  {
    description:
      "The Coder is responsible for writing and executing Python code to solve problems such as mathematical computations, data analysis, and more.",
    activeNodes: ["ResearchTeam", "Coder"],
    activeEdges: ["ResearchTeam->Coder", "Coder->ResearchTeam"],
  },
  {
    description:
      "Once the research tasks are completed, the Researcher will hand over to the Planner.",
    activeNodes: ["ResearchTeam", "Planner"],
    activeEdges: ["ResearchTeam->Planner"],
  },
  {
    description:
      "If no additional information is required, the Planner will handoff to the Reporter.",
    activeNodes: ["Planner", "Reporter"],
    activeEdges: ["Planner->Reporter"],
  },
  {
    description: "The Reporter will prepare a report summarizing the results.",
    activeNodes: ["Reporter", "End"],
    activeEdges: ["Reporter->End"],
  },
];

const ROW_HEIGHT = 75;
const ROW_1 = 0;
const ROW_2 = ROW_HEIGHT;
const ROW_3 = ROW_HEIGHT * 2;
const ROW_4 = ROW_HEIGHT * 2;
const ROW_5 = ROW_HEIGHT * 3;
const ROW_6 = ROW_HEIGHT * 4;
const initialNodes = [
  {
    id: "Start",
    type: "circle",
    data: { label: "Start" },
    position: { x: 25, y: ROW_1 },
  },
  {
    id: "Coordinator",
    data: { icon: MessageSquareQuote, label: "Coordinator" },
    position: { x: 150, y: ROW_1 },
  },
  {
    id: "Planner",
    data: { icon: Brain, label: "Planner" },
    position: { x: 150, y: ROW_2 },
  },
  {
    id: "Reporter",
    data: { icon: FilePen, label: "Reporter" },
    position: { x: 275, y: ROW_3 },
  },
  {
    id: "HumanFeedback",
    data: { icon: UserCheck, label: "Human Feedback" },
    position: { x: 25, y: ROW_4 },
  },
  {
    id: "ResearchTeam",
    data: { icon: Users, label: "Research Team" },
    position: { x: 25, y: ROW_5 },
  },
  {
    id: "Researcher",
    data: { icon: Microscope, label: "Researcher" },
    position: { x: -75, y: ROW_6 },
  },
  {
    id: "Coder",
    data: { icon: SquareTerminal, label: "Coder" },
    position: { x: 125, y: ROW_6 },
  },
  {
    id: "End",
    type: "circle",
    data: { label: "End" },
    position: { x: 330, y: ROW_6 },
  },
];

const initialEdges = [
  {
    id: "Start->Coordinator",
    source: "Start",
    target: "Coordinator",
    sourceHandle: "right",
    targetHandle: "left",
    animated: true,
  },
  {
    id: "Coordinator->Planner",
    source: "Coordinator",
    target: "Planner",
    sourceHandle: "bottom",
    targetHandle: "top",
    animated: true,
  },
  {
    id: "Planner->Reporter",
    source: "Planner",
    target: "Reporter",
    sourceHandle: "right",
    targetHandle: "top",
    animated: true,
  },
  {
    id: "Planner->HumanFeedback",
    source: "Planner",
    target: "HumanFeedback",
    sourceHandle: "left",
    targetHandle: "top",
    animated: true,
  },
  {
    id: "HumanFeedback->Planner",
    source: "HumanFeedback",
    target: "Planner",
    sourceHandle: "right",
    targetHandle: "bottom",
    animated: true,
  },
  {
    id: "HumanFeedback->ResearchTeam",
    source: "HumanFeedback",
    target: "ResearchTeam",
    sourceHandle: "bottom",
    targetHandle: "top",
    animated: true,
  },
  {
    id: "Reporter->End",
    source: "Reporter",
    target: "End",
    sourceHandle: "bottom",
    targetHandle: "top",
    animated: true,
  },
  {
    id: "ResearchTeam->Researcher",
    source: "ResearchTeam",
    target: "Researcher",
    sourceHandle: "left",
    targetHandle: "top",
    animated: true,
  },
  {
    id: "ResearchTeam->Coder",
    source: "ResearchTeam",
    target: "Coder",
    sourceHandle: "bottom",
    targetHandle: "left",
    animated: true,
  },
  {
    id: "ResearchTeam->Planner",
    source: "ResearchTeam",
    target: "Planner",
    sourceHandle: "right",
    targetHandle: "bottom",
    animated: true,
  },
  {
    id: "Researcher->ResearchTeam",
    source: "Researcher",
    target: "ResearchTeam",
    sourceHandle: "right",
    targetHandle: "bottom",
    animated: true,
  },
  {
    id: "Coder->ResearchTeam",
    source: "Coder",
    target: "ResearchTeam",
    sourceHandle: "top",
    targetHandle: "right",
    animated: true,
  },
];

const nodeTypes = {
  circle: CircleNode,
  agent: AgentNode,
  default: AgentNode,
};

export function MultiAgentVisualization() {
  const [nodes, setNodes] = useNodesState(initialNodes);
  const [edges, setEdges] = useEdgesState(initialEdges);

  return (
    <ReactFlow
      style={{
        ["--xy-background-color-default" as string]: "transparent",
      }}
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      fitView
      attributionPosition="top-right"
      colorMode="dark"
      panOnScroll={false}
      zoomOnScroll={false}
      preventScrolling={false}
      panOnDrag={false}
    >
      <Background
        className="[mask-image:radial-gradient(800px_circle_at_center,white,transparent)]"
        bgColor="var(--background)"
      />
    </ReactFlow>
  );
}

function CircleNode({ data }: { data: { label: string; active: boolean } }) {
  return (
    <>
      {data.active && (
        <ShineBorder
          className="rounded-full"
          shineColor={["#A07CFE", "#FE8FB5", "#FFBE7B"]}
        />
      )}
      <div className="flex h-10 w-10 items-center justify-center rounded-full border bg-[var(--xy-node-background-color-default)]">
        <p className="text-xs">{data.label}</p>
      </div>
      {data.label === "Start" && (
        <Handle
          className="invisible"
          type="source"
          position={Position.Right}
          id="right"
        />
      )}
      {data.label === "End" && (
        <Handle
          className="invisible"
          type="target"
          position={Position.Top}
          id="top"
        />
      )}
    </>
  );
}

function AgentNode({
  data,
  id,
}: {
  data: { icon?: LucideIcon; label: string; active: boolean };
  id: string;
}) {
  return (
    <>
      {data.active && (
        <ShineBorder
          shineColor={["#A07CFE", "#FE8FB5", "#FFBE7B"]}
          className="rounded-[2px]"
        />
      )}
      <div
        id={id}
        className="relative flex w-full items-center justify-center text-xs"
      >
        <div className="flex items-center gap-2">
          {data.icon && <data.icon className="h-[1rem] w-[1rem]" />}
          <span>{data.label}</span>
        </div>
      </div>
      <Handle
        className="invisible"
        type="source"
        position={Position.Left}
        id="left"
      />
      <Handle
        className="invisible"
        type="source"
        position={Position.Right}
        id="right"
      />
      <Handle
        className="invisible"
        type="source"
        position={Position.Top}
        id="top"
      />
      <Handle
        className="invisible"
        type="source"
        position={Position.Bottom}
        id="bottom"
      />
      <Handle
        className="invisible"
        type="target"
        position={Position.Left}
        id="left"
      />
      <Handle
        className="invisible"
        type="target"
        position={Position.Right}
        id="right"
      />
      <Handle
        className="invisible"
        type="target"
        position={Position.Top}
        id="top"
      />
      <Handle
        className="invisible"
        type="target"
        position={Position.Bottom}
        id="bottom"
      />
    </>
  );
}
