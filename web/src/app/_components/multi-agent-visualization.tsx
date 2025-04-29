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
    position: { x: 80, y: ROW_1 },
  },
  {
    id: "Coordinator",
    data: { label: "Coordinator" },
    position: { x: 150, y: ROW_1 },
  },
  {
    id: "Planner",
    data: { label: "Planner" },
    position: { x: 150, y: ROW_2 },
  },
  {
    id: "Reporter",
    data: { label: "Reporter" },
    position: { x: 25, y: ROW_3 },
  },
  {
    id: "Human Feedback",
    data: { label: "Human Feedback" },
    position: { x: 275, y: ROW_4 },
  },
  {
    id: "Research Team",
    data: { label: "Research Team" },
    position: { x: 275, y: ROW_5 },
  },
  {
    id: "Researcher",
    data: { label: "Researcher" },
    position: { x: 175, y: ROW_6 },
  },
  {
    id: "Coder",
    data: { label: "Coder" },
    position: { x: 375, y: ROW_6 },
  },
  {
    id: "End",
    type: "circle",
    data: { label: "End" },
    position: { x: 80, y: ROW_6 },
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
    sourceHandle: "left",
    targetHandle: "top",
    animated: true,
  },
  {
    id: "Planner->Human Feedback",
    source: "Planner",
    target: "Human Feedback",
    sourceHandle: "bottom",
    targetHandle: "left",
    animated: true,
  },
  {
    id: "Human Feedback->Planner",
    source: "Human Feedback",
    target: "Planner",
    sourceHandle: "top",
    targetHandle: "right",
    animated: true,
  },
  {
    id: "Human Feedback->Research Team",
    source: "Human Feedback",
    target: "Research Team",
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
    id: "Research Team->Researcher",
    source: "Research Team",
    target: "Researcher",
    sourceHandle: "left",
    targetHandle: "top",
    animated: true,
  },
  {
    id: "Research Team->Coder",
    source: "Research Team",
    target: "Coder",
    sourceHandle: "bottom",
    targetHandle: "left",
    animated: true,
  },
  {
    id: "Research Team->Planner",
    source: "Research Team",
    target: "Planner",
    sourceHandle: "left",
    targetHandle: "bottom",
    animated: true,
  },
  {
    id: "Researcher->Research Team",
    source: "Researcher",
    target: "Research Team",
    sourceHandle: "right",
    targetHandle: "bottom",
    animated: true,
  },
  {
    id: "Coder->Research Team",
    source: "Coder",
    target: "Research Team",
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
  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  return (
    <ReactFlow
      style={{
        ["--xy-background-color-default" as string]: "transparent",
      }}
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      fitView
      attributionPosition="top-right"
      colorMode="dark"
      panOnScroll={false}
      zoomOnScroll={false}
      preventScrolling={false}
    >
      <Background
        className="[mask-image:radial-gradient(800px_circle_at_center,white,transparent)]"
        bgColor="var(--background)"
      />
    </ReactFlow>
  );
}

function CircleNode({ data }: { data: { label: string } }) {
  return (
    <>
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

function AgentNode({ data }: { data: { label: string } }) {
  return (
    <>
      <div className="text-xs">{data.label}</div>
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
