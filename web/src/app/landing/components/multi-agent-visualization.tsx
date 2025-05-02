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
  type Node,
  type Edge,
} from "@xyflow/react";
import {
  Brain,
  FilePen,
  MessageSquareQuote,
  Microscope,
  RotateCcw,
  SquareTerminal,
  UserCheck,
  Users,
  type LucideIcon,
} from "lucide-react";
import {
  useCallback,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import "@xyflow/react/dist/style.css";

import { Tooltip } from "~/components/deer-flow/tooltip";
import { ShineBorder } from "~/components/magicui/shine-border";
import { Button } from "~/components/ui/button";
import { useIntersectionObserver } from "~/hooks/use-intersection-observer";

const ROW_HEIGHT = 85;
const ROW_1 = 0;
const ROW_2 = ROW_HEIGHT;
const ROW_3 = ROW_HEIGHT * 2;
const ROW_4 = ROW_HEIGHT * 2;
const ROW_5 = ROW_HEIGHT * 3;
const ROW_6 = ROW_HEIGHT * 4;

export type WorkflowNode = Node<{
  label: string;
  icon?: LucideIcon;
  active?: boolean;
}>;

const initialNodes: WorkflowNode[] = [
  {
    id: "Start",
    type: "circle",
    data: { label: "Start" },
    position: { x: -75, y: ROW_1 },
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

const initialEdges: Edge[] = [
  {
    id: "Start->Coordinator",
    source: "Start",
    target: "Coordinator",
    sourceHandle: "right",
    targetHandle: "left",
    animated: false,
  },
  {
    id: "Coordinator->Planner",
    source: "Coordinator",
    target: "Planner",
    sourceHandle: "bottom",
    targetHandle: "top",
    animated: false,
  },
  {
    id: "Planner->Reporter",
    source: "Planner",
    target: "Reporter",
    sourceHandle: "right",
    targetHandle: "top",
    animated: false,
  },
  {
    id: "Planner->HumanFeedback",
    source: "Planner",
    target: "HumanFeedback",
    sourceHandle: "left",
    targetHandle: "top",
    animated: false,
  },
  {
    id: "HumanFeedback->Planner",
    source: "HumanFeedback",
    target: "Planner",
    sourceHandle: "right",
    targetHandle: "bottom",
    animated: false,
  },
  {
    id: "HumanFeedback->ResearchTeam",
    source: "HumanFeedback",
    target: "ResearchTeam",
    sourceHandle: "bottom",
    targetHandle: "top",
    animated: false,
  },
  {
    id: "Reporter->End",
    source: "Reporter",
    target: "End",
    sourceHandle: "bottom",
    targetHandle: "top",
    animated: false,
  },
  {
    id: "ResearchTeam->Researcher",
    source: "ResearchTeam",
    target: "Researcher",
    sourceHandle: "left",
    targetHandle: "top",
    animated: false,
  },
  {
    id: "ResearchTeam->Coder",
    source: "ResearchTeam",
    target: "Coder",
    sourceHandle: "bottom",
    targetHandle: "left",
    animated: false,
  },
  {
    id: "ResearchTeam->Planner",
    source: "ResearchTeam",
    target: "Planner",
    sourceHandle: "right",
    targetHandle: "bottom",
    animated: false,
  },
  {
    id: "Researcher->ResearchTeam",
    source: "Researcher",
    target: "ResearchTeam",
    sourceHandle: "right",
    targetHandle: "bottom",
    animated: false,
  },
  {
    id: "Coder->ResearchTeam",
    source: "Coder",
    target: "ResearchTeam",
    sourceHandle: "top",
    targetHandle: "right",
    animated: false,
  },
];

const nodeTypes = {
  circle: CircleNode,
  agent: AgentNode,
  default: AgentNode,
};

const WORKFLOW_STEPS = [
  {
    description:
      "The Coordinator is responsible for engaging with the user to understand their problem and requirements.",
    tooltipPosition: "right",
    activeNodes: ["Start", "Coordinator"],
    activeEdges: ["Start->Coordinator"],
  },
  {
    description:
      "If the user's problem is clearly defined, the Coordinator will hand it over to the Planner.",
    tooltipPosition: "left",
    activeNodes: ["Coordinator", "Planner"],
    activeEdges: ["Coordinator->Planner"],
  },
  {
    description: "Awaiting human feedback to refine the plan.",
    tooltipPosition: "left",
    activeNodes: ["Planner", "HumanFeedback"],
    activeEdges: ["Planner->HumanFeedback"],
  },
  {
    description: "Updating the plan based on human feedback.",
    tooltipPosition: "left",
    activeNodes: ["HumanFeedback", "Planner"],
    activeEdges: ["HumanFeedback->Planner"],
  },
  {
    description:
      "The Research Team is responsible for conducting the core research tasks.",
    tooltipPosition: "left",
    activeNodes: ["HumanFeedback", "ResearchTeam"],
    activeEdges: ["HumanFeedback->ResearchTeam", "ResearchTeam->HumanFeedback"],
  },
  {
    description:
      "The Researcher is responsible for gathering information using search and crawling tools.",
    tooltipPosition: "left",
    activeNodes: ["ResearchTeam", "Researcher"],
    activeEdges: ["ResearchTeam->Researcher", "Researcher->ResearchTeam"],
  },
  {
    description:
      "The Coder is responsible for writing Python code to solve math problems, data analysis, and more.",
    tooltipPosition: "right",
    activeNodes: ["ResearchTeam", "Coder"],
    activeEdges: ["ResearchTeam->Coder", "Coder->ResearchTeam"],
  },
  {
    description:
      "Once the research tasks are completed, the Researcher will hand over to the Planner.",
    tooltipPosition: "left",
    activeNodes: ["ResearchTeam", "Planner"],
    activeEdges: ["ResearchTeam->Planner"],
  },
  {
    description:
      "If no additional information is required, the Planner will handoff to the Reporter.",
    tooltipPosition: "right",
    activeNodes: ["Reporter", "Planner"],
    activeEdges: ["Planner->Reporter"],
  },
  {
    description: "The Reporter will prepare a report summarizing the results.",
    tooltipPosition: "bottom",
    activeNodes: ["End", "Reporter"],
    activeEdges: ["Reporter->End"],
  },
];

function useWorkflowRun(
  setNodes: Dispatch<SetStateAction<WorkflowNode[]>>,
  setEdges: Dispatch<SetStateAction<Edge[]>>,
) {
  const [isRunning, setIsRunning] = useState(false);

  const clearAnimation = useCallback(() => {
    setEdges((edges) => {
      return edges.map((edge) => ({
        ...edge,
        animated: true,
      }));
    });

    setNodes((nodes) => {
      return nodes.map((node) => ({
        ...node,
        data: { ...node.data, active: false },
      }));
    });
  }, [setEdges, setNodes]);

  const run = useCallback(async () => {
    setIsRunning(true);
    clearAnimation();
    for (const step of WORKFLOW_STEPS) {
      setNodes((nodes) => {
        return nodes.map((node) => ({
          ...node,
          data: {
            ...node.data,
            active: step.activeNodes.includes(node.id),
            stepDescription:
              step.activeNodes.indexOf(node.id) === step.activeNodes.length - 1
                ? step.description
                : undefined,
            stepTooltipPosition: step.tooltipPosition,
          },
        }));
      });

      setEdges((edges) => {
        return edges.map((edge) => ({
          ...edge,
          animated: step.activeEdges.includes(edge.id),
        }));
      });

      await sleep(step.description.split(" ").length * 360);
    }
    clearAnimation();
    setIsRunning(false);
  }, [setNodes, setEdges, clearAnimation]);

  return { run, isRunning };
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function MultiAgentVisualization() {
  const [nodes, setNodes] = useNodesState(initialNodes);
  const [edges, setEdges] = useEdgesState(initialEdges);

  const { run, isRunning } = useWorkflowRun(setNodes, setEdges);
  const [hasAutoRun, setHasAutoRun] = useState(false);

  const { ref } = useIntersectionObserver({
    rootMargin: "0%",
    threshold: 0,
    onChange: (isIntersecting) => {
      if (isIntersecting && !hasAutoRun) {
        setHasAutoRun(true);
        void run();
      }
    },
  });

  return (
    <ReactFlow
      style={{
        ["--xy-background-color-default" as string]: "transparent",
      }}
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      fitView
      proOptions={{ hideAttribution: true }}
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
      <div
        id="auto-run-animation-trigger"
        ref={ref}
        className="absolute bottom-0 left-[50%] h-px w-px"
      />
      <div className="absolute top-0 right-0 left-0 z-200 flex items-center justify-center">
        {!isRunning && (
          <Button
            className="translate-x-[22vw]"
            variant="outline"
            size="sm"
            onClick={() => {
              void run();
            }}
          >
            <RotateCcw />
            Replay
          </Button>
        )}
      </div>
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
  data: {
    icon?: LucideIcon;
    label: string;
    active: boolean;
    stepDescription?: string;
    stepTooltipPosition?: "left" | "right" | "top" | "bottom";
  };
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
      <Tooltip
        className="max-w-50 text-[15px] opacity-70"
        style={{
          ["--primary" as string]: "#333",
          ["--primary-foreground" as string]: "white",
        }}
        open={data.active && !!data.stepDescription}
        title={data.stepDescription}
        side={data.stepTooltipPosition}
        sideOffset={20}
      >
        <div
          id={id}
          className="relative flex w-full items-center justify-center text-xs"
        >
          <div className="flex items-center gap-2">
            {data.icon && <data.icon className="h-[1rem] w-[1rem]" />}
            <span>{data.label}</span>
          </div>
        </div>
      </Tooltip>
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
