import logging
import json
from typing import Literal, Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.llms.llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.prompts.template import apply_prompt_template
from src.prompts.planner_model import Plan, StepType
from src.utils.json_utils import repair_json_output
from src.agents.agents import research_agent, coder_agent
from .types import State

logger = logging.getLogger(__name__)


@tool
def handoff_to_planner(
    task_title: Annotated[str, "The title of the task to be handoffed."],
):
    """Handoff to planner agent to do plan."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to planner agent
    return


def planner_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team", "reporter", "__end__"]]:
    """Planner node that generate the full plan."""
    logger.info("Planner generating full plan")
    configurable = Configuration.from_runnable_config(config)
    messages = apply_prompt_template("planner", state, configurable)
    if AGENT_LLM_MAP["planner"] == "basic":
        llm = get_llm_by_type(AGENT_LLM_MAP["planner"]).with_structured_output(
            Plan, method="json_mode"
        )
    else:
        llm = get_llm_by_type(AGENT_LLM_MAP["planner"])
    current_plan = state.get("current_plan", None)
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0

    # if the plan iterations is greater than the max plan iterations, return the reporter node
    if plan_iterations >= configurable.max_plan_iterations:
        return Command(goto="reporter")

    full_response = ""
    if AGENT_LLM_MAP["planner"] == "basic":
        response = llm.invoke(messages)
        full_response = response.model_dump_json(indent=4, exclude_none=True)
    else:
        response = llm.stream(messages)
        for chunk in response:
            full_response += chunk.content
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"Planner response: {full_response}")

    goto = "research_team"
    try:
        full_response = repair_json_output(full_response)
        # increment the plan iterations
        plan_iterations += 1
        # parse the plan
        new_plan = json.loads(full_response)
        if new_plan["has_enough_context"]:
            goto = "reporter"
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        if plan_iterations > 0:
            return Command(goto="reporter")
        else:
            return Command(goto="__end__")

    return Command(
        update={
            "messages": [HumanMessage(content=full_response, name="planner")],
            "last_plan": current_plan,
            "current_plan": Plan.model_validate(new_plan),
            "plan_iterations": plan_iterations,
        },
        goto=goto,
    )


def coordinator_node(state: State) -> Command[Literal["planner", "__end__"]]:
    """Coordinator node that communicate with customers."""
    logger.info("Coordinator talking.")
    messages = apply_prompt_template("coordinator", state)
    response = (
        get_llm_by_type(AGENT_LLM_MAP["coordinator"])
        .bind_tools([handoff_to_planner])
        .invoke(messages)
    )
    logger.debug(f"Current state messages: {state['messages']}")

    goto = "__end__"
    if len(response.tool_calls) > 0:
        goto = "planner"

    return Command(
        goto=goto,
    )


def reporter_node(state: State):
    """Reporter node that write a final report."""
    logger.info("Reporter write final report")
    messages = apply_prompt_template("reporter", state)
    observations = state.get("observations", [])
    invoke_messages = messages[:2]
    for observation in observations:
        invoke_messages.append(
            HumanMessage(
                content=f"Below is some observations for the user query:\n\n{observation}",
                name="observation",
            )
        )
    logger.debug(f"Current invoke messages: {invoke_messages}")
    response = get_llm_by_type(AGENT_LLM_MAP["reporter"]).invoke(invoke_messages)
    response_content = response.content
    logger.info(f"reporter response: {response_content}")

    return {"final_report": response_content}


def research_team_node(
    state: State,
) -> Command[Literal["planner", "researcher", "coder"]]:
    """Research team node that collaborates on tasks."""
    logger.info("Research team is collaborating on tasks.")
    current_plan = state.get("current_plan")
    if not current_plan or not current_plan.steps:
        return Command(goto="planner")
    if all(step.execution_res for step in current_plan.steps):
        return Command(goto="planner")
    for step in current_plan.steps:
        if not step.execution_res:
            break
    if step.step_type and step.step_type == StepType.RESEARCH:
        return Command(goto="researcher")
    if step.step_type and step.step_type == StepType.PROCESSING:
        return Command(goto="coder")
    return Command(goto="planner")


def _execute_agent_step(
    state: State, agent, agent_name: str
) -> Command[Literal["research_team"]]:
    """Helper function to execute a step using the specified agent."""
    current_plan = state.get("current_plan")

    # Find the first unexecuted step
    for step in current_plan.steps:
        if not step.execution_res:
            break

    logger.info(f"Executing step: {step.title}")

    # Prepare the input for the agent
    agent_input = {
        "messages": [
            HumanMessage(
                content=f"#Task\n\n##title: {step.title}\n\n##description: {step.description}"
            )
        ]
    }

    # Invoke the agent
    result = agent.invoke(input=agent_input)

    # Process the result
    response_content = result["messages"][-1].content
    logger.debug(f"{agent_name.capitalize()} full response: {response_content}")

    # Update the step with the execution result
    step.execution_res = response_content
    logger.info(f"Step '{step.title}' execution completed by {agent_name}")

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response_content,
                    name=agent_name,
                )
            ],
            "observations": [response_content],
        },
        goto="research_team",
    )


def researcher_node(state: State) -> Command[Literal["research_team"]]:
    """Researcher node that do research"""
    logger.info("Researcher node is researching.")
    return _execute_agent_step(state, research_agent, "researcher")


def coder_node(state: State) -> Command[Literal["research_team"]]:
    """Coder node that do code analysis."""
    logger.info("Coder node is coding.")
    return _execute_agent_step(state, coder_agent, "coder")
