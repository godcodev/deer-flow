import operator

from langgraph.graph import MessagesState
from typing import Annotated
from src.prompts.planner_model import Plan


class State(MessagesState):
    """State for the agent system, extends MessagesState with next field."""

    # Runtime Variables
    observations: Annotated[list[str], operator.add] = []
    plan_iterations: int = 0
    last_plan: Plan = None
    current_plan: Plan = None
    final_report: str = ""
