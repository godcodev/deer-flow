# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging

from langchain.schema import HumanMessage, SystemMessage

from src.config.agents import AGENT_LLM_MAP
from src.llms.llm import get_llm_by_type
from src.prose.graph.state import ProseState

logger = logging.getLogger(__name__)


def prose_zap_node(state: ProseState):
    logger.info("Generating prose zap content...")
    model = get_llm_by_type(AGENT_LLM_MAP["prose_writer"])
    prose_content = model.invoke(
        [
            SystemMessage(
                content="""
You area an AI writing assistant that generates text based on a prompt. 
- You take an input from the user and a command for manipulating the text."
- Use Markdown formatting when appropriate.
"""
            ),
            HumanMessage(
                content=f"For this text: {state['content']}.\nYou have to respect the command: {state['command']}"
            ),
        ],
    )
    logger.info(f"prose_content: {prose_content}")
    return {"output": prose_content.content}
