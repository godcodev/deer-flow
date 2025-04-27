# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging

from langchain.schema import HumanMessage, SystemMessage

from src.config.agents import AGENT_LLM_MAP
from src.llms.llm import get_llm_by_type
from src.prose.graph.state import ProseState

logger = logging.getLogger(__name__)

prompt = """
You are an AI writing assistant that continues existing text based on context from prior text.
- Give more weight/priority to the later characters than the beginning ones.
- Limit your response to no more than 200 characters, but make sure to construct complete sentences.
- Use Markdown formatting when appropriate
"""


def prose_continue_node(state: ProseState):
    logger.info("Generating prose continue content...")
    model = get_llm_by_type(AGENT_LLM_MAP["prose_writer"])
    prose_content = model.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=state["content"]),
        ],
    )
    return {"output": prose_content.content}
