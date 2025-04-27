# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging

from langchain.schema import HumanMessage, SystemMessage

from src.config.agents import AGENT_LLM_MAP
from src.llms.llm import get_llm_by_type
from src.prose.graph.state import ProseState

logger = logging.getLogger(__name__)

prompt = """
You are an AI writing assistant that fixes grammar and spelling errors in existing text. 
- Limit your response to no more than 200 characters, but make sure to construct complete sentences.
- Use Markdown formatting when appropriate.
- If the text is already correct, just return the original text.
"""


def prose_fix_node(state: ProseState):
    logger.info("Generating prose fix content...")
    model = get_llm_by_type(AGENT_LLM_MAP["prose_writer"])
    prose_content = model.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=f"The existing text is: {state['content']}"),
        ],
    )
    logger.info(f"prose_content: {prose_content}")
    return {"output": prose_content.content}
