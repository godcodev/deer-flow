from typing import List, Optional, Union

from pydantic import BaseModel, Field


class ContentItem(BaseModel):
    type: str = Field(..., description="The type of content (text, image, etc.)")
    text: Optional[str] = Field(None, description="The text content if type is 'text'")
    image_url: Optional[str] = Field(
        None, description="The image URL if type is 'image'"
    )


class ChatMessage(BaseModel):
    role: str = Field(
        ..., description="The role of the message sender (user or assistant)"
    )
    content: Union[str, List[ContentItem]] = Field(
        ...,
        description="The content of the message, either a string or a list of content items",
    )


class ChatRequest(BaseModel):
    messages: Optional[List[ChatMessage]] = Field(
        [], description="History of messages between the user and the assistant"
    )
    debug: Optional[bool] = Field(False, description="Whether to enable debug logging")
    thread_id: Optional[str] = Field(
        "__default__", description="A specific conversation identifier"
    )
    max_plan_iterations: Optional[int] = Field(
        1, description="The maximum number of plan iterations"
    )
    max_step_num: Optional[int] = Field(
        3, description="The maximum number of steps in a plan"
    )
    auto_accepted_plan: Optional[bool] = Field(
        False, description="Whether to automatically accept the plan"
    )
    interrupt_feedback: Optional[str] = Field(
        None, description="Interrupt feedback from the user on the plan"
    )
