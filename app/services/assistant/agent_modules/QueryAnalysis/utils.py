from typing import List
from pydantic import BaseModel, Field

__all__ = ["AssistantTools"]


class UserRequest(BaseModel):
    "The complete user request"
    user_request: str = Field(
        description="The complete user request based on the previous messages"
    )

class AnswerRequest(BaseModel):
    "The answer to the user"
    answer: str = Field(
        description="A complete and polite answerr to the request of the user"
    )

class AssistantTools:
    @staticmethod
    def _get_sensitive_tools():
        return []

    @staticmethod
    def _get_safe_tools():
        return [UserRequest, AnswerRequest]

    @staticmethod
    def _get_tools(additional_tools: List = []):
        sensitive_tools = AssistantTools._get_sensitive_tools()
        save_tools = AssistantTools._get_safe_tools()
        tools = sensitive_tools + save_tools + additional_tools
        return tools