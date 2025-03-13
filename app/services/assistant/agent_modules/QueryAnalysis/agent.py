from langchain_core.prompts import ChatPromptTemplate

from agent_toolbox.base_agents.base import BaseAgent
from .utils import AssistantTools

__all__ = ["QueryAnalysis"]


class QueryAnalysis(BaseAgent, AssistantTools):
    """
    Analyse the user request and the previous conversation to make a complete request
    
    In the future he will also extract the meta data.
    """

    @staticmethod
    def _get_prompt():
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an AI assistant specialized in providing valuable insights into file search.
                    Your role is to answer the queries of the user strictly related to file search. 
                    If its a greeting answer respectfully to the User. You focus on finding files only.
                    
                    Guidelines for Responses:
                    1. If its a user request analyse the request in context of the previous conversation:
                    2. Reject unrelated queries by responding:
                        - "I am sorry, I can only provide information on files"

                    """,
                ),
                (
                    "human",
                    """
                    Think step by step:
                    1. Analyse the user request. What information does he want? Is the request of the User with respect to a filse search? 
                    2. Analyse the previous chat history and make a complete user request with that context
                    3. Analyse use `UserRequest` tool to generate a complete request which looks upp files and retrieves additional context.
                    4. If you can answer use the `AnswerRequest` tool
                    
                    The conversation history between you and the user:
                    - {messages}
                    """
                ),
            ]
        ).partial()
        return prompt

    def __new__(
        cls,
        model="claude-3-7-sonnet-latest",
        additional_tools=[],
        streaming=False,
    ):
        instance = super(QueryAnalysis, cls).__new__(cls)
        prompt = cls._get_prompt()
        llm = instance.tool_model(
            prompt=prompt,
            model=model,
            tools=cls._get_tools(additional_tools),
            streaming=streaming,
            add_parser=False,
        )
        return llm
