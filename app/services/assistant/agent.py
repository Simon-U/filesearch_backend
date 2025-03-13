
import asyncio
import json
import httpx
from os import environ
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from typing import Optional, List, Dict, Any
from typing_extensions import TypedDict, Literal
from langchain_core.messages import AIMessage


from .utils import GraphState
from .agent_modules.QueryAnalysis.agent import QueryAnalysis
from .agent_modules.GenerateAnswer.agent import GenerateAnswer
class PersonalAssistant():
    def __new__(
        cls,
        checkpointer=None,
        streaming=False,
    ):
        instance = super(PersonalAssistant, cls).__new__(cls)
        return instance.create_agent(
            checkpointer=checkpointer,
            streaming=streaming,
        )
        

    async def hybrid_search(
        self,
        query: str,
        tenant_id: str,
        meta_data: Optional[Dict[str, Any]] = {},
        probability_limit: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Test the hybrid search endpoint which combines topic and chunk search.
        
        Args:
            query: Search query string
            tenant_id: Tenant ID for data isolation
            meta_data: Optional metadata filtering conditions
            probability_limit: Topic probability threshold
        
        Returns:
            Dict containing API response
        """
        # Construct the endpoint URL
        url = environ.get("SEARCH_ENDPOINT")
        
        # Build query parameters
        params = {
            "query": query,
            "tenant_id": tenant_id,
            "meta_data": json.dumps(meta_data)
        }
        
        # If metadata is provided, convert to JSON string

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"Error response: {response.text}")
                    return {"error": response.text}
        except Exception as e:
            print(f"Request failed: {str(e)}")
            return {"error": str(e)}
        
    @staticmethod
    def entry(state:GraphState) -> Command[Literal["QueryAnalysis"]]:
        if not state['user']['tenant_id']:
            raise ValueError("The user with tentendid needs to be provided in a dic 'user'")
        return Command(goto="QueryAnalysis")
    
    @staticmethod
    def query_analysis(state:GraphState) -> Command[Literal["DocumentRetrival", END]]:
        agent = QueryAnalysis()
        response = agent.invoke(state)
        tool = response.tool_calls[0]
        state_updates = {}
        if tool['name'] == 'AnswerRequest':
            next_node = END
            state_updates['messages'] = AIMessage(tool['args']['answer'])
        
        elif tool['name'] == 'UserRequest':
            next_node = 'DocumentRetrival'
            state_updates['user_request'] = tool['args']['user_request']
        else:
            raise ValueError("The QueryAnalysis did not generate the expected answer with the tools")
        return Command(goto=next_node, update=state_updates)

    async def ducument_retrival(self, state:GraphState) -> Command[Literal["GenerateAnswer"]]:

        response = await self.hybrid_search(
            query=state['user_request'], 
            tenant_id=state['user']['tenant_id'],
        )
        documents = response['data']
        topics = response['data']['topics']
        
        documents = response['data']['chunks']
        return Command(goto="GenerateAnswer", update={'context': {'topics': topics, "documents": documents}})    
    
    @staticmethod
    def generate_answer(state:GraphState) -> Command[Literal[END]]:
        agent = GenerateAnswer()
        response = agent.invoke(state)
        return Command(goto=END, update={'messages': [response]}) 
    
    def create_agent(self, checkpointer, streaming):
        
        workflow = StateGraph(GraphState)
        workflow.add_node("Entry", self.entry)
        workflow.set_entry_point("Entry")
        
        workflow.add_node("QueryAnalysis", self.query_analysis)
        workflow.add_node("DocumentRetrival", self.ducument_retrival)
        
        workflow.add_node("GenerateAnswer", self.generate_answer)
        
        agent = workflow.compile(checkpointer=checkpointer)
        return agent