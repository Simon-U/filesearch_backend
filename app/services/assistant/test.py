from app.services.assistant.agent import PersonalAssistant
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
import asyncio

load_dotenv(override=True)

agent = PersonalAssistant()

async def query():
    response = await agent.ainvoke({'messages': [HumanMessage("I am looking for a file which is about Mineral mining and exploration that was most recent modified.")],
                            'user': {'tenant_id': 'dhi'}})

    print(response['messages'][-1])
    
asyncio.run(query())