from app.services.assistant.agent import PersonalAssistant
from typing import Dict, Any
import time
import uuid
from datetime import datetime
from langchain_core.messages import HumanMessage

def register_socket_events(sio):

    @sio.event
    async def user_request(sid, data):

        thread_id = data.get("thread_id", uuid.uuid4())
        user_request = data["user_request"]
        browser_socket = data["browser_socket"]
        user = data["user"]

        inputs = {"messages": HumanMessage(user_request), "user": user}
        agent = PersonalAssistant()
        response = await agent.ainvoke(inputs)
        now = datetime.now()
        answer = {
            "content": response['messages'][-1].content,
            "role": "Assistant",
            "date": now.strftime("%I:%M%p %d %b %Y"),
            "message_id": str(uuid.uuid4()),
            "type": "full",
            }

        await sio.emit("agent_response",answer , room=browser_socket)