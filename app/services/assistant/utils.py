import json
import operator
from datetime import datetime
from enum import Enum
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing import List, Annotated, Any, Tuple, Literal, Dict
from langchain_core.messages import BaseMessage
from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages


__all__ = ["ProcessStatus", "StreamUpdate"]


class GraphState(TypedDict):
    user_request: str
    messages: Annotated[list[AnyMessage], add_messages]
    user: dict
    context: dict
    