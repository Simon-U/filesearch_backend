from dotenv import load_dotenv
from agent_toolbox.base_agents.base import BaseAgent
from typing import List
import os
from pydantic import BaseModel, Field
from langchain.chains.question_answering import load_qa_chain

# Load .env file
load_dotenv()

class Topic(BaseModel):
    """Representation of a  topic"""
    topic_id: int = Field(description='The given opic id')
    topic_label: str = Field(description="A label from maximum 4/5 words for each topic")
    topic_description: str = Field(description="A description of the topic based on the provided keywords and documents")


class Topics(BaseModel):
    """Human interpretable topics generate from documents and keywords"""

    topics: List[Topic] = Field(
        description="List of topics"
    )
    
class TopicTransformer(BaseAgent):
    def __init__(self, model='claude-3-5-sonnet-latest', provider='anthropic'):
        self.model = model
        self.provider = provider
        
    def transform(self, topics):
        prompt = [
            ("system", """You are an expert linguist and topic analyzer, specialized in creating concise topic labels and detailed descriptions.
            Note: When 'img_description' appears in the input, it simply indicates text extracted from an image and should not affect the analysis."""),
            
            ("human", """
            I will provide you with topics and their details. For each topic, follow these steps:

            1. Analysis each topic:
            - Analyze the provided keywords and documents. What content do they have and what meaning? What whanted the author to convey?
            - Identify the central theme and relationships between elements
            
            2. Output Format:
            Label: Create a concise topic label (maximum 4-5 words)
            Description: Write a description (2-10 sentences) that:
            - Captures the essence of the topic
            - Provides context beyond just listing keywords
            
            3. Requirements:
            - Labels must be human-readable and interpretable
            - Descriptions should be detailed but focused
            - Avoid mentioning technical aspects like 'keywords' or 'documents' in the output
            
            The provided data:
            {topics_input}
            """)
        ]
        model = self.structured_model(prompt=prompt, model=self.model, provider=self.provider, return_class=Topics, add_parser=False)
        response = model.invoke({'topics_input': topics})
        print(response)
        return response['parsed'].topics
    
    def get_chain(self):
        return  load_qa_chain(self._get_model(model=self.model, provider=self.provider, temperature=0, api_key=None), chain_type="stuff")