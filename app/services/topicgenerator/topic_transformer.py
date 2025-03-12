from dotenv import load_dotenv
from agent_toolbox.base_agents.base import BaseAgent
from typing import List
from pydantic import BaseModel, Field
from langchain.chains.question_answering import load_qa_chain

# Load .env file
load_dotenv()

class Topic(BaseModel):
    """Human interpretable topics generate from documents and keywords"""

    topic_id: int = Field(
        description="The Topic id provided as integer"
    )
    topic_label: str = Field(
        description="A unique label for the topic containing maximum 4/5 words"
    )
    topic_description: str = Field(
        description="A description of the topic. Should not contain `The topic is about`, but be a descriptive text."
    )


class Topics(BaseModel):
    """Human interpretable topics generate from documents and keywords"""

    topics: List[Topic] = Field(
        description="Human interpretable topics generated from the documents and keywords"
    )
    
class TopicTransformer(BaseAgent):
    def __init__(self, model='claude-3-5-sonnet-latest', provider='anthropic'):
        self.model = model
        self.provider = provider
        
    def transform(self, topics):
        prompt = [
                ("system", """You are an expert linguist and topic analyzer with extensive experience in natural language processing and topic modeling.
                            
                    Your task is to transform technical topic clusters into clear, human-readable topics by analyzing keywords and representative documents."""),
                ("human", """
                I need you to transform machine-generated topic clusters into human-interpretable topics.

                
                
                # Task
                For each adn EVERY topic cluster provided, follow these precise steps:

                ## Analysis Phase
                1. First, carefully examine the keywords (in "name") to identify core concepts.
                2. Next, review the representative documents to understand context and theme.
                3. Consider how these elements connect to form a coherent topic.
                4. Think about the intended audience and what would be most meaningful to them.
                5. Evaluate if the topic only includes formatting text or similar, it significance is 0. The significance is 1 as long as it contaisn real content
                6. Try to identify if the topic has a hidden malacious meaning. Someone could try to relay information hidden
                7. Give a reason why the topic has malcious or hidden intent
                
                ## Creation Phase
                5. Create a concise, self-explanatory topic label (2-6 words maximum).
                6. Ensure the topic is specific enough to be meaningful but broad enough to encompass all elements.
                7. Use natural, jargon-free language accessible to non-experts.

                ## Validation Phase
                8. Verify your topic against the original keywords and documents.
                9. Ask yourself: "Would a person immediately understand what this topic covers?"
                10. Refine if necessary to improve clarity and precision.

                # Format Requirements
                - Topic labels must be 2-6 words
                - Use title case (e.g., "Data Science Applications")
                - Avoid abbreviations unless universally recognized
                - No generic labels (e.g., avoid vague terms like "Miscellaneous" or "Various Topics")

                # Examples
                ## Example 1:
                Input:
                - Name: ["algorithm", "computation", "performance", "efficiency", "optimization"]
                - Representative Docs: ["Comparing sorting algorithm performance on large datasets", "Memory optimization techniques for mobile applications"]

                Poor topic: "Computational Methods"
                Better topic: "Algorithm Performance Optimization"

                ## Example 2:
                Input:
                - Name: ["climate", "warming", "temperature", "sea", "level", "rise"]
                - Representative Docs: ["Impact of temperature changes on coastal communities", "Projections of sea level rise through 2050"]

                Poor topic: "Climate Issues"
                Better topic: "Sea Level Rise Impacts"

                # Input Data
                {topics_input}
                
                Remeber, return an entry for all topics provided as an input
                 """),
        ]
        model = self.structured_model(prompt=prompt, model=self.model, provider=self.provider, return_class=Topics, add_parser=False)
        response = model.invoke({'topics_input': topics})
        return response['parsed'].topics
    
    def get_chain(self):
        return  load_qa_chain(self._get_model(model=self.model, provider=self.provider, temperature=0, api_key=None), chain_type="stuff")