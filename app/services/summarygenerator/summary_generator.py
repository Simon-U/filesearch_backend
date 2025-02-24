from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os
import random
from typing import List
from agent_toolbox.base_agents.base import BaseAgent
# Load .env file
load_dotenv()

__all__ = ['Summarizer']

class Summary(BaseModel):
    "Concicse summary of the content. Use a single concise text with the tool"
    summary: str = Field(
        description="A clear and cocise summary capturing the relevant content of the documents in a single text"
    )
        
class Summarizer(BaseAgent):
    def __init__(self, model='claude-3-5-sonnet-latest', provider='anthropic'):
        self.model = model
        self.provider = provider
        self.llm_call_count = 0  # Track the number of LLM calls

    def summarize_chunk(self, chunk):
        """
        Summarize a single chunk of text using the LLM.
        """

        prompt = [
                ("system", """
                    You are a professional summarization AI. Your task is to generate a **concise summary** of the provided text..
                    
                    """),
                ("human",  """
                    Please summarize the following text context from a document into a **single coherent summary** while following the required JSON format provided with the tool Summary.

                    #Provided text to summarise:
                    
                    {chunk_input}
                    
                    Remember:
                    - Output ONLY valid JSON with single key summary defined by the tool provided to you.
                    - Do NOT add any additional text or formatting.
                    - Do NOT wrap JSON in triple backticks.
                    
                    ## **Output Format (Strictly JSON)**
                    You must output ONLY valid JSON with the tool provided to you:

                    ```json
                    {{
                    "response": "Your detailed and concise summary in a single text block"
                    }}
                    """),
        ]
    
        self.llm_call_count += 1  # Increment the LLM call count
        model = self.string_model(prompt=prompt, model=self.model, provider=self.provider)
        response = model.invoke({"chunk_input": chunk})
        return response

    def batch_and_summarize(self, summaries, tokenthreshold):
        # If the combined length exceeds the token threshold, split into batches
        combined_summary = "\n".join(summaries)
        if len(combined_summary.split()) <= tokenthreshold:
            return combined_summary

        # Break into smaller batches and summarize each batch
        batch_size = 5  # Start with 5 summaries per batch (adjustable)
        batched_summaries = []
        for i in range(0, len(summaries), batch_size):
            batch = summaries[i:i + batch_size]
            batch_summary = self.summarize_chunk("\n".join(batch))
            batched_summaries.append(batch_summary)

        # Recursively summarize the batches
        return self.batch_and_summarize(batched_summaries, tokenthreshold)

    def summarize(self, chunks, skip=1, tokenthreshold=25000):
        """
        Summarize a list of text chunks while ensuring we stay within the token threshold.

        :param chunks: List of text chunks to summarize.
        :param skip: Skip strategy: 1 for no skip, 2 for every second chunk, etc.
        :param tokenthreshold: Maximum token limit for the summaries.
        :return: A single summary string and the LLM call count.
        """
        self.llm_call_count = 0  # Reset call count
        accumulated_chunks = []  # Store collected chunks
        batch_summaries = []  # Store summaries for each batch
        current_token_count = 0  # Track token usage in current batch

        i = 0
        while i < len(chunks):
            # Select a random chunk from the next 'skip' chunks
            group = chunks[i:i+skip]  # Get the current skip-sized window
            if group:  
                selected_chunk = random.choice(group)  # Pick one randomly

                chunk_tokens = len(selected_chunk.split())  # Estimate token count
                if current_token_count + chunk_tokens > tokenthreshold:
                    # If adding this chunk exceeds the limit, summarize the accumulated batch first
                    batch_summary = self.summarize_chunk("\n".join(accumulated_chunks))
                    batch_summaries.append(batch_summary)
                    
                    # Reset batch collection
                    accumulated_chunks = []
                    current_token_count = 0

                # Add selected chunk to the batch
                accumulated_chunks.append(selected_chunk)
                current_token_count += chunk_tokens
            
            i += skip  # Move to the next skip group

        # Summarize the final batch if there are remaining chunks
        if accumulated_chunks:
            batch_summary = self.summarize_chunk("\n".join(accumulated_chunks))
            batch_summaries.append(batch_summary)

        # Recursively summarize the batch summaries to ensure it stays within limits
        final_summary = self.batch_and_summarize(batch_summaries, tokenthreshold)

        return final_summary, self.llm_call_count
