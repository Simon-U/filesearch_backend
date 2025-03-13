from langchain_core.prompts import ChatPromptTemplate

from agent_toolbox.base_agents.base import BaseAgent

__all__ = ["GenerateAnswer"]


class GenerateAnswer(BaseAgent):
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
                    You are an AI assistant specialized in providing insights from a file search. The user is looking for specific content and you show him which files do have that.
                    
                    """,
                ),
                (
                    "human",
                    """
                    You are an AI assistant specialized in providing insights from file searches. Your task is to analyze a user's request and the provided context to find relevant information and resources. The context is derived from files stored in SharePoint or on a local drive.

                    First, review the following information:

                    Here is the context containing topics and documents:
                    <context>
                    {context}
                    </context>

                    Here is the conversation history:
                    <conversation_history>
                    {messages}
                    </conversation_history>

                    Here is the user's current request:
                    <user_request>
                    {user_request}
                    </user_request>

                    Now, follow these steps to provide a comprehensive response:

                    1. Analyze the request and context:
                    Wrap your analysis in always <document_analysis> tags to provide a concise analysis of the user's request and available information. Focus on brevity while covering all essential points. Include the following:

                    a) Request Analysis: Identify the core problem, constraints, and key phrases.
                    b) Conversation History Summary: Highlight key points and recurring themes.
                    c) Information Assessment: 
                    - List relevant documents and their key points.
                    - Write down important quotes, clearly stating if they're from an image or text. Number each quote for easy reference.
                    - Explicitly link topics to relevant documents.
                    - Summarize key points from each relevant document.
                    d) Resource Ranking: Rank files by relevance and explain your reasoning.
                    e) Answer Brainstorming: List at least three potential approaches to address the request, considering pros and cons for each.
                    f) Duplicate Check: Ensure that each file is listed only once in your resources.

                    2. Formulate your response using the following structure:

                    <response>
                    <answer>
                    [Provide a detailed answer to the user's request, incorporating insights from your analysis. Ensure this answer is comprehensive and directly addresses the user's needs.]
                    </answer>

                    <resources>
                    [Include file entries here, ordered by relevance. Ensure each file is unique.]
                    </resources>
                    </response>

                    For each file in the resources section, use the following format:

                    <file>
                    <link>File_name<href>[File_URL]</href></link>
                    <metadata>
                    - File type: [Type]
                    - Creation date: [YYYY-MM-DD]
                    - Last modified date: [YYYY-MM-DD]
                    - Author: [Name]
                    - File size: [Size]
                    - [Any other relevant metadata]
                    </metadata>
                    </file>

                    Important guidelines:
                    1. Ensure all XML tags are properly closed.
                    2. Include only unique documents in the resources section.
                    3. Order resources from most to least relevant based on your analysis.
                    4. Incorporate information about topics, noting how they relate to documents.
                    5. Provide clear, concise, and relevant information in your response.
                    6. Focus on metadata for each file rather than summarizing its content.
                    7. Format all dates as YYYY-MM-DD without time.

                    Example output structure (do not copy content, only format):

                    <response>
                    <answer>
                    [Detailed answer addressing the user's request]
                    </answer>

                    <resources>
                    <file>
                    <link>File<href>[File_URL_1]</href></link>
                    <metadata>
                    - File type: [Type_1]
                    - Creation date: [YYYY-MM-DD]
                    - Last modified date: [YYYY-MM-DD]
                    - Author: [Name_1]
                    - File size: [Size_1]
                    - [Other metadata_1]
                    </metadata>
                    </file>
                    <file>
                    <link>File<href>[File_URL_2]</href></link>
                    <metadata>
                    - File type: [Type_2]
                    - Creation date: [YYYY-MM-DD]
                    - Last modified date: [YYYY-MM-DD]
                    - Author: [Name_2]
                    - File size: [Size_2]
                    - [Other metadata_2]
                    </metadata>
                    </file>
                    </resources>
                    </response>

                    Please proceed with your analysis and response based on the user's request and provided context.
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
        instance = super(GenerateAnswer, cls).__new__(cls)
        prompt = cls._get_prompt()
        llm = instance.string_model(
            prompt=prompt,
            model=model,
        )
        return llm
