from abc import ABC, abstractmethod
from typing import List, Dict

class BaseTopicGenerator(ABC):
    """
    Abstract base class for topic generators.
    """

    @abstractmethod
    def generate_topics(self, chunk_list: List[str], **kwargs) -> Dict:
        """
        Abstract method to generate topics from a list of text chunks.

        Args:
            chunk_list (List[str]): A list of text chunks.
            nr_topics (int): Number of topics to extract.
            **kwargs: Additional parameters for specific topic models.

        Returns:
            Dict: A dictionary containing topics and related information.
        """
        pass