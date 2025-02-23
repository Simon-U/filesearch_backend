from typing import List, Dict
from .base_topic_generator import BaseTopicGenerator
from .bert_topic_generator import BertTopicGenerator
from .topic_transformer import TopicTransformer

class TopicGenerator:
    """
    A flexible topic generator that dynamically selects a topic model based on user input.
    """

    _generators = {
        "bertopic": BertTopicGenerator,
        # Future models can be added here, e.g., "lda": LDATopicGenerator
    }

    def __init__(self, method: str, model:str = 'claude-3-5-sonnet-latest',provider:str ='anthropic', **kwargs):
        """
        Initializes the topic generator with the selected method.

        Args:
            method (str): The name of the topic generation method (e.g., "bertopic").
            **kwargs: Additional arguments to pass to the selected generator class.
        """
        if method not in self._generators:
            raise ValueError(f"Unsupported topic generator: {method}. Available options: {list(self._generators.keys())}")
        self.model = model
        self.provider = provider
        self.generator: BaseTopicGenerator = self._generators[method](**kwargs)

    def transform_topic(self, topic_info, **kwargs) -> Dict:
        """
        Generate topics using the selected topic generator.

        Args:
            chunk_list (List[str]): A list of text chunks.
            nr_topics (int): Number of topics to extract.
            **kwargs: Additional parameters for the specific topic model.

        Returns:
            Dict: A dictionary containing topics and related information.
        """
        topic_transformer = TopicTransformer(model=self.model, provider=self.provider)
        topics = topic_transformer.transform(topic_info)
        return topics
    
    def train_model(self, chunk_list: List[str], **kwargs) -> Dict:
        """
        Generate topics using the selected topic generator.

        Args:
            chunk_list (List[str]): A list of text chunks.
            nr_topics (int): Number of topics to extract.
            **kwargs: Additional parameters for the specific topic model.

        Returns:
            Dict: A dictionary containing topics and related information.
        """
        self.generator.train_model(chunk_list, **kwargs)
        
    def get_topic(self, chunk_list: List[str], **kwargs) -> Dict:
        """
        Generate topics using the selected topic generator.

        Args:
            chunk_list (List[str]): A list of text chunks.
            nr_topics (int): Number of topics to extract.
            **kwargs: Additional parameters for the specific topic model.

        Returns:
            Dict: A dictionary containing topics and related information.
        """
        return self.generator.get_topic(chunk_list, **kwargs)
    
    def get_topic_info(self):
        return self.generator.get_topic_info()
    
    def merge_models(self, models: list):
        self.generator.merge_models(models)
        