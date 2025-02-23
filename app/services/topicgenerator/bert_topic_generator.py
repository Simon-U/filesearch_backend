from typing import List, Dict
from umap import UMAP
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN


class BertTopicGenerator():
    def __init__(self, sentence_transformer="paraphrase-multilingual-MiniLM-L12-v2", language='english', vector_count_min_df=0.01, vector_count_max_df=0.95, ngram_range=(1, 2), tokenizer=None, enable_bm25=False, reduce_frequent_words=False,
                 diversity=0.1, top_n_words_per_topic=10, min_cluster_size=10, min_topic_size=10):
        # We choose the transformer model to make sentences
        self.sentence_model = SentenceTransformer(sentence_transformer)
        
        # Creating the topic representation
        self.vectorizer_model = CountVectorizer(ngram_range=ngram_range, stop_words=language, tokenizer=tokenizer)
        self.ctfidf_model = ClassTfidfTransformer(bm25_weighting=enable_bm25 ,reduce_frequent_words=reduce_frequent_words)
        umap_model = UMAP(random_state=42)
        representation_model = MaximalMarginalRelevance(diversity=diversity)
        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        self.model = BERTopic(embedding_model=self.sentence_model, 
                              vectorizer_model=self.vectorizer_model,
                              ctfidf_model=self.ctfidf_model,
                              representation_model=representation_model,
                              umap_model=umap_model,
                              hdbscan_model=hdbscan_model,
                              nr_topics="auto",
                              min_topic_size=min_topic_size
                              )
    
    def train_model(self, chunk_list: List[str], **kwargs):
        self.model = self.model.fit(chunk_list)
        
    def merge_models(self, models: List):
        models = [self.model] + models
        self.model = BERTopic.merge_models(models, min_similarity=0.99)
        
    def get_topic_info(self):
        return self.model.get_topic_info()
        
    def get_topic(self, chunk) -> Dict:
        """
        Retrieve the topic for a single chunk of text.
        
        Args:
            chunk (str): The text chunk to analyze.
        
        Returns:
            Dict: A dictionary containing the topic number, probability, and detailed topic keywords.
        """


        # Preprocess the chunk
        preprocessed_chunk = chunk

        # Get topic and probability
        topic, prob = self.model.transform(preprocessed_chunk)

        # Get detailed topic information
        topic_info = self.model.get_topic(topic[0]) if topic[0] != -1 else []

        return {
            "topic": topic[0],  # Topic number
            "probability": prob[0],  # Probability score
            "keywords": topic_info  # Keywords defining the topic
        }


    def generate_topics(self, chunk_list: List[str], **kwargs) -> Dict:
        preprocessed_text = chunk_list #self.preprocess_text(chunk_list, **kwargs)

        if not preprocessed_text or all(len(chunk.strip()) == 0 for chunk in preprocessed_text):
            raise ValueError("Preprocessed text is empty. Ensure the input text is valid.")

        bert = self.model
        topics, probs = bert.fit_transform(preprocessed_text)

        topic_info = bert.get_topic_info()
        detailed_topics = {
            topic: bert.get_topic(topic) for topic in topic_info["Topic"].unique() if topic != -1
        }

        return {
            #"topics": topics,
            #"probabilities": probs,
            "topic_info": topic_info,
            "detailed_topics": detailed_topics,
        }
