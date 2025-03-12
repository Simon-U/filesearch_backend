import torch
import tempfile
import os
from typing import Dict, Any, Optional, List

from umap import UMAP
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer, OnlineCountVectorizer
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN

from .bert_s3_loader import BertTopicS3Loader

from sklearn.decomposition import IncrementalPCA

class BertTopicGenerator(BertTopicS3Loader):
    def __init__(self, sentence_transformer="paraphrase-multilingual-MiniLM-L12-v2", language='english', 
                 vector_count_min_df=0.01, vector_count_max_df=0.95, ngram_range=(1, 3), 
                 tokenizer=None, enable_bm25=False, reduce_frequent_words=False,
                 diversity=0.2, top_n_words_per_topic=10, min_cluster_size=10, 
                 min_topic_size=15, verbose=False, calculate_probabilities=False,
                 online_topic=False, river_algorithm="DBSTREAM", decay=0.01,
                 n_components=5):

        """
        Initialize the BertTopicGenerator with options for standard, incremental, or River-based topic modeling.
        
        Args:
            sentence_transformer (str): Name of the sentence transformer model to use
            language (str): Language for stopwords
            vector_count_min_df (float): Minimum document frequency for CountVectorizer
            vector_count_max_df (float): Maximum document frequency for CountVectorizer
            ngram_range (tuple): N-gram range for CountVectorizer
            tokenizer (callable, optional): Custom tokenizer for CountVectorizer
            enable_bm25 (bool): Whether to use BM25 weighting
            reduce_frequent_words (bool): Whether to reduce the importance of frequent words
            diversity (float): Diversity parameter for MaximalMarginalRelevance
            top_n_words_per_topic (int): Number of words per topic
            min_cluster_size (int): Minimum cluster size for HDBSCAN
            min_topic_size (int): Minimum topic size for BERTopic
            verbose (bool): Whether to print verbose output
            calculate_probabilities (bool): Whether to calculate probabilities
            online_topic (bool): Whether to use online topic modeling
            river_algorithm (str): River algorithm to use ('DBSTREAM', 'STREAMKMeans', 'DenStream')
            decay (float): Decay parameter for OnlineCountVectorizer
            n_components (int): Number of components for dimensionality reduction
        """
        # We choose the transformer model to make sentences
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # 1. Initialize a Sentence Transformer model with GPU support
        self.sentence_model = SentenceTransformer(sentence_transformer)
        if device == 'cuda':
            self.sentence_model = self.sentence_model.to(device)
        
        # Track if the model is trained
        self.is_trained = False
        
        # Set up models based on online_topic parameter
        if online_topic:
            # Set up models for online topic modeling with River
            from river import cluster, stream
            
            # River wrapper class for BERTopic
            class River:
                def __init__(self, model):
                    self.model = model

                def partial_fit(self, umap_embeddings):
                    for umap_embedding, _ in stream.iter_array(umap_embeddings):
                        self.model.learn_one(umap_embedding)

                    labels = []
                    for umap_embedding, _ in stream.iter_array(umap_embeddings):
                        label = self.model.predict_one(umap_embedding)
                        labels.append(label)

                    self.labels_ = labels
                    return self
            
            # Select the River algorithm
            if river_algorithm == "DBSTREAM":
                river_model = cluster.DBSTREAM()
            elif river_algorithm == "STREAMKMeans":
                river_model = cluster.STREAMKMeans(n_clusters=min(100, min_cluster_size*2))
            elif river_algorithm == "DenStream":
                river_model = cluster.DenStream()
            else:
                raise ValueError(f"Unsupported River algorithm: {river_algorithm}")
            
            # Set up models for River-based online learning
            self.vectorizer_model = OnlineCountVectorizer(
                stop_words=language, 
                decay=decay, 
                min_df=vector_count_min_df, 
                max_df=vector_count_max_df,
                ngram_range=ngram_range,
                tokenizer=tokenizer
            )
            self.ctfidf_model = ClassTfidfTransformer(
                bm25_weighting=enable_bm25,
                reduce_frequent_words=reduce_frequent_words
            )
            umap_model = IncrementalPCA(n_components=n_components)
            hdbscan_model = River(river_model)
            
        else:
            # Standard topic modeling setup
            self.vectorizer_model = CountVectorizer(
                ngram_range=ngram_range, 
                stop_words=language, 
                min_df=vector_count_min_df,
                max_df=vector_count_max_df,
                tokenizer=tokenizer
            )
            self.ctfidf_model = ClassTfidfTransformer(
                bm25_weighting=enable_bm25,
                reduce_frequent_words=reduce_frequent_words
            )
            umap_model = UMAP(random_state=42)
            hdbscan_model = HDBSCAN(
                min_cluster_size=min_cluster_size, 
                metric='euclidean', 
                cluster_selection_method='eom', 
                prediction_data=True
            )

        self.representation_model = MaximalMarginalRelevance(diversity=diversity)
        self.online_topic = online_topic
        
        self.model = BERTopic(
            embedding_model=self.sentence_model,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.representation_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics="auto",
            min_topic_size=min_topic_size,
            verbose=verbose,
            calculate_probabilities=calculate_probabilities
        )

    def train_model(self, chunk_list: List[str], **kwargs):
        """Initial training of the model"""
        if self.online_topic:
            # For online topic modeling, use partial_fit from the beginning
            self.model.partial_fit(chunk_list)
        else:
            # For standard topic modeling, use fit
            self.model = self.model.fit(chunk_list)
        self.is_trained = True
    
    def update_model(self, new_documents: List[str], **kwargs):
        """Incrementally update the model with new documents"""
        if not self.is_trained:
            self.train_model(new_documents)
        else:
            # Use partial_fit for incremental learning
            self.model.partial_fit(new_documents)
    
    def save_model(self, path: str, **kwargs):
        self.model.save(path, **kwargs)
        
    def load_model(self, path: str, **kwargs):
        self.model = self.model.load(path, **kwargs)
        
    def load_model_from_s3(self, s3_path: str, bucket_name: str, 
                          region_name: str = "us-west-2",
                          aws_access_key_id: Optional[str] = None, 
                          aws_secret_access_key: Optional[str] = None,
                          endpoint_url: Optional[str] = None):
        """
        Load a BERTopic model from S3.
        
        :param s3_path: Path to the model in S3
        :param bucket_name: S3 bucket name
        :param region_name: AWS region
        :param aws_access_key_id: AWS access key ID
        :param aws_secret_access_key: AWS secret access key
        :param endpoint_url: Custom S3 endpoint URL
        """
        local_path, temp_dir = self.download_model_from_s3(
            s3_path=s3_path,
            bucket_name=bucket_name,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=endpoint_url
        )
        self.load_model(path=str(local_path))
        self.cleanup_temp_dir(temp_dir)
        
    def save_model_to_s3(self, s3_path: str, bucket_name: str,
                        region_name: str = "us-west-2",
                        aws_access_key_id: Optional[str] = None,
                        aws_secret_access_key: Optional[str] = None,
                        endpoint_url: Optional[str] = None,
                        **model_save_kwargs):
        """
        Save the current BERTopic model to S3.
        
        :param s3_path: Path in S3 where to save the model
        :param bucket_name: S3 bucket name
        :param region_name: AWS region
        :param aws_access_key_id: AWS access key ID
        :param aws_secret_access_key: AWS secret access key
        :param endpoint_url: Custom S3 endpoint URL
        :param model_save_kwargs: Additional keyword arguments to pass to model.save()
        :return: True if successful
        """
        # Create a temporary directory for saving the model
        temp_dir = tempfile.mkdtemp()
        try:
            # Determine local path - use the name from s3_path if it's a .pickle file

            local_filename = "model"
            
            local_path = os.path.join(temp_dir, local_filename)
            
            # Save the model to the temporary location
            self.model.save(local_path, **model_save_kwargs)
            
            # Upload the model to S3
   
            success = self.save_model_to_s3_(
                local_model_path=local_path,
                s3_path=s3_path,
                bucket_name=bucket_name,
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                endpoint_url=endpoint_url
            )
            
            return success
        finally:
            # Clean up the temporary directory
            self.cleanup_temp_dir(temp_dir)

    def merge_models(self, models: List, min_similarity=0.7):
        models = [self.model] + models
        self.model = BERTopic.merge_models(models, min_similarity=min_similarity)
    
    def fit_transform(self, docs:list):
        return self.model.fit_transform(docs)

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

    def get_document_topic_distribution(self, document: str, threshold: float = 0.05, min_topics: int = 1):
            """
            Calculate the topic distribution for a single document with a probability threshold.
            
            Args:
                document (str): The document text to analyze
                threshold (float, optional): Minimum probability threshold to include a topic. Defaults to 0.05.
                min_topics (int, optional): Minimum number of topics to return, even if below threshold. Defaults to 1.
                
            Returns:
                List[tuple]: A list of tuples containing (topic_id, probability) sorted by probability in descending order.
                            Will return at least min_topics (if available), regardless of threshold.
            """
            # Ensure document is not empty
            if not document or not document.strip():
                return []
            
            topic_probs = []
            
            try:
                # If we initialized BERTopic with calculate_probabilities=True (works only with HDBSCAN)
                if hasattr(self.model, 'calculate_probabilities') and self.model.calculate_probabilities:
                    # Transform the document to get topic distribution
                    topics, probabilities = self.model.transform([document])
                    
                    # Get the topic probabilities for the document
                    for topic_id, prob in enumerate(probabilities[0]):
                        if topic_id != -1:  # Skip outlier topic -1
                            topic_probs.append((topic_id, prob))
                else:
                    # Use approximate_distribution for a more reliable distribution (works with any clustering algorithm)
                    try:
                        topic_distr, _ = self.model.approximate_distribution([document], window=4, stride=1)
                        
                        # Extract all topics with their probabilities
                        for topic_id, prob in enumerate(topic_distr[0]):
                            if topic_id != -1:  # Skip outlier topic -1
                                topic_probs.append((topic_id, prob))
                    except Exception as e:
                        # If approximate_distribution fails, fall back to simple transform
                        topics = self.model.transform([document])
                        if isinstance(topics, tuple):
                            topics = topics[0]  # Extract topics if transform returns (topics, probs)
                        
                        # Create a simple distribution - assign 1.0 to the predicted topic
                        if len(topics) > 0 and topics[0] != -1:
                            topic_probs = [(topics[0], 1.0)]
            except Exception as e:
                print(f"Error in topic distribution calculation: {str(e)}")
                return []
            
            # Sort by probability in descending order
            topic_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Apply threshold, but ensure we return at least min_topics if available
            if topic_probs:
                # Filter by threshold
                filtered_probs = [tp for tp in topic_probs if tp[1] >= threshold]
                
                # If we have fewer than min_topics after filtering, take the top min_topics
                if len(filtered_probs) < min_topics and len(topic_probs) > 0:
                    return topic_probs[:min(min_topics, len(topic_probs))]
                
                return filtered_probs
            
            return []