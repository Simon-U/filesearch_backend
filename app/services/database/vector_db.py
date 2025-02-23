import uuid
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, MatchExcept, MatchText, Range, DatetimeRange, NestedCondition, Nested
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

class QuantVectorDB:
    def __init__(self, tenant_id, mode="remote", host="localhost", port=6333, path=None, use_gpu=True):
        """
        Initializes connection to Qdrant Vector Database, ensures a collection exists for the given tenant,
        and configures GPU acceleration if available.

        :param tenant_id: Unique identifier for the tenant (company).
        :param mode: Connection mode - "memory", "local", or "remote".
        :param host: Hostname for Qdrant server (used in remote mode).
        :param port: Port for Qdrant server (used in remote mode).
        :param path: Path for local persistent storage (used in local mode).
        :param use_gpu: Whether to enable GPU acceleration if available (default: True).
        """
        self.tenant_id = tenant_id
        self.collection_name = f"tenant_{tenant_id}_collection"
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.valid_conditions = {
            "match", "match_any", "match_except", "full_text_match", 
            "range", "datetime_range", "nested"
        }

        # Initialize Qdrant client
        if mode == "memory":
            self.client = QdrantClient(":memory:")
        elif mode == "local":
            if not path:
                raise ValueError("Path must be provided for local mode.")
            self.client = QdrantClient(path=path)
        elif mode == "remote":
            self.client = QdrantClient(host=host, port=port)
        else:
            raise ValueError("Invalid mode. Choose from 'memory', 'local', or 'remote'.")

        # Set up GPU acceleration if enabled
        if use_gpu:
            try:
                self.client.set_model(
                    self.client.DEFAULT_EMBEDDING_MODEL,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                print("GPU acceleration enabled!")
            except Exception as e:
                print(f"Warning: GPU acceleration could not be enabled. Falling back to CPU. Error: {e}")

        # Ensure tenant collection exists
        self.ensure_collection_exists()

    def embed_topics_concat(self, topics, max_topics=5):
        """
        Converts a list of topics into a single concatenated vector.
        - Embeds each topic separately.
        - Limits to `max_topics` (default 5).
        - If fewer topics, pads with zeros.

        :param topics: List of topic strings (1 to `max_topics` topics).
        :param model: Pretrained embedding model.
        :param max_topics: Maximum number of topics to consider.
        :return: Concatenated topic vector.
        """
        if not topics:
            raise ValueError("At least one topic is required.")

        # Ensure the topic list is at most `max_topics`
        topics = topics[:max_topics]

        # Generate embeddings for each topic
        topic_vectors = [self.embedding_model.encode(topic) for topic in topics]

        # If fewer than `max_topics`, pad with zero vectors of same size
        vector_dim = len(topic_vectors[0])  # Get vector size from the model
        while len(topic_vectors) < max_topics:
            topic_vectors.append(np.zeros(vector_dim))  # Zero-padding

        # Concatenate all topic vectors into one long vector
        concat_vector = np.concatenate(topic_vectors, axis=0).tolist()

        return concat_vector

    def test_connection(self):
        """
        Tests the connection by getting Qdrant's server status.
        """
        try:
            return self.client.get_collections()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def ensure_collection_exists(self):
        """
        Checks if the tenant's collection exists, and if not, creates it.
        """

        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "topics": VectorParams(size=1920, distance=Distance.COSINE),
                    "summary": VectorParams(size=384, distance=Distance.COSINE)
                }
            )
            
    def insert_document(self, topic, summary, metadata=None, id=None):
        """
        Inserts a document into Qdrant with separate topic and summary embeddings and metadata.
        Uses `upsert()` and allows Qdrant to auto-generate the document ID.

        :param topic: The topic of the document.
        :param summary: The summary of the document.
        :param metadata: Optional dictionary containing additional metadata.
        """
        # Base metadata payload
        payload = {
            "tenant_id": self.tenant_id,
            "topics": topic,
            "summary": summary
        }

        # Merge user-provided metadata (if any)
        if metadata:
            payload.update(metadata)
        
        topic_vector = self.embed_topics_concat(topic)
        summary_vector = self.embedding_model.encode(summary).tolist()
    
        # Upsert the document with auto-generated ID
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()) if not id else id,  # Let Qdrant generate the ID automatically
                    vector={
                        "topics": topic_vector,  # Store topic embedding
                        "summary": summary_vector  # Store summary embedding
                    },
                    payload=payload  # Store metadata
                )
            ]
        )
    
     
    def build_filter(self, metadata):
            """
            Build a Qdrant filter based on the metadata dictionary.
            
            Args:
                metadata (dict): Dictionary containing filter clauses and conditions.
            
            Returns:
                Filter: A Qdrant Filter object.
            """
            filter_conditions = {"must": [], "should": [], "must_not": []}

            for clause, conditions in metadata.items():
                for key, condition in conditions.items():
                    # Validate condition type
                    condition_type = next(iter(condition))
                    if condition_type not in self.valid_conditions:
                        raise ValueError(f"Invalid condition type: {condition_type}")

                    # Create appropriate FieldCondition or NestedCondition
                    if condition_type == "match":
                        field_condition = FieldCondition(key=key, match=MatchValue(value=condition["match"]))
                    elif condition_type == "match_any":
                        field_condition = FieldCondition(key=key, match=MatchAny(any=condition["match_any"]))
                    elif condition_type == "match_except":
                        field_condition = FieldCondition(key=key, match=MatchExcept(except_=condition["match_except"]))
                    elif condition_type == "full_text_match":
                        field_condition = FieldCondition(key=key, match=MatchText(text=condition["full_text_match"]))
                    elif condition_type == "range":
                        field_condition = FieldCondition(key=key, range=Range(**condition["range"]))
                    elif condition_type == "datetime_range":
                        field_condition = FieldCondition(key=key, range=DatetimeRange(**condition["datetime_range"]))
                    elif condition_type == "nested":
                        nested_filter = self.build_filter({"must": condition["nested"]})
                        field_condition = NestedCondition(nested=Nested(key=key, filter=nested_filter))
                    
                    # Append the created condition to the appropriate clause
                    filter_conditions[clause].append(field_condition)

            return Filter(
                must=filter_conditions["must"] if filter_conditions["must"] else None,
                should=filter_conditions["should"] if filter_conditions["should"] else None,
                must_not=filter_conditions["must_not"] if filter_conditions["must_not"] else None
            )
        
    def search_document(self, topic=None, summary=None, metadata={}):
        """
        Searches documents in Qdrant using topic and summary vectors while applying metadata filters.

        Args:
            topic (str): The topic text used to generate the topic vector.
            summary (str): The summary text used to generate the summary vector.
            metadata (dict): A dictionary containing filter conditions for the search.
            
            Metadata Structure Example:
            {
                "must": {
                    "city": {"match": "London"},
                    "price": {"range": {"gte": 100, "lte": 450}},
                    "date": {"datetime_range": {"gte": "2023-02-08T10:49:00Z", "lte": "2024-01-31T10:14:31Z"}}
                },
                "should": {
                    "color": {"match_any": ["red", "green"]},
                    "category": {"full_text_match": "AI research"}
                },
                "must_not": {
                    "status": {"match_except": ["inactive", "banned"]}
                },
                "nested": {
                    "diet": {  
                        "must": {  
                            "food": {"match": "meat"},
                            "likes": {"match": True}
                        }
                    }
                }
            }
            
            Enabled conditions: match, match_any, match_except, full_text_match, range, datetime_range, nested.

        Returns:
            dict: A merged dictionary containing search results from both topic and summary queries.
            If an error occurs, returns {"error": <error_message>}.

        """

        try:
            # Generate vectors
            payload_filter = self.build_filter(metadata)
            response_topics = None
            response_summary = None
            if topic:
                topic_vector = self.embed_topics_concat(topic)
                response_topics = self.client.query_points(
                collection_name=self.collection_name,
                query=topic_vector,  
                using="topics",  # Search in the "topics" vector field
                limit=3,
                with_payload=True,
                query_filter=payload_filter
                )
            if summary:
                summary_vector = self.embedding_model.encode(summary).tolist()
                response_summary = self.client.query_points(
                collection_name=self.collection_name,
                query=summary_vector,  
                using="summary",  # Search in the "summary" vector field
                limit=3,
                with_payload=True,
                query_filter=payload_filter
                )
            
            # Build metadata filter
            
            
            # Query for topics
            

            # Query for summary
            

            # Merge the results from both queries
            merged_results = {}
            merged_results["topics_results"] = response_topics
            merged_results["summary_results"] = response_summary

            return merged_results

        except Exception as e:
            return {"error": str(e)}