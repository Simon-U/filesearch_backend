import uuid
from datetime import datetime, timezone
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, MatchExcept, MatchText, Range, DatetimeRange, NestedCondition, Nested
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

class QuantVectorDB:
    def __init__(self, tenant_id, mode="remote", host="localhost", port=6333, path=None, use_gpu=False):
        """
        Initializes connection to Qdrant Vector Database, ensures collections exist for the given tenant,
        and configures GPU acceleration if available.

        :param tenant_id: Unique identifier for the tenant (company).
        :param mode: Connection mode - "memory", "local", or "remote".
        :param host: Hostname for Qdrant server (used in remote mode).
        :param port: Port for Qdrant server (used in remote mode).
        :param path: Path for local persistent storage (used in local mode).
        :param use_gpu: Whether to enable GPU acceleration if available (default: True).
        """
        self.tenant_id = tenant_id
        self.topics_collection = f"tenant_{tenant_id}_topics"
        self.chunks_collection = f"tenant_{tenant_id}_chunks"
        
        import os
        os.environ['NO_PROXY'] = 'localhost,127.0.0.1,172.17.0.0/16'
        os.environ['no_proxy'] = 'localhost,127.0.0.1,172.17.0.0/16'
        # Clear any potential invalid proxy settings
        for var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
            if os.environ.get(var) == 'True':
                os.environ[var] = ''

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
                self.client.set_model(
                    self.client.DEFAULT_EMBEDDING_MODEL,
                    providers=["CPUExecutionProvider"]
                )
                print(f"Warning: GPU acceleration could not be enabled. Falling back to CPU. Error: {e}")

        # Ensure tenant collections exist
        self.ensure_collections_exist()

    def ensure_collections_exist(self):
        """
        Checks if the tenant's collections exist, and if not, creates them.
        """
        # Create topics collection if it doesn't exist

        if not self.client.collection_exists(collection_name=self.topics_collection):
            self.client.create_collection(
                collection_name=self.topics_collection,
                vectors_config={
                    "description": VectorParams(size=384, distance=Distance.COSINE)
                }
            )
            
        # Create chunks collection if it doesn't exist
        if not self.client.collection_exists(collection_name=self.chunks_collection):
            self.client.create_collection(
                collection_name=self.chunks_collection,
                vectors_config={
                    "content": VectorParams(size=384, distance=Distance.COSINE)
                }
            )

    def test_connection(self):
        """
        Tests the connection by getting Qdrant's server status.
        """
        try:
            return self.client.get_collections()
        except Exception as e:
            return {"status": "error", "message": str(e)}

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

    def insert_topic(self, topic_id, topic_label, topic_description, metadata=None):
        """
        Inserts a topic into the topics collection with its vector embedding and metadata.
        
        :param topic_id: Unique identifier for the topic.
        :param topic_label: Label for the topic.
        :param topic_description: Description of the topic.
        :param metadata: Optional dictionary containing additional metadata.
        :return: True if successful, False otherwise.
        """
        try:
            # Ensure metadata exists
            if metadata is None:
                metadata = {}
            
            # Get current timestamp in ISO format
            current_time = datetime.now(timezone.utc).isoformat()
            
            # Create base payload
            payload = {
                "topic_id": topic_id,
                "topic_label": topic_label,
                "topic_description": topic_description,
                "topic_created_at": current_time,
                "topic_updated_at": current_time
            }
            
            # Merge with provided metadata
            payload.update(metadata)
            
            # Generate embedding for topic description
            description_vector = self.embedding_model.encode(topic_description).tolist()
            
            # Insert the topic - using topic_id as the point ID
            self.client.upsert(
                collection_name=self.topics_collection,
                points=[
                    PointStruct(
                        id= str(uuid.uuid4()),  # Use topic_id as the point ID
                        vector={
                            "description": description_vector
                        },
                        payload=payload
                    )
                ]
            )
            
            return True
        except Exception as e:
            print(f"Error inserting topic: {str(e)}")
            return e

    def insert_chunk(self, chunk_text, metadata=None, chunk_id=None):
        """
        Inserts a document chunk into Qdrant with its vector representation and metadata.
        
        This method stores a document chunk with its topic probability distribution and other metadata.
        If no chunk_id is provided, Qdrant automatically assigns an ID to the chunk.
        
        :param chunk_text: The text content of the chunk.
        :param metadata: Dict containing metadata including topic_probabilities, document_id, etc.
        :param chunk_id: Optional ID for the chunk. If not provided, Qdrant will generate one.
        :return: True if successful, False otherwise.
        """
        try:
            # Ensure metadata exists
            if metadata is None:
                metadata = {}
            
            # Add chunk text to metadata
            metadata["chunk_text"] = chunk_text
            
            # Generate embedding for chunk content
            chunk_vector = self.embedding_model.encode(chunk_text).tolist()
            
            # If this is an update and chunk_updated_at exists, update it
            if chunk_id and "chunk_created_at" in metadata:
                metadata["chunk_updated_at"] = datetime.now(timezone.utc).isoformat()
            
            # Create point structure with or without ID
            point = PointStruct(
                id= str(uuid.uuid4()),
                vector={"content": chunk_vector},
                payload=metadata
            )
            
            # Insert the chunk
            self.client.upsert(
                collection_name=self.chunks_collection,
                points=[point]
            )
            
            return True
        except Exception as e:
            print(f"Error inserting chunk: {str(e)}")
            return False

    def search_topics(self, topic_query=None, topic_ids=None, limit=3):
        """
        Searches for topics using both semantic search on descriptions and direct matching by IDs.
        Returns both sets of results separately in a structured dictionary.
        
        :param topic_query: Query text for semantic search on topic descriptions.
        :param topic_ids: Optional topic ID or list of topic IDs for direct matching (from BERTopic classification).
        :param limit: Maximum number of results to return for semantic search.
        :return: Dictionary containing both semantic and direct match results.
        """
        try:
            results = {"semantic_results": [], "direct_results": []}
            
            # Direct matching via topic_ids
            if topic_ids is not None:
                direct_filter = self.build_filter({})
                # Handle both single topic_id and list of topic_ids
                if isinstance(topic_ids, list):
                    # For multiple topic IDs, use MatchAny
                    topic_id_condition = FieldCondition(key="topic_id", match=MatchAny(any=topic_ids))
                else:
                    # For single topic ID, use MatchValue
                    topic_id_condition = FieldCondition(key="topic_id", match=MatchValue(value=topic_ids))
                    
                if direct_filter.must:
                    direct_filter.must.append(topic_id_condition)
                else:
                    direct_filter.must = [topic_id_condition]
                # Get direct matches
                direct_response = self.client.scroll(
                    collection_name=self.topics_collection,
                    scroll_filter=direct_filter,
                    with_payload=True,
                )
                # Format direct results with score 1.0
                for point in direct_response[0]:
                    results["direct_results"].append({
                        "score": 1.0,  # Direct matches get perfect score
                        "topic": point.payload
                    })
                    
            # Semantic search on topic descriptions
            if topic_query:
                query_vector = self.embedding_model.encode(topic_query).tolist()
                
                semantic_response = self.client.query_points(
                    collection_name=self.topics_collection,
                    query=query_vector,
                    using="description",
                    limit=limit,
                    with_payload=True,
                )
                
                # Format semantic results with actual scores
                for point in semantic_response.points:
                    results["semantic_results"].append({
                        "score": point.score,
                        "topic": point.payload
                    })
                        
            return {
                "semantic_results": results["semantic_results"],
                "direct_results": results["direct_results"],
                "result_counts": {
                    "direct_results": len(results["direct_results"]),
                    "semantic_results": len(results["semantic_results"])
                }
            }
        except Exception as e:
            print(e, flush=True)
            return {"error": str(e)}

    def search_chunks(self, query, topic_ids=None, metadata={}, prob_threshold=0.1):
        """
        Searches for document chunks matching the query, optionally filtered by topic IDs.
        :param query: Query text for semantic search on chunk content.
        :param topic_ids: Optional list of topic IDs to filter chunks by.
        :param metadata: Optional metadata for filtering.
        :param prob_threshold: Minimum probability threshold for topic relevance (default: 0.1)
        :return: Dictionary containing search results.
        """
        try:
            # Build metadata filter
            payload_filter = self.build_filter(metadata)
            
            # If topic IDs provided, add to filter
            if topic_ids:
                # Ensure topic_ids are strings
                topic_ids = [str(topic) for topic in topic_ids]
                
                # Create filters for each topic ID
                topic_filters = []
                for topic_id in topic_ids:
                    # Filter for documents where the topic ID exists as a key in topic_probabilities
                    # and has a probability greater than the threshold
                    topic_filter = FieldCondition(
                        key=f"topic_probabilities.{topic_id}",
                        range=Range(gt=prob_threshold)
                    )
                    topic_filters.append(topic_filter)
                
                # Add topic filters to the should array (OR condition)
                if payload_filter.should is None:
                    payload_filter = Filter(
                        must=payload_filter.must,
                        should=topic_filters,
                        must_not=payload_filter.must_not
                    )
                else:
                    payload_filter.should.extend(topic_filters)
            
            # Generate vector for query
            query_vector = self.embedding_model.encode(query).tolist()
            
            # Search chunks
            response = self.client.query_points(
                collection_name=self.chunks_collection,
                query=query_vector,
                using="content",
                limit=10,
                with_payload=True,
                query_filter=payload_filter
            )
            
            return {"results": response}
        except Exception as e:
            print(e, flush=True)
            return {"error": str(e)}

    def hybrid_search(self, query, topic_id=None, topic_label=None, metadata={}):
        """
        Performs a two-phase search:
        1. First identifies relevant topics (by ID/label and/or semantic search)
        2. Then searches chunks within those topics

        :param query: Query text for searching.
        :param topic_id: Optional topic ID for direct matching.
        :param topic_label: Optional topic label for direct matching.
        :param metadata: Optional metadata for filtering both topics and chunks.
        :return: Dictionary containing hierarchically organized search results.
        """
        try:
            # Phase 1: Search for relevant topics
            topic_results = self.search_topics(
                topic_query=query,
                topic_id=topic_id,
                topic_label=topic_label,
                metadata=metadata
            )
            
            if "error" in topic_results:
                return topic_results
            
            # Extract topic IDs from results
            topic_ids = []
            topics_data = {}
            
            for topic in topic_results.get("results", []):
                tid = topic.payload.get("topic_id")
                if tid:
                    topic_ids.append(tid)
                    topics_data[tid] = {
                        "topic_id": tid,
                        "topic_label": topic.payload.get("topic_label"),
                        "topic_description": topic.payload.get("topic_description"),
                        "score": topic.score if hasattr(topic, "score") else 1.0,
                        "chunks": []
                    }
            
            # Phase 2: Search for chunks within the identified topics
            if topic_ids:
                chunk_results = self.search_chunks(
                    query=query,
                    topic_ids=topic_ids,
                    metadata=metadata
                )
                
                if "error" in chunk_results:
                    return chunk_results
                
                # Organize chunks by their associated topics
                for chunk in chunk_results.get("results", []):
                    chunk_topic_ids = chunk.payload.get("topic_ids", [])
                    
                    # Add this chunk to each of its associated topics that we found
                    for tid in chunk_topic_ids:
                        if tid in topics_data:
                            topics_data[tid]["chunks"].append({
                                "text": chunk.payload.get("chunk_text"),
                                "document_id": chunk.payload.get("document_id"),
                                "position": chunk.payload.get("position"),
                                "score": chunk.score
                            })
            
            # Organize results in a hierarchical structure
            organized_results = {
                "topics": list(topics_data.values()),
                "total_topics": len(topics_data),
                "total_chunks": sum(len(topic_data["chunks"]) for topic_data in topics_data.values())
            }
            
            return organized_results

        except Exception as e:
            return {"error": str(e)}