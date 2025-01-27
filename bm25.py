
from elasticsearch.helpers import bulk, BulkIndexError
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from contextual_vector_db import ContextualVectorDB


class ElasticsearchBM25:
    def __init__(self, index_name: str = "contextual_bm25_index") -> None:
        self.es_client = Elasticsearch("http://localhost:9200")
        if self.es_client.indices.exists(index="contextual_bm25_index"):
            self.es_client.indices.delete(index="contextual_bm25_index")
            print("Index deleted.")   # in case of a problem with indexes, uncomment these lines
        self.index_name = index_name
        self.create_index()

    def create_index(self) -> None:
        index_settings = {
            "settings": {
                "analysis": {"analyzer": {"default": {"type": "english"}}},
                "similarity": {"default": {"type": "BM25"}},
                "index.queries.cache.enabled": False
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "english"},
                    "contextualized_content": {"type": "text", "analyzer": "english"},
                    "doc_id": {"type": "keyword", "index": False},
                    "chunk_id": {"type": "keyword", "index": False},
                    "original_index": {"type": "keyword", "index": False},
                }
            },
        }
        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(index=self.index_name, body=index_settings)
            print(f"Created index: {self.index_name}")

    def index_documents(self, documents: List[Dict[str, Any]]) -> int:
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "content": doc["original_content"],
                    "contextualized_content": doc["contextualized_content"],
                    "doc_id": doc["doc_id"],
                    "chunk_id": doc["chunk_id"],
                    "original_index": doc["original_index"],
                },
            }
            for doc in documents
        ]

        try:
            results, _ = bulk(self.es_client, actions)
        except BulkIndexError as e:
            for error in e.errors:
                print(f"Indexing error for document: {error.get('index', {}).get('_source', {})}")
                print(f"Error details: {error}")
            raise
        self.es_client.indices.refresh(index=self.index_name)
        return results

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        self.es_client.indices.refresh(index=self.index_name)
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "contextualized_content"],
                }
            },
            "size": k,
        }
        response = self.es_client.search(index=self.index_name, body=search_body)
        return [
            {
                "doc_id": hit["_source"]["doc_id"],
                "original_index": hit["_source"]["original_index"],
                "content": hit["_source"]["content"],
                "contextualized_content": hit["_source"]["contextualized_content"],
                "score": hit["_score"],
            }
            for hit in response["hits"]["hits"]
        ]


def create_elasticsearch_bm25_index(db: ContextualVectorDB) -> ElasticsearchBM25:
    es_bm25 = ElasticsearchBM25()
    es_bm25.index_documents(db.metadata['metadatas'])
    return es_bm25
