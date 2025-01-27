
import json
from typing import List, Dict, Any
from tqdm import tqdm

from contextual_vector_db import ContextualVectorDB
from bm25 import  create_elasticsearch_bm25_index
from retrieval import retrieve_advanced


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def evaluate_db_advanced(db: ContextualVectorDB, original_jsonl_path: str, k: int) -> Dict[str, Any]:
    original_data = load_jsonl(original_jsonl_path)
    es_bm25 = create_elasticsearch_bm25_index(db)

    try:
        # Warm-up queries
        warm_up_queries = original_data[:10]
        for query_item in warm_up_queries:
            _ = retrieve_advanced(query_item['query'], db, es_bm25, k)

        total_score = 0
        total_semantic_count = 0
        total_bm25_count = 0
        total_results = 0

        for query_item in tqdm(original_data, desc="Evaluating retrieval"):
            query = query_item['query']
            golden_chunk_uuids = query_item['golden_chunk_uuids']

            golden_contents = []
            for doc_uuid, chunk_index in golden_chunk_uuids:
                golden_doc = next((doc for doc in query_item['golden_documents'] if doc['uuid'] == doc_uuid), None)
                if golden_doc:
                    golden_chunk = next((chunk for chunk in golden_doc['chunks'] if chunk['index'] == chunk_index),
                                        None)
                    if golden_chunk:
                        golden_contents.append(golden_chunk['content'].strip())

            if not golden_contents:
                print(f"Warning: No golden contents found for query: {query}")
                continue

            retrieved_docs, semantic_count, bm25_count = retrieve_advanced(query, db, es_bm25, k)

            chunks_found = 0
            for golden_content in golden_contents:
                for doc in retrieved_docs[:k]:
                    retrieved_content = doc['chunk']['original_content'].strip()
                    if retrieved_content == golden_content:
                        chunks_found += 1
                        break

            query_score = chunks_found / len(golden_contents)
            total_score += query_score

            total_semantic_count += semantic_count
            total_bm25_count += bm25_count
            total_results += len(retrieved_docs)

        total_queries = len(original_data)
        average_score = total_score / total_queries
        pass_at_n = average_score * 100

        semantic_percentage = (total_semantic_count / total_results) * 100 if total_results > 0 else 0
        bm25_percentage = (total_bm25_count / total_results) * 100 if total_results > 0 else 0

        results = {
            "pass_at_n": pass_at_n,
            "average_score": average_score,
            "total_queries": total_queries
        }

        print(f"Pass@{k}: {pass_at_n:.2f}%")
        print(f"Average Score: {average_score:.2f}")
        print(f"Total queries: {total_queries}")
        print(f"Percentage of results from semantic search: {semantic_percentage:.2f}%")
        print(f"Percentage of results from BM25: {bm25_percentage:.2f}%")

        return results

    finally:
        # Delete the Elasticsearch index
        if es_bm25.es_client.indices.exists(index=es_bm25.index_name):
            es_bm25.es_client.indices.delete(index=es_bm25.index_name)
            print(f"Deleted Elasticsearch index: {es_bm25.index_name}")


if __name__ == "__main__":
    with open('data/codebase_chunks.json', 'r', encoding="utf-8") as f:
        transformed_dataset = json.load(f)

    contextual_db = ContextualVectorDB("my_contextual_db")
    contextual_db.load_data(transformed_dataset)
    results5 = evaluate_db_advanced(contextual_db, 'data/evaluation_set.jsonl', 5)
    results20 = evaluate_db_advanced(contextual_db, 'data/evaluation_set.jsonl', 20)
