
from flashrank import Ranker, RerankRequest
from contextual_vector_db import ContextualVectorDB
from bm25 import ElasticsearchBM25
from typing import List, Dict, Any, Tuple


def retrieve_advanced(
    query: str,
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    semantic_weight: float = 0.8,
    bm25_weight: float = 0.2
) -> Tuple[List[Dict[str, Any]], float, float]:

    num_chunks_to_recall = k*10
    semantic_results = db.search(query, k=num_chunks_to_recall)
    ranked_chunk_ids_semantic = [
        (res["metadata"]["doc_id"], res["metadata"]["original_index"])
        for res in semantic_results
    ]

    bm25_results = es_bm25.search(query, k=num_chunks_to_recall)
    ranked_chunk_ids_bm25 = [
        (res["doc_id"], res["original_index"])
        for res in bm25_results
    ]

    all_chunk_ids = list(set(ranked_chunk_ids_semantic + ranked_chunk_ids_bm25))

    chunk_id_to_score = {}
    for chunk_id in all_chunk_ids:
        score = 0.0

        if chunk_id in ranked_chunk_ids_semantic:
            idx_sem = ranked_chunk_ids_semantic.index(chunk_id)
            score += semantic_weight * (1 / (idx_sem + 1))

        if chunk_id in ranked_chunk_ids_bm25:
            idx_bm = ranked_chunk_ids_bm25.index(chunk_id)
            score += bm25_weight * (1 / (idx_bm + 1))

        chunk_id_to_score[chunk_id] = score

    sorted_chunk_ids = sorted(
        chunk_id_to_score.keys(),
        key=lambda x: (chunk_id_to_score[x], x[0], x[1]),
        reverse=True
    )

    top_chunk_ids = sorted_chunk_ids[:num_chunks_to_recall]
    chunk_data = {}

    for res in semantic_results:
        c_id = (res["metadata"]["doc_id"], res["metadata"]["original_index"])
        chunk_data[c_id] = res["metadata"]

    for res in bm25_results:
        c_id = (res["doc_id"], res["original_index"])
        if c_id not in chunk_data:
            chunk_data[c_id] = next(
                m for m in db.metadata['metadatas']
                if m["doc_id"] == c_id[0] and m["original_index"] == c_id[1]
            )

    passages = []
    for idx, c_id in enumerate(top_chunk_ids):
        metadata = chunk_data[c_id]
        combined_text = (
                metadata.get("original_content", "")
                + "\n\nContext: " +
                metadata.get("contextualized_content", "")
        )
        passages.append({
            "id": idx,
            "text": combined_text
        })
    print(f"Final passages: {passages}")

    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/opt")
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked_results = ranker.rerank(rerank_request)
    reranked_results_sorted = sorted(reranked_results, key=lambda x: x["score"], reverse=True)
    print(f"Reranked passages: {reranked_results_sorted}")

    final_results = []
    semantic_count = 0
    bm25_count = 0

    for res in reranked_results_sorted[:k]:
        original_idx = res["id"]
        chunk_id = top_chunk_ids[original_idx]
        chunk_metadata = chunk_data[chunk_id]

        is_from_semantic = (chunk_id in ranked_chunk_ids_semantic)
        is_from_bm25 = (chunk_id in ranked_chunk_ids_bm25)

        if is_from_semantic and not is_from_bm25:
            semantic_count += 1
        elif is_from_bm25 and not is_from_semantic:
            bm25_count += 1
        else:
            semantic_count += 0.5
            bm25_count += 0.5

        final_results.append({
            "chunk": chunk_metadata,
            "score": res["score"],
            "from_semantic": is_from_semantic,
            "from_bm25": is_from_bm25
        })
    print(f"Final after rerank: {final_results}")
    return final_results, semantic_count, bm25_count
