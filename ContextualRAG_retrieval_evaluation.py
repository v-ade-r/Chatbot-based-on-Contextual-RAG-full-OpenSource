
from concurrent.futures import ThreadPoolExecutor, as_completed
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from langchain_community.llms import Ollama
import numpy as np
from flashrank import Ranker, RerankRequest
from elasticsearch.helpers import bulk, BulkIndexError
import json
from typing import List, Dict, Any
from tqdm import tqdm
from elasticsearch import Elasticsearch


class ContextualVectorDB:
    def __init__(self, name: str):
        self.name = name
        self.db_path = f"./data/{name}/chromadb"
        self.metadata = []
        self.client = PersistentClient(path=self.db_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
        self.collection = self.client.get_or_create_collection(name=self.name, embedding_function=self.embedding_fn)
        self.llm = Ollama(model="llama3.1")

    def situate_context(self, doc: str, chunk: str) -> str:
        """Generate context for a chunk using OLlama and llama3.1 model."""
        document_prompt = f"""
        <document>
        {doc}
        </document>
        """

        chunk_prompt = f"""
        Here is the chunk we want to situate within the whole document:
        <chunk>
        {chunk}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

        combined_prompt = f"{document_prompt}\n{chunk_prompt}"
        response = self.llm.invoke(combined_prompt)
        return response

    def load_data(self, dataset: List[Dict[str, Any]], parallel_threads: int = 4):
        if self.collection.count() > 0:
            print("Vector database is already loaded. Skipping data loading.")
            self.metadata = self.collection.get(include=['metadatas'])
            return

        texts_to_embed = []
        metadata = []
        total_chunks = sum(len(doc['chunks']) for doc in dataset)

        def process_chunk(doc, chunk):
            contextualized_text = self.situate_context(doc['content'], chunk['content'])
            return {
                'text_to_embed': f"{chunk['content']}\n\n{contextualized_text}",
                'metadata': {
                    'doc_id': doc['doc_id'],
                    'chunk_id': chunk['chunk_id'],
                    'original_index': chunk['original_index'],
                    'original_content': chunk['content'],
                    'contextualized_content': contextualized_text
                }
            }

        print(f"Processing {total_chunks} chunks with {parallel_threads} threads")
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = []
            for doc in dataset:
                for chunk in doc['chunks']:
                    futures.append(executor.submit(process_chunk, doc, chunk))

            for future in tqdm(as_completed(futures), total=total_chunks, desc="Processing chunks"):
                result = future.result()
                texts_to_embed.append(result['text_to_embed'])
                metadata.append(result['metadata'])

        self._embed_and_store(texts_to_embed, metadata)
        print(f"Contextual Vector database loaded and saved. Total chunks processed: {len(texts_to_embed)}")

    def _embed_and_store(self, texts: List[str], data: List[Dict[str, Any]]):
        """Generate embeddings and store them in the ChromaDB collection."""
        batch_size = 128
        with tqdm(total=len(texts), desc="Embedding chunks") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i: i + batch_size]
                batch_metadata = data[i: i + batch_size]

                self.collection.add(
                    ids=[f"doc_{idx}" for idx in range(i, i + len(batch_texts))],
                    documents=batch_texts,
                    metadatas=batch_metadata
                )
                pbar.update(len(batch_texts))
        self.metadata = data


    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """Search for the most relevant chunks based on the query."""
        query_embedding = self.embedding_fn([query])[0]

        documents = self.collection.get(include=['metadatas', 'embeddings'])
        document_embeddings = documents.get("embeddings", None)
        metadata = documents.get("metadatas", None)

        if document_embeddings is None or metadata is None:
            raise ValueError("Embeddings or metadata are missing from the collection.")

        document_embeddings = np.array(document_embeddings)
        similarities = np.dot(document_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]

        top_results = []
        for idx in top_indices:
            single_metadata = metadata[idx]
            result = {
                "metadata": single_metadata,
                "original_content": single_metadata["original_content"],
                "contextualized_content": single_metadata["contextualized_content"],
                "similarity": float(similarities[idx]),
            }
            top_results.append(result)
        return top_results


class ElasticsearchBM25:
    def __init__(self, index_name: str = "contextual_bm25_index"):
        self.es_client = Elasticsearch("http://localhost:9200")
        self.index_name = index_name
        self.create_index()

    def create_index(self):
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

    def index_documents(self, documents: List[Dict[str, Any]]):
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
            success, _ = bulk(self.es_client, actions)
        except BulkIndexError as e:
            for error in e.errors:
                print(f"Indexing error for document: {error.get('index', {}).get('_source', {})}")
                print(f"Error details: {error}")
            raise
        self.es_client.indices.refresh(index=self.index_name)
        return success

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


def create_elasticsearch_bm25_index(db: ContextualVectorDB):
    es_bm25 = ElasticsearchBM25()
    es_bm25.index_documents(db.metadata['metadatas'] )
    return es_bm25


def retrieve_advanced(query: str, db: ContextualVectorDB, es_bm25: ElasticsearchBM25, k: int,
                      semantic_weight: float = 0.8, bm25_weight: float = 0.2):

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

    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/opt")
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked_results = ranker.rerank(rerank_request)

    reranked_results_sorted = sorted(reranked_results, key=lambda x: x["score"], reverse=True)
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
    return final_results, semantic_count, bm25_count


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def evaluate_db_advanced(db: ContextualVectorDB, original_jsonl_path: str, k: int) -> Dict:
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
        if es_bm25.es_client.indices.exists(index=es_bm25.index_name):
            es_bm25.es_client.indices.delete(index=es_bm25.index_name)
            print(f"Deleted Elasticsearch index: {es_bm25.index_name}")


with open('data/codebase_chunks.json', 'r', encoding='utf-8') as f:
    transformed_dataset = json.load(f)


contextual_db = ContextualVectorDB("my_contextual_db")
contextual_db.load_data(transformed_dataset)

results5 = evaluate_db_advanced(contextual_db, 'data/evaluation_set.jsonl', 5)
results20 = evaluate_db_advanced(contextual_db, 'data/evaluation_set.jsonl', 20)
