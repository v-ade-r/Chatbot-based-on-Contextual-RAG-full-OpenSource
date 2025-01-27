
from concurrent.futures import ThreadPoolExecutor, as_completed
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from langchain_community.llms import Ollama
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm


class ContextualVectorDB:
    def __init__(self, name: str) -> None:
        self.name = name
        self.db_path = f"./data/{name}/chromadb"
        self.metadata = []
        self.client = PersistentClient(path=self.db_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
        self.collection = self.client.get_or_create_collection(name=self.name, embedding_function=self.embedding_fn)
        self.llm = Ollama(model="llama3.1")

    def situate_context(self, doc: str, chunk: str) -> str:
        """Generate context for a chunk"""
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

    def load_data(self, dataset: List[Dict[str, Any]], parallel_threads: int = 4) -> None:
        if self.collection.count() > 0:
            print("Vector database is already loaded. Skipping data loading.")
            self.metadata = self.collection.get(include=['metadatas'])
            return

        texts_to_embed = []
        metadata = []
        total_chunks = sum(len(doc['chunks']) for doc in dataset)

        def process_chunk(doc: Dict[str, Any], chunk: Dict[str, Any]) -> Dict[str, Any]:
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

    def _embed_and_store(self, texts: List[str], data: List[Dict[str, Any]]) -> None:
        """Generate embeddings and store them in the ChromaDB collection."""
        batch_size = 128
        with tqdm(total=len(texts), desc="Batching") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i: i + batch_size]
                batch_metadata = data[i: i + batch_size]

                self.collection.add(
                    ids=[f"doc_{idx}" for idx in range(i, i + len(batch_texts))],
                    documents=batch_texts,
                    metadatas=batch_metadata
                )
                pbar.update(len(batch_texts))
        self.metadata = self.collection.get(include=['metadatas'])
        print(f"Total documents in database: {self.collection.count()}")

    def append_data(self, new_dataset: List[Dict[str, Any]], parallel_threads: int = 4) -> None:
        if not new_dataset:
            print("No data to add.")
            return

        existing_count = self.collection.count()
        print(f"Current documents number in the database: {existing_count}")

        total_chunks = sum(len(doc['chunks']) for doc in new_dataset)

        def process_chunk(doc: Dict[str, Any], chunk: Dict[str, Any]) -> Dict[str, Any]:
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

        texts_to_embed = []
        metadata = []

        print(f"Processing {total_chunks} chunks with {parallel_threads} threads")
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = []
            for doc in new_dataset:
                for chunk in doc['chunks']:
                    futures.append(executor.submit(process_chunk, doc, chunk))

            for future in tqdm(as_completed(futures), total=total_chunks, desc="Processing chunks"):
                result = future.result()
                texts_to_embed.append(result['text_to_embed'])
                metadata.append(result['metadata'])

        self._append_embed_and_store(texts_to_embed, metadata, existing_count)
        print(f"Total chunks added: {len(texts_to_embed)}")

    def _append_embed_and_store(self, texts: List[str], data: List[Dict[str, Any]], offset: int = 0) -> None:
        batch_size = 128
        with tqdm(total=len(texts), desc="Batching") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i: i + batch_size]
                batch_metadata = data[i: i + batch_size]
                batch_ids = [f"doc_{offset + j}" for j in range(i, i + len(batch_texts))]

                self.collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadata
                )
                pbar.update(len(batch_texts))

        self.metadata = self.collection.get(include=['metadatas'])
        print(f"Total documents in database: {self.collection.count()}")

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
