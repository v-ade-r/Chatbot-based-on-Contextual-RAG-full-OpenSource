# Contextual-RAG-full-OpenSource
Contextual RAG with Hybrid Search and Reranking, full OpenSource.

## **Justification of this code**
The idea is to create a fully functional, completely open source Contextual RAG with Hybrid Search and Reranking application.

## **Current project situation**
Right know I made a few tweaks in the example code from Anthropic post about Contextual RAG.
The most important is a swap to open source tools:

1. **Model creating context:**                    **Anthropic Claude**  ->   **Ollama/llama3.1 **(sometimes input exceeds context length, because of the default number set by Ollama - I have to try to change it. Because of that sometimes created context is wrong or incomplete/empty)
2. **Model creating embeddings:**                 **VoyageAI**  ->  **all-mpnet-base-v2** (used with chromadb vector database)
3. **Model for reranking search results:**        **Cohere**  ->  **Flashrank**

The code allows currently only to evaluate the whole approach on the data prepared by anthropic.

## **Usage tips**
1. Download and install Ollama
2. (Download the Llama3.1 model) In command line type: ollama run llama3.1
3. In command line type: Ollama serve
4. Download Docker and set it up.
5. Open Docker Desktop
6. In cmd type: docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:8.8.0
7. Good to go. Let's evaluate the retrieving capabilities!

## **Retrieval evaluation** 
I evaluated the retrieving accuracy at each stage. Pass@n - describes accuracy of getting 'golden chunk' (ideal chunk consisting of the most appropriate knowledge for the query) i top-n (top5 and top20) retrieved chunks.

VectorDB (only semantic search):
Pass@5: 63.76%
Pass@20: 79.66%

ContextualVectorDB:
Pass@5: 69.84%
Pass@20: 83.13%

ContextualVectorDB + BM25 (adding BM25 creates Hybrid Search):
Pass@5: 76.53%
Pass@20: 87.37%

ContextualVectorDB + BM25 + Reranker:
Pass@5: 81.32%
Pass@20: 90.83%

## References
https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb
