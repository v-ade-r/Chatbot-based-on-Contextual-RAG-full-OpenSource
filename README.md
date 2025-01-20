# Contextual-RAG-full-OpenSource
Contextual RAG with Hybrid Search and Reranking, full OpenSource.

## **Justification of this code**
The idea is to create a fully functional, completely open source Contextual RAG with Hybrid Search and Reranking application.

## **Current project situation**
Right know I made a few tweaks in the example code from Anthropic post about Contextual RAG.
The most important is a swap to open source tools:

1. Model creating context:                    Anthropic Claude  ->   Ollama/llama3.1
2. Model creating embeddings:                 VoyageAI  ->  all-mpnet-base-v2 (used with chromadb vector database)
3. Model for reranking search results:        Cohere  ->  Flashrank

The code allows currently only to evaluate the whole approach on the data prepared by anthropic.

## **Usage tips**
1. Download and install Ollama
2. (Download the Llama3.1 model) In command line type: ollama run llama3.1
3. In command line type: Ollama serve
4. Download Docker and set it up.
5. Open Docker Desktop
6. In cmd type: docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:8.8.0
7. Good to go. Let's evaluate it!

## References
https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb
