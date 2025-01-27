# Chatbot based on Contextual RAG with Hybrid Search
Chatbot with Conversation History Awareness based on Contextual RAG with Hybrid Search and Reranking, fully OpenSource.

## **Justification of this code**
The idea was to create a fully functional, completely open source Contextual RAG with Hybrid Search and Reranking application.

## Some idea and code explanations 
Todo

## **Usage tips**
1. Download and install Ollama
2. (Download the Llama3.1 model) In command line type: ollama run llama3.1
3. In command line type: Ollama serve
4. Download Docker and set it up.
5. Open Docker Desktop
6. In cmd type: docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:8.8.0
7. Good to go. Let's evaluate the retrieving capabilities running evaluate.py!

## **Retrieval evaluation** 
The current code is capable of evaluating the approach using data structurized like the data prepared by Anthropic.

I evaluated the retrieving accuracy at each stage. Pass@n - represents the accuracy of getting the 'golden chunk' (most relevant chunk for the query) within the top-n (top5 and top20) retrieved chunks.

VectorDB (only semantic search):<br>
Pass@5: 63.76%<br>
Pass@20: 79.66%<br>

ContextualVectorDB:<br>
Pass@5: 69.84%<br>
Pass@20: 83.13%<br>

ContextualVectorDB + BM25 (adding BM25 creates Hybrid Search):<br>
Pass@5: 76.53%<br>
Pass@20: 87.37%<br>

ContextualVectorDB + BM25 + Reranker:<br>
Pass@5: 81.32%<br>
Pass@20: 90.83%<br>

------------------
(Context created by GPT-4o-mini)<br>
Contextual + BM25 + reranker:<br>
Pass@5: 81.99%<br>
Pass@20: 93.75%<br>

## References
https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb
