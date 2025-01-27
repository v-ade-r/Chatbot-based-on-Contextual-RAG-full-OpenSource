from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn
import json
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Dict

from inference_by_Ollama import get_response, summarize_dialog
from contextual_vector_db import ContextualVectorDB
from bm25 import create_elasticsearch_bm25_index
from ocr_and_chunking import merge_context, upload_create_chunks, save_chunks_to_file
from retrieval import retrieve_advanced


UPLOAD_FOLDER = "Files_uploaded"
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global db, es_bm25, last_summary
    last_summary = ""
    with open('data/test_chunks_trimmed2.json', 'r', encoding='utf-8') as f:
        transformed_dataset = json.load(f)

    db = ContextualVectorDB("my_contextual_db_final_test2")
    db.load_data(transformed_dataset)
    es_bm25 = create_elasticsearch_bm25_index(db)
    yield

app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    input: str


@app.post("/chat")
def chat(query: ChatRequest) -> Dict:
    print(f"Query: {query.input}")
    global last_summary
    extended_query = f"{last_summary}\n\nUser question: {query.input}"
    print(f"Extended query: {extended_query}")

    retrieved_context, _, _ = retrieve_advanced(extended_query, db, es_bm25, 10)
    context = merge_context(retrieved_context)
    response = get_response(extended_query, context)
    summary = summarize_dialog(query.input, response["content"])
    last_summary = summary
    return response


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)) -> Dict:
    global es_bm25
    if not file.filename.lower().endswith(".pdf"):
        return {"status": "error", "message": "It's not a pdf file."}

    chunks = upload_create_chunks(db, file, UPLOAD_FOLDER)
    json_name = file.filename.replace('.pdf', '.json')
    save_chunks_to_file(chunks, f"{json_name}")

    with open(f"data/{json_name}", 'r', encoding="utf-8") as f:
        uploaded_data = json.load(f)
    db.append_data(uploaded_data)
    es_bm25 = create_elasticsearch_bm25_index(db)
    return {"status": "success", "message": f"File {json_name} converted and added to the database"}


@app.post("/reset_chat")
def reset_chat() -> Dict:
    global last_summary
    last_summary = ""
    return {"status": "success", "message": "New chat ready."}


@app.get("/", response_class=HTMLResponse)
def chat_page() -> str:
    return """
    <!DOCTYPE html>
<html>
<head>
    <title>Chatbot with Conversation History Awareness based on Contextual RAG with Hybrid Search</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        #chatBox { border: 1px solid #ccc; padding: 10px; width: 500px; height: 300px; overflow-y: auto; }
        #userInput { width: 400px; }

        button {
            background-color: #007bff;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
        }

        button:hover {
            background-color: #0056b3;
        }

        label {
            background-color: #007bff;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            cursor: pointer;
            display: inline-block;
        }

        label:hover {
            background-color: #0056b3;
        }

        input[type="file"] {
            display: none;
        }

        #fileName {
            margin-left: 10px;
            font-style: italic;
        }

        #uploadStatus {
            margin-top: 10px;
            font-weight: bold;
            color: #007bff;
        }
    </style>
</head>
<body>
    <h1>Chatbot with Conversation History Awareness based on Contextual RAG with Hybrid Search</h1>

    <!-- Chat window -->
    <div id="chatBox"></div>
    <br>
    <input type="text" id="userInput" placeholder="Type your message here..." onkeypress="handleKeyPress(event)">
    <button onclick="sendMessage()">Send</button>
    <button onclick="resetChat()">New Query</button> 

    <hr>

    <!-- PDF Upload -->
    <h2>Add a PDF file </h2>
    <form id="uploadForm">
        <label for="pdfFileInput">Choose a file</label>
        <input type="file" id="pdfFileInput" accept="application/pdf" onchange="displayFileName()" />
        <span id="fileName">No file selected</span>
        <button type="button" onclick="uploadPDF()">Upload PDF</button>
        <div id="uploadStatus"></div>
    </form>

    <script>
        // Display selected file name
        function displayFileName() {
            const pdfFile = document.getElementById("pdfFileInput").files[0];
            const fileNameDisplay = document.getElementById("fileName");
            if (pdfFile) {
                fileNameDisplay.textContent = pdfFile.name;
            } else {
                fileNameDisplay.textContent = "No file selected";
            }
        }

        // Sending question to the endpoint
        async function sendMessage() {
            const userInput = document.getElementById("userInput").value;
            if (!userInput) return;

            const chatBox = document.getElementById("chatBox");
            chatBox.innerHTML += `<p><strong>You:</strong> ${userInput}</p>`;

            const response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ input: userInput })
            });

            const data = await response.json();
            const botResponse = data.content;
            chatBox.innerHTML += `<p><strong>Assistant:</strong> ${botResponse}</p>`;
            document.getElementById("userInput").value = "";
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        // Handle Enter key press
        function handleKeyPress(event) {
            if (event.key === "Enter") {
                sendMessage();
            }
        }

        // Creating new chat
        async function resetChat() {
            const response = await fetch("/reset_chat", {
                method: "POST"
            });

            const data = await response.json();
            if (data.status === "success") {
                // New chat is prepared
                const chatBox = document.getElementById("chatBox");
                chatBox.innerHTML = "";
            } else {
                alert("Failed to reset chat history.");
            }
        }

        // Sending pdf
        async function uploadPDF() {
            const pdfFile = document.getElementById("pdfFileInput").files[0];
            const uploadStatus = document.getElementById("uploadStatus");

            if (!pdfFile) {
                alert("No file selected!");
                return;
            }

            const formData = new FormData();
            formData.append('file', pdfFile);

            uploadStatus.textContent = "Upload in progress...";

            try {
                const response = await fetch("/upload_pdf", {
                    method: "POST",
                    body: formData
                });
                const data = await response.json();

                if (response.ok) {
                    uploadStatus.textContent = "Success: " + data.message;
                    setTimeout(() => uploadStatus.textContent = "", 3000);
                } else {
                    uploadStatus.textContent = "Error: " + data.message;
                }
            } catch (error) {
                console.error("Error:", error);
                uploadStatus.textContent = "Error during the file upload.";
            }
        }
    </script>
</body>
</html>
    """


if __name__ == '__main__':
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
    )
