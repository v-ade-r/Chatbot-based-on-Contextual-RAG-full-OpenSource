
from pdf2image import convert_from_path
import pytesseract
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
from typing import Any, List, Dict


def upload_ocr(file: Any, folder: str) -> str:
    file_path = os.path.join(folder, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    images = convert_from_path(file_path)
    text = ''
    for image in images:
        raw_text = pytesseract.image_to_string(image, lang='eng')
        text += raw_text
    return text


def upload_create_chunks(db: Any, file: Any, folder: str, chunk_size: int = 512, overlap: int = 32) -> (
        List)[Dict[str, Any]]:

    doc_id = len(db.collection.get()['ids']) + 1
    text = upload_ocr(file, folder)
    original_uuid = hashlib.sha256(text.encode("utf-8")).hexdigest()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = text_splitter.create_documents([text])

    docs_data = []
    chunks_data = []
    for i, doc in enumerate(docs):
        chunk_dict = {
            "chunk_id": f"doc_{doc_id}_chunk_{i}",
            "original_index": i,
            "content": doc.page_content
        }
        chunks_data.append(chunk_dict)

    print(f"Chunks data: {chunks_data}")
    doc_data = {
        "doc_id": f"doc_{doc_id}",
        "original_uuid": original_uuid,
        "content": text,
        "chunks": chunks_data
    }

    docs_data.append(doc_data)
    print(f"Chunks data after ocr: {docs_data}")
    return docs_data


def save_chunks_to_file(chunks: List[Dict[str, Any]], output_file_name: str, output_path: str = "data/") -> None:
    output_file_path = os.path.join(output_path, output_file_name)
    with open(output_file_path, "w", encoding="utf-8") as json_file:
        json.dump(chunks, json_file, ensure_ascii=False, indent=2)
    print(f"Saved into: {output_file_path}")


def merge_context(response: List[Dict[str, Any]]) -> str:
    full_context = ""
    for doc in response:
        context = doc['chunk']['original_content'].strip()
        full_context = "\n\n".join([full_context, context])
    return full_context
