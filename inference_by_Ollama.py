
import time
from langchain_ollama import OllamaLLM
from typing import Dict


model_id = "Llama3.1"
ollama = OllamaLLM(model=model_id, max_tokens=512, temperature=0.0)


def get_response(user_input: str, context: str) -> Dict[str, str]:
    t0 = time.time()
    system_msg = f"You are a helpful assistant. Answer the question ONLY based on the context.\nContext: {context}\n"
    user_msg = f"{user_input}\n"
    prompt = f"System: {system_msg}\nUser: {user_msg}"

    print("Generating answer")
    llm_result = ollama.generate(prompts=[prompt])
    response = llm_result.generations[0][0].text
    print(f"{time.time() - t0} sec")
    print(f"Model response: {response}")
    return {"role": "user", "content": f"{response}\n"}


def summarize_dialog(user_question: str, assistant_answer: str) -> str:
    summary_prompt = (
        "Please summarize the following conversation in two sentences:\n\n"
        f"User: {user_question}\nAssistant: {assistant_answer}\n\n"
    )
    summary_result = ollama.generate(prompts=[summary_prompt])
    summary_text = summary_result.generations[0][0].text.strip()
    print(f"Summary text: {summary_text}")
    return summary_text
