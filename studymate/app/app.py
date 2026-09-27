"""
The main chatbot app: combines LOCAL document retrieval (Chroma) with LIVE
web search (Tavily), runs them in PARALLEL to save time, streams the answer
token-by-token from Ollama (CPU-friendly, no GPU/CUDA required), and cites
both local and web sources.

PREREQUISITES:
1. Install Ollama (https://ollama.com/download) and pull a model:
       ollama pull qwen2.5:1.5b
2. Get a free Tavily API key at https://tavily.com and put it in a .env file.
3. (Optional) Put documents in data/documents/ and run: python rag/ingest.py

Run: python app/app.py
"""

import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.append(str(Path(__file__).parent.parent))

import gradio as gr
import requests

from rag.retriever import retrieve_local_context, local_index_exists
from search.web_search import search_web

# ---- Config ----
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:1.5b"  # switch to your fine-tuned model name after Colab fine-tuning

# Speed-related settings:
MAX_RESPONSE_TOKENS = 300      # caps generation length (num_predict)
CONTEXT_WINDOW = 2048          # num_ctx -- lower if your context is short
KEEP_ALIVE = "30m"             # keeps model loaded in RAM between questions,
                                # avoiding reload cost on every message

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using the "
    "provided context, which may include local documents and/or live web "
    "search results. Prefer local documents for anything specific to the "
    "user's own material; use web results for general or current-events "
    "questions. If neither source answers the question, say you don't know. "
    "Keep answers concise. Mention which source(s) you used."
)


def check_ollama_running():
    try:
        requests.get("http://localhost:11434", timeout=2)
        return True
    except requests.exceptions.ConnectionError:
        return False


if not check_ollama_running():
    raise RuntimeError(
        "Can't reach Ollama at http://localhost:11434. Make sure Ollama is "
        "installed (https://ollama.com/download) and running, and that "
        f"you've pulled the model with: ollama pull {MODEL_NAME}"
    )

if not local_index_exists():
    print("Note: no local document index found. Run `python rag/ingest.py` "
          "after adding documents to data/documents/ to enable local "
          "retrieval. Continuing with web search only for now.")

_executor = ThreadPoolExecutor(max_workers=2)


def gather_context(question: str):
    """Runs local retrieval and web search IN PARALLEL instead of one after
    the other -- this is the single biggest speed win since both are
    network/disk-bound and don't need to happen sequentially."""
    local_future = _executor.submit(retrieve_local_context, question)
    web_future = _executor.submit(search_web, question)

    local_context, local_sources = local_future.result()
    try:
        web_context, web_sources = web_future.result()
    except RuntimeError:
        web_context, web_sources = "", []  # Tavily not configured -- degrade gracefully

    return local_context, local_sources, web_context, web_sources


def generate_answer(question: str, history):
    local_context, local_sources, web_context, web_sources = gather_context(question)

    combined_context_parts = []
    if local_context:
        combined_context_parts.append("=== From your local documents ===\n" + local_context)
    if web_context:
        combined_context_parts.append("=== From live web search ===\n" + web_context)
    combined_context = "\n\n".join(combined_context_parts)

    if not combined_context:
        yield "I couldn't find anything relevant in your local documents or on the web. Try rephrasing the question."
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion: {question}"},
    ]

    # Stream the response so the person sees tokens appear immediately
    # instead of waiting for the whole answer -- feels much faster even
    # though total generation time is the same.
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": True,
            "keep_alive": KEEP_ALIVE,
            "options": {
                "num_predict": MAX_RESPONSE_TOKENS,
                "num_ctx": CONTEXT_WINDOW,
            },
        },
        stream=True,
        timeout=120,
    )
    response.raise_for_status()

    partial = ""
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        token = chunk.get("message", {}).get("content", "")
        partial += token
        yield partial
        if chunk.get("done"):
            break

    source_lines = []
    for s in local_sources:
        source_lines.append(f"- 📄 {s['title']} (local document)")
    for s in web_sources:
        source_lines.append(f"- 🌐 [{s['title']}]({s['url']})")
    sources_block = "\n".join(source_lines) if source_lines else "None"

    yield partial + f"\n\n---\n**Sources:**\n{sources_block}"


demo = gr.ChatInterface(
    fn=generate_answer,
    title="Hybrid Chatbot (Local Docs + Live Web Search)",
    description=f"Model: {MODEL_NAME} (via Ollama, CPU, streaming) — combines local documents with live Tavily web search",
)

if __name__ == "__main__":
    demo.launch()
