# 🎓 StudyMate — Hybrid AI Chatbot (Local Docs + Live Web Search)

> A personal AI study companion that combines your own notes with live web search — built to survive on a laptop with **no dedicated GPU**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Runs%20on-Ollama%20(CPU)-black.svg)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Status](https://img.shields.io/badge/Status-Active%20Development-yellow.svg)]()

---

## 🚀 The Pitch

Most "build your own LLM chatbot" tutorials assume you have a beefy GPU. **StudyMate doesn't.**

It runs entirely on **CPU** using [Ollama](https://ollama.com), and answers questions by combining two sources of truth **at the same time**:

| Source | What it's for |
|---|---|
| 📄 **Your local notes/documents** | Course material, syllabus, lecture notes — answers grounded in *your* content |
| 🌐 **Live web search** (via Tavily) | Anything current, general, or outside your notes |

The result: ask it anything, and it tells you *where the answer came from* — your notes or the live web — so you always know how much to trust it.

```
                    Your Question
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
  📄 Local Document Search      🌐 Live Web Search
   (Chroma + your notes)          (Tavily API)
            │                           │
            └─────────────┬─────────────┘
                          ▼
              🧠 Local LLM (Qwen2.5, via Ollama)
                    runs on CPU, no GPU needed
                          │
                          ▼
              ✅ Answer + cited sources
```

---

## 🎯 Why I'm Building This

I'm currently a **BSc CSIT student (2nd Semester)**, and I built this project for two reasons:

1. **As a study tool** — I wanted a chatbot that could answer questions using *my own* class notes, syllabus, and reference material first, and fall back to a live web search for anything my notes don't cover (recent updates, broader explanations, examples my textbook doesn't have).
2. **As a learning project** — this is my hands-on introduction to practical LLM engineering: retrieval-augmented generation (RAG), local model inference, fine-tuning workflows, and building something usable end-to-end — all without needing an expensive GPU setup.

This isn't a polished commercial product. It's a **learning-by-building project**, evolving as I learn more, and openly documenting the process (including the hardware constraints, dead ends, and fixes) along the way.

---

## ✨ Features

- 🧠 **Hybrid retrieval** — searches your local documents *and* the live web in parallel for every question
- 💻 **Runs on CPU** — no NVIDIA GPU required, works fine on integrated graphics
- 📚 **Bring your own notes** — drop `.txt`, `.md`, or `.pdf` files into `data/documents/` and it indexes them automatically
- 🌐 **Always current** — live web search fills in anything your notes don't cover
- 🔗 **Source-cited answers** — every response shows whether it came from your notes (📄) or the web (🌐)
- ⚡ **Streaming responses** — answers appear token-by-token instead of making you wait
- 🎛️ **Fine-tunable** — optional persona/tone fine-tuning via free Google Colab GPU + QLoRA, no local GPU needed

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| LLM inference | [Ollama](https://ollama.com) running `qwen2.5:1.5b` |
| Local retrieval | [LangChain](https://www.langchain.com/) + [Chroma](https://www.trychroma.com/) + `bge-small-en-v1.5` embeddings |
| Web search | [Tavily API](https://tavily.com) |
| UI | [Gradio](https://www.gradio.app/) |
| Fine-tuning (optional) | [Unsloth](https://unsloth.ai/) + QLoRA on free Google Colab GPU |

---

## 📦 Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/download) installed
- A free [Tavily](https://tavily.com) API key

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/studymate.git
cd studymate

# 2. Pull the local model
ollama pull qwen2.5:1.5b

# 3. Set up Python environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt

# 4. Add your Tavily API key
cp .env.example .env
# then edit .env and paste your key

# 5. (Optional) Add your study notes
# Drop .txt / .md / .pdf files into data/documents/
python rag/ingest.py

# 6. Launch
python app/app.py
```

Open the local URL Gradio prints (usually `http://127.0.0.1:7860`) and start asking questions.

---

## 📁 Project Structure

```
studymate/
├── data/
│   ├── documents/                 # your study notes go here
│   ├── prepare_finetune_data.py   # formats fine-tuning examples
│   └── finetune_examples.jsonl    # your persona/tone training examples
├── rag/
│   ├── ingest.py                  # builds the local search index
│   └── retriever.py               # local document retrieval logic
├── search/
│   └── web_search.py              # live web search (Tavily)
├── finetune/
│   └── README_COLAB.md            # fine-tuning guide (free Colab GPU)
├── app/
│   └── app.py                     # main chatbot app
├── .env.example
└── requirements.txt
```

---

## 🗺️ Roadmap

- [x] Local document RAG pipeline
- [x] Live web search integration
- [x] Hybrid (local + web) retrieval running in parallel
- [x] CPU-only inference via Ollama
- [x] Streaming responses
- [ ] Persona fine-tuning via Colab + QLoRA (in progress)
- [ ] Conversation memory across sessions
- [ ] Better source-priority logic (notes vs. web)
- [ ] Deploy a shareable version

---

## 🤝 Contributing / Feedback

This is a personal learning project, but suggestions, issues, and PRs are welcome — especially if you're also studying CS/IT and want to adapt this for your own coursework.

---

## 📄 License

MIT — free to use, modify, and learn from.

---

*Built by Nikesh Budhathoki, one debugging session at a time. 🐛→✅*