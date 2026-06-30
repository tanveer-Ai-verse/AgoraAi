# 🧠 AgoraAi — AI Knowledge & Debate Engine

Transform any document into an interactive thinking simulator. Upload a PDF, DOCX, TXT, or MD file and AgoraAi builds a knowledge graph, runs AI-powered debates, simulates real-world scenarios, generates exams, and assists academic research — all powered by **Groq LLaMA-3.3-70B-Versatile**.

---

## 🚀 Features

- **📤 Upload** — Extract text from PDF / DOCX / TXT / MD, or paste content directly
- **🗺️ Knowledge Graph** — Auto-extracted concepts rendered as an interactive Plotly network
- **🎭 Debate Arena** — Single-persona, multi-agent panel, Socratic, and Devil's Advocate debate modes
- **🌍 Scenario Simulator** — 10 scenario types (Business Crisis, Medical Diagnosis, AI Ethics, etc.) with AI-graded decisions
- **🧪 Thinking Lab** — 11 reasoning modes including First Principles, Bias Detector, Einstein Mode, Elon Musk Reasoning
- **🎓 Exam Mode** — Generates and grades exams across 7 question styles and 5 difficulty levels
- **🔬 Research Tools** — 10 academic tools: gap finder, hypothesis generator, peer reviewer, and more
- **📊 Analytics** — XP growth, skill radar, and concept mastery visualizations
- **🏆 Progress** — XP, levels, streaks, and badge gamification system
- **6 AI Personas** (Scientist, Engineer, Critic, Investor, Philosopher, Teacher) across **5 languages**

A lightweight FAISS-based retrieval system (RAG) grounds every AI response in the uploaded document's content.

---

## 🏗️ Project Structure

```
agoraai/
├── app.py             ← Main Streamlit application
├── requirements.txt   ← Python dependencies (PyPI only)
└── README.md          ← This file
```

---

## ⚙️ Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/agoraai.git
cd agoraai
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` in your project root:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

Get a free key at [console.groq.com](https://console.groq.com). Then run:

```bash
streamlit run app.py
```

> ⚠️ Add `.streamlit/secrets.toml` to your `.gitignore` — never commit API keys to GitHub.

---

## ☁️ Deploying to Streamlit Cloud

1. Push `app.py`, `requirements.txt`, and `README.md` to a GitHub repository (do **not** include `secrets.toml`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo → set main file to `app.py` → **Deploy**
3. **Set your API key:** open your app's **⋮ menu → Settings → Secrets**, paste `GROQ_API_KEY = "gsk_your_key_here"`, then click **Save** — the app restarts automatically with AI features enabled.

*(Optional)* Add `GEMINI_API_KEY = "your_gemini_key"` the same way to enable the optional Gemini fallback path — Groq alone is sufficient for full functionality.

---

## 🛡️ Deployment Notes

- No hardcoded API keys anywhere in the source — both Groq and Gemini clients are loaded only via `st.secrets`, with safe fallbacks if a key is missing.
- No spaCy model or runtime model downloads are used in this project, so the `OSError: Permission denied` issue some Streamlit Cloud apps hit with `spacy.cli.download()` does not apply here.
- `requirements.txt` contains only standard PyPI package names — no direct binary/wheel URLs.
- Every AI call, file upload, and visualization is wrapped in `try/except` so a single failure (bad file, API hiccup, malformed AI response) shows a friendly message instead of crashing the app.

---

## 🧪 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| AI Backend | Groq · LLaMA-3.3-70B-Versatile (+ optional Gemini 1.5 Flash fallback) |
| Retrieval | FAISS (flat L2 index) |
| Graph | NetworkX |
| Visualization | Plotly |
| File Parsing | PyPDF2 · python-docx |

---

<div align="center"><strong>🧠 AgoraAi · AI Knowledge & Debate Engine</strong></div>
