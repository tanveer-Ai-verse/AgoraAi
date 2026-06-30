"""
🧠 AgoraAi — AI Knowledge & Debate Engine
=============================================
Transform any document into an interactive thinking simulator.
Backend : Groq LLaMA-3.3-70B-Versatile (+ optional Gemini fallback)
Stack   : Streamlit · FAISS · NetworkX · Plotly · PyPDF2 · python-docx
"""

import streamlit as st
import os
import json
import time
import random
import re
import math
import hashlib
from datetime import datetime, timedelta
import PyPDF2
import docx
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ── Optional dependencies (loaded defensively) ────────────────────────────────
try:
    from groq import Groq
    GROQ_LIB_OK = True
except ImportError:
    GROQ_LIB_OK = False

try:
    import google.generativeai as genai
    GEMINI_LIB_OK = True
except ImportError:
    GEMINI_LIB_OK = False

try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgoraAi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════════════
#  SECURE CLIENT INITIALIZATION (st.secrets only — never hardcoded)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_groq_client():
    """Initialize the Groq client from Streamlit secrets. Returns None on failure."""
    if not GROQ_LIB_OK:
        return None
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=api_key)
    except (KeyError, FileNotFoundError):
        return None
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_gemini_model():
    """Initialize the optional Gemini fallback model. Returns None if unavailable."""
    if not GEMINI_LIB_OK:
        return None
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except (KeyError, FileNotFoundError):
        return None
    except Exception:
        return None


groq_client = get_groq_client()
gemini_model = get_gemini_model()
GROQ_READY = groq_client is not None

if not GROQ_READY:
    st.error(
        "⚠️ **GROQ_API_KEY is not configured.** AgoraAi's AI features "
        "(debates, scenarios, exams, research tools) require a valid Groq API key.\n\n"
        "Add it under **Settings → Secrets** in Streamlit Cloud, or in a local "
        "`.streamlit/secrets.toml` file. See the README for instructions."
    )

# ── Custom CSS (Twilight Theme: #564A96 / #B75F67) ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');

:root {
    --primary: #564A96;
    --secondary: #B75F67;
    --bg-dark: #0f0d1a;
    --bg-card: #1a1628;
    --bg-card2: #211832;
    --text-main: #f0eaff;
    --text-muted: #ccc0f0;
    --accent: #c97de0;
    --grad: linear-gradient(135deg, #564A96 0%, #B75F67 100%);
}

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: var(--bg-dark) !important;
    color: var(--text-main) !important;
}

.stApp { background: linear-gradient(160deg, #0f0d1a 0%, #1a0e2e 50%, #1a1020 100%) !important; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #130f22 0%, #1c1230 100%) !important;
    border-right: 1px solid rgba(86,74,150,0.4) !important;
}

h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; }

.agoraai-header {
    background: linear-gradient(135deg, rgba(86,74,150,0.3) 0%, rgba(183,95,103,0.3) 100%);
    border: 1px solid rgba(86,74,150,0.6);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.agoraai-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 30% 50%, rgba(86,74,150,0.2) 0%, transparent 70%),
                radial-gradient(ellipse at 70% 50%, rgba(183,95,103,0.2) 0%, transparent 70%);
    pointer-events: none;
}

.title-glow {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(90deg, #a78bfa, #B75F67, #564A96, #c97de0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    text-shadow: none;
    letter-spacing: 4px;
}

.creator-tag {
    color: #d4c8f8;
    font-size: 0.85rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 6px;
}

.card {
    background: var(--bg-card);
    border: 1px solid rgba(86,74,150,0.35);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    transition: border-color 0.3s;
}

.card:hover { border-color: rgba(183,95,103,0.6); }

.stat-box {
    background: linear-gradient(135deg, rgba(86,74,150,0.25), rgba(183,95,103,0.25));
    border: 1px solid rgba(86,74,150,0.5);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}

.stat-number {
    font-family: 'Orbitron', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #B75F67);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.persona-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-right: 6px;
    margin-bottom: 4px;
}

.badge-scientist { background: rgba(86,74,150,0.5); border: 1px solid #564A96; color: #c4baff; }
.badge-engineer  { background: rgba(59,130,246,0.3); border: 1px solid #3b82f6; color: #93c5fd; }
.badge-critic    { background: rgba(183,95,103,0.4); border: 1px solid #B75F67; color: #fca5a5; }
.badge-investor  { background: rgba(234,179,8,0.3);  border: 1px solid #eab308; color: #fde68a; }
.badge-philosopher { background: rgba(168,85,247,0.3); border: 1px solid #a855f7; color: #d8b4fe; }
.badge-teacher   { background: rgba(16,185,129,0.3); border: 1px solid #10b981; color: #6ee7b7; }

.debate-bubble {
    background: rgba(86,74,150,0.15);
    border-left: 3px solid #564A96;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 10px 0;
}

.debate-bubble.user-bubble {
    background: rgba(183,95,103,0.15);
    border-left-color: #B75F67;
    margin-left: 20px;
}

.xp-bar-bg {
    background: rgba(86,74,150,0.2);
    border-radius: 10px;
    height: 8px;
    overflow: hidden;
}

.xp-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #564A96, #B75F67);
    border-radius: 10px;
    transition: width 0.5s ease;
}

.stButton > button {
    background: linear-gradient(135deg, #564A96, #B75F67) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover { opacity: 0.85 !important; }

.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div {
    background: var(--bg-card2) !important;
    border: 1px solid rgba(86,74,150,0.4) !important;
    color: var(--text-main) !important;
    border-radius: 8px !important;
}

.stTabs [data-baseweb="tab"] {
    color: #c4b8e8 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
}

.stTabs [aria-selected="true"] {
    color: white !important;
    border-bottom: 2px solid #B75F67 !important;
}

/* hide default streamlit menu */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────────────────────
defaults = {
    "doc_text": "",
    "doc_chunks": [],
    "concepts": [],
    "knowledge_graph": None,
    "debate_history": [],
    "scenario_history": [],
    "xp": 0,
    "level": 1,
    "streak": 0,
    "badges": [],
    "sessions": 0,
    "mastery": {},
    "thinking_mode": "Scientific",
    "active_persona": "Scientist",
    "user_name": "Learner",
    "language": "English",
    "faiss_index": None,
    "chunk_embeddings": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ════════════════════════════════════════════════════════════════════════════
#  RAG HELPERS — embeddings, FAISS index, retrieval
# ════════════════════════════════════════════════════════════════════════════
def simple_embed(text, dim=128):
    """Deterministic pseudo-embedding using character statistics (no heavy model needed)."""
    vec = np.zeros(dim, dtype=np.float32)
    for i, ch in enumerate(text[:512]):
        vec[i % dim] += ord(ch) / 128.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def build_faiss_index(chunks):
    """Build a FAISS flat-L2 index over the document chunks. Returns (None, []) on failure."""
    if not FAISS_OK:
        return None, []
    try:
        dim = 128
        index = faiss.IndexFlatL2(dim)
        embeddings = [simple_embed(chunk) for chunk in chunks]
        if embeddings:
            matrix = np.array(embeddings, dtype=np.float32)
            index.add(matrix)
        return index, embeddings
    except Exception as e:
        st.warning(f"⚠️ Could not build search index: {e}. Falling back to first chunks.")
        return None, []


def retrieve_chunks(query, index, chunks, k=3):
    """Retrieve the most relevant chunks for a query. Falls back gracefully if index is unavailable."""
    if index is None or len(chunks) == 0:
        return chunks[:3]
    try:
        q_emb = simple_embed(query).reshape(1, -1)
        k = min(k, len(chunks))
        distances, indices = index.search(q_emb, k)
        return [chunks[i] for i in indices[0] if i < len(chunks)]
    except Exception:
        return chunks[:3]


def chunk_text(text, size=400, overlap=80):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


def extract_text(uploaded_file):
    """Extract text from an uploaded PDF, DOCX, TXT, or MD file with robust error handling."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                try:
                    pg = page.extract_text()
                    if pg:
                        text += pg + "\n"
                except Exception:
                    continue
            return text
        elif name.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])
        elif name.endswith(".txt") or name.endswith(".md"):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        else:
            return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"⚠️ Could not read '{uploaded_file.name}': {e}")
        return ""


# ════════════════════════════════════════════════════════════════════════════
#  AI BACKEND — Groq primary, optional Gemini fallback
# ════════════════════════════════════════════════════════════════════════════
def call_groq(prompt, system="You are AgoraAi, an expert AI learning assistant.", max_tokens=1024):
    """Call the Groq LLaMA-3.3-70B-Versatile model with robust error handling."""
    if groq_client is None:
        return (
            "⚠️ **AI unavailable.** GROQ_API_KEY is not configured. "
            "Please add it under Streamlit Cloud → Settings → Secrets."
        )
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq API error: {e}. Please try again in a moment."


def call_gemini(prompt):
    """Optional Gemini fallback for rich/complex tasks. Degrades gracefully if unavailable."""
    if gemini_model is None:
        return "⚠️ Gemini fallback is not configured (optional feature). Using Groq instead is recommended."
    try:
        result = gemini_model.generate_content(prompt)
        return result.text
    except Exception as e:
        return f"⚠️ Gemini API error: {e}"


def smart_call(prompt, system="You are AgoraAi.", use_gemini=False, max_tokens=1024):
    """Dispatch to Gemini if explicitly requested and available, otherwise use Groq (default)."""
    if use_gemini and gemini_model is not None:
        return call_gemini(prompt)
    return call_groq(prompt, system, max_tokens)


def extract_concepts(text):
    """Extract 12-18 key concepts from text via Groq, with safe JSON parsing fallback."""
    snippet = text[:3000]
    prompt = (
        "Extract exactly 12-18 key concepts from the following text.\n"
        "Return ONLY a JSON array of strings. No explanation.\n"
        'Example: ["Machine Learning", "Neural Networks"]\n\n'
        + snippet
    )
    raw = call_groq(prompt)
    try:
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    try:
        cleaned = re.sub(r'[\[\]"]', "", raw)
        return [c.strip() for c in cleaned.split(",") if len(c.strip()) > 2][:15]
    except Exception:
        return []


# ════════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE GRAPH
# ════════════════════════════════════════════════════════════════════════════
def build_knowledge_graph(concepts):
    """Build a co-occurrence-style graph linking related concepts."""
    G = nx.Graph()
    try:
        for c in concepts:
            G.add_node(c)
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                w1 = set(concepts[i].lower().split())
                w2 = set(concepts[j].lower().split())
                shared = w1 & w2
                if shared or (i < j and (j - i) <= 3):
                    weight = len(shared) + 1 if shared else 1
                    G.add_edge(concepts[i], concepts[j], weight=weight)
    except Exception as e:
        st.warning(f"⚠️ Knowledge graph build issue: {e}")
    return G


def render_knowledge_graph(G):
    """Render the knowledge graph as an interactive Plotly network."""
    if G is None or len(G.nodes) == 0:
        st.info("No concepts extracted yet.")
        return
    try:
        pos = nx.spring_layout(G, seed=42, k=2.0)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color="rgba(86,74,150,0.5)"),
            hoverinfo="none",
            mode="lines"
        )

        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_labels = list(G.nodes())
        node_sizes = [20 + G.degree(n) * 8 for n in G.nodes()]
        node_colors = [G.degree(n) for n in G.nodes()]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=node_labels,
            textposition="top center",
            hoverinfo="text",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale=[[0, "#564A96"], [0.5, "#8b5cf6"], [1, "#B75F67"]],
                line=dict(width=2, color="white"),
                showscale=False
            ),
            textfont=dict(color="white", size=10)
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(text="Knowledge Graph", font=dict(color="white", size=16)),
                paper_bgcolor="rgba(15,13,26,0)",
                plot_bgcolor="rgba(26,22,40,0.6)",
                font=dict(color="white"),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500,
                margin=dict(l=20, r=20, t=50, b=20),
                showlegend=False
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Could not render knowledge graph: {e}")


# ════════════════════════════════════════════════════════════════════════════
#  XP / GAMIFICATION
# ════════════════════════════════════════════════════════════════════════════
def add_xp(points, reason=""):
    st.session_state.xp += points
    level = 1 + st.session_state.xp // 200
    if level > st.session_state.level:
        st.session_state.level = level
        st.balloons()
    if reason:
        st.toast(f"+{points} XP — {reason}", icon="⚡")


def render_xp_bar():
    xp = st.session_state.xp
    level = st.session_state.level
    xp_in_level = xp % 200
    pct = int((xp_in_level / 200) * 100)
    st.markdown(
        f"""
        <div class="stat-box">
          <div class="stat-number">LVL {level}</div>
          <div style="color:#ccc0f0;font-size:0.8rem;margin:4px 0;">Total XP: {xp}</div>
          <div class="xp-bar-bg"><div class="xp-bar-fill" style="width:{pct}%"></div></div>
          <div style="color:#ccc0f0;font-size:0.75rem;margin-top:4px;">{xp_in_level}/200 to next level</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════════════════════
#  PERSONAS / MODES / LANGUAGES
# ════════════════════════════════════════════════════════════════════════════
PERSONAS = {
    "Scientist":    {"icon": "🔬", "badge": "badge-scientist", "desc": "Evidence-based reasoning"},
    "Engineer":     {"icon": "⚙️", "badge": "badge-engineer",  "desc": "Practical scalability"},
    "Critic":       {"icon": "🎭", "badge": "badge-critic",    "desc": "Finds weaknesses"},
    "Investor":     {"icon": "💰", "badge": "badge-investor",  "desc": "ROI & market thinking"},
    "Philosopher":  {"icon": "📜", "badge": "badge-philosopher", "desc": "Fundamental questions"},
    "Teacher":      {"icon": "🎓", "badge": "badge-teacher",   "desc": "Socratic guidance"},
}

THINKING_MODES = [
    "Scientific", "First Principles", "Systems Thinking",
    "Strategic", "Design Thinking", "Startup Mode",
    "Risk Analysis", "Ethical Dilemma", "Historical Simulation",
    "Mathematical", "Socratic", "Devil's Advocate"
]

LANGUAGES = ["English", "Urdu", "Hindi", "Arabic", "Pashto"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """<div style="text-align:center;padding:16px 0;">
        <span style="font-family:'Orbitron',sans-serif;font-size:1.4rem;
        background:linear-gradient(90deg,#a78bfa,#B75F67);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;font-weight:900;">🧠 AgoraAi</span>
        <div style="color:#ccc0f0;font-size:0.7rem;letter-spacing:2px;
        text-transform:uppercase;margin-top:4px;">AI Knowledge & Debate Engine</div>
        </div>""",
        unsafe_allow_html=True
    )
    st.divider()
    render_xp_bar()
    st.markdown(f"🔥 Streak: **{st.session_state.streak} days**")
    st.divider()

    st.subheader("⚙️ Settings")
    st.session_state.user_name = st.text_input("Your Name", value=st.session_state.user_name)
    st.session_state.active_persona = st.selectbox(
        "AI Persona",
        list(PERSONAS.keys()),
        index=list(PERSONAS.keys()).index(st.session_state.active_persona)
    )
    st.session_state.thinking_mode = st.selectbox(
        "Thinking Mode",
        THINKING_MODES,
        index=THINKING_MODES.index(st.session_state.thinking_mode)
    )
    st.session_state.language = st.selectbox(
        "Language",
        LANGUAGES,
        index=LANGUAGES.index(st.session_state.language)
    )
    st.divider()

    p = PERSONAS[st.session_state.active_persona]
    st.markdown(
        f"""<div class="card">
        <span class="persona-badge {p['badge']}">{p['icon']} {st.session_state.active_persona}</span>
        <div style="color:#ccc0f0;font-size:0.82rem;margin-top:8px;">{p['desc']}</div>
        </div>""",
        unsafe_allow_html=True
    )

    if st.session_state.badges:
        st.subheader("🏆 Badges")
        st.write(" ".join(st.session_state.badges))

    if not GROQ_READY:
        st.divider()
        st.caption("🔴 AI backend offline — configure GROQ_API_KEY in Secrets.")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """<div class="agoraai-header">
    <div class="title-glow">⚡ AGORAAI</div>
    <div class="creator-tag">AI Knowledge & Debate Engine</div>
    <div style="color:#ccc0f0;font-size:0.88rem;margin-top:10px;">
      Transform any document into an interactive thinking simulator
    </div>
    </div>""",
    unsafe_allow_html=True
)

# ── Stats row ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(
        f"""<div class="stat-box">
        <div class="stat-number">{len(st.session_state.doc_chunks)}</div>
        <div style="color:#ccc0f0;font-size:0.78rem;">Chunks</div>
        </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(
        f"""<div class="stat-box">
        <div class="stat-number">{len(st.session_state.concepts)}</div>
        <div style="color:#ccc0f0;font-size:0.78rem;">Concepts</div>
        </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(
        f"""<div class="stat-box">
        <div class="stat-number">{len(st.session_state.debate_history)}</div>
        <div style="color:#ccc0f0;font-size:0.78rem;">Debates</div>
        </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(
        f"""<div class="stat-box">
        <div class="stat-number">{st.session_state.xp}</div>
        <div style="color:#ccc0f0;font-size:0.78rem;">XP Earned</div>
        </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(
        f"""<div class="stat-box">
        <div class="stat-number">{st.session_state.level}</div>
        <div style="color:#ccc0f0;font-size:0.78rem;">Level</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📤 Upload",
    "🗺️ Knowledge Graph",
    "🎭 Debate Arena",
    "🌍 Scenario Sim",
    "🧪 Thinking Lab",
    "📊 Analytics",
    "🎓 Exam Mode",
    "🔬 Research",
    "🏆 Progress"
])

# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — UPLOAD
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### 📤 Upload Your Knowledge Source")
    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload PDF, DOCX, TXT, or MD",
            type=["pdf", "docx", "txt", "md"]
        )
        st.markdown("**Or paste text directly:**")
        pasted = st.text_area("Paste content here", height=180, placeholder="Paste lecture notes, articles, research papers...")

        if st.button("🚀 Process & Forge Knowledge", use_container_width=True):
            raw_text = ""
            try:
                if uploaded:
                    with st.spinner("Extracting text..."):
                        raw_text = extract_text(uploaded)
                if pasted.strip():
                    raw_text += "\n" + pasted.strip()

                if raw_text.strip():
                    with st.spinner("Chunking document..."):
                        chunks = chunk_text(raw_text)
                        st.session_state.doc_text = raw_text
                        st.session_state.doc_chunks = chunks

                    with st.spinner("Extracting concepts..."):
                        concepts = extract_concepts(raw_text)
                        st.session_state.concepts = concepts

                    with st.spinner("Building knowledge graph..."):
                        G = build_knowledge_graph(concepts)
                        st.session_state.knowledge_graph = G

                    with st.spinner("Building search index..."):
                        idx, embs = build_faiss_index(chunks)
                        st.session_state.faiss_index = idx
                        st.session_state.chunk_embeddings = embs

                    add_xp(50, "Document processed")
                    if "📚 First Upload" not in st.session_state.badges:
                        st.session_state.badges.append("📚 First Upload")
                    st.success(f"✅ Processed {len(chunks)} chunks, {len(concepts)} concepts extracted!")
                    st.rerun()
                else:
                    st.warning("Please upload a file or paste some text.")
            except Exception as e:
                st.error(f"⚠️ Something went wrong while processing your document: {e}")

    with col_info:
        st.markdown(
            """<div class="card">
            <b>Supported Sources</b>
            <ul style="color:#ccc0f0;font-size:0.85rem;margin-top:8px;">
            <li>📄 PDF papers</li>
            <li>📝 DOCX documents</li>
            <li>📋 Plain text / Markdown</li>
            <li>✂️ Pasted content</li>
            </ul>
            </div>
            <div class="card">
            <b>What AgoraAi Creates</b>
            <ul style="color:#ccc0f0;font-size:0.85rem;margin-top:8px;">
            <li>🗺️ Knowledge graph</li>
            <li>🎭 AI debate engine</li>
            <li>🌍 Real-world scenarios</li>
            <li>🎓 Exam questions</li>
            <li>📊 Thinking analytics</li>
            </ul>
            </div>""",
            unsafe_allow_html=True
        )

    if st.session_state.doc_text:
        with st.expander("📖 Document Preview (first 800 chars)"):
            st.text(st.session_state.doc_text[:800])

# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — KNOWLEDGE GRAPH
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### 🗺️ Interactive Knowledge Graph")
    if not st.session_state.concepts:
        st.info("Upload a document first to generate the knowledge graph.")
    else:
        render_knowledge_graph(st.session_state.knowledge_graph)

        st.markdown("#### 🔍 Concept Explorer")
        selected = st.selectbox("Select a concept to explore", st.session_state.concepts)
        if selected and st.button("🔬 Deep Dive"):
            try:
                relevant = retrieve_chunks(
                    selected,
                    st.session_state.faiss_index,
                    st.session_state.doc_chunks
                )
                context = " ".join(relevant)
                persona = st.session_state.active_persona
                p_info = PERSONAS[persona]
                lang = st.session_state.language
                prompt = (
                    f"You are a {persona} AI mentor. {p_info['desc']}. "
                    f"Explain the concept '{selected}' in {lang} language, "
                    "from your persona's perspective, using the context below. "
                    "Be insightful, challenge assumptions, ask one probing question at the end.\n\n"
                    f"Context: {context[:1500]}"
                )
                with st.spinner(f"{p_info['icon']} {persona} is analyzing..."):
                    result = smart_call(prompt)
                add_xp(20, f"Explored {selected}")
                st.markdown(
                    f"""<div class="debate-bubble">
                    <span class="persona-badge {p_info['badge']}">{p_info['icon']} {persona}</span><br><br>
                    {result}
                    </div>""",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"⚠️ Could not complete deep dive: {e}")

        st.markdown("#### 📋 All Concepts")
        cols = st.columns(4)
        for i, concept in enumerate(st.session_state.concepts):
            with cols[i % 4]:
                deg = st.session_state.knowledge_graph.degree(concept) if st.session_state.knowledge_graph else 0
                st.markdown(
                    f"""<div class="card" style="text-align:center;padding:12px;">
                    <div style="font-weight:700;font-size:0.9rem;">{concept}</div>
                    <div style="color:#ccc0f0;font-size:0.75rem;">connections: {deg}</div>
                    </div>""",
                    unsafe_allow_html=True
                )

# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — DEBATE ARENA
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### 🎭 AI Debate Arena")
    if not st.session_state.doc_text:
        st.info("Upload a document to start debating.")
    else:
        st.markdown("#### Choose Your Debate Mode")
        debate_mode = st.radio(
            "",
            ["Single Persona", "Multi-Agent Panel", "Socratic Dialogue", "Devil's Advocate"],
            horizontal=True
        )

        if debate_mode == "Multi-Agent Panel":
            st.markdown("##### 🎙️ Expert Panel — All personas will respond")
            panel_q = st.text_input("Your question for the panel:", placeholder="Ask anything about the document...")
            if st.button("🚀 Launch Panel Debate") and panel_q:
                try:
                    relevant = retrieve_chunks(
                        panel_q,
                        st.session_state.faiss_index,
                        st.session_state.doc_chunks
                    )
                    context = " ".join(relevant)[:1200]
                    lang = st.session_state.language
                    for persona_name, p_info in PERSONAS.items():
                        with st.spinner(f"{p_info['icon']} {persona_name} thinking..."):
                            prompt = (
                                f"You are the {persona_name} on an expert panel. {p_info['desc']}. "
                                f"Answer in {lang}: '{panel_q}' "
                                f"Context: {context}. "
                                "Be concise (3-5 sentences) and true to your persona. "
                                "End with a challenging follow-up question."
                            )
                            reply = smart_call(prompt)
                        st.markdown(
                            f"""<div class="debate-bubble">
                            <span class="persona-badge {p_info['badge']}">{p_info['icon']} {persona_name}</span><br><br>
                            {reply}
                            </div>""",
                            unsafe_allow_html=True
                        )
                    add_xp(60, "Multi-Agent Panel")
                    if "🎭 Panel Debater" not in st.session_state.badges:
                        st.session_state.badges.append("🎭 Panel Debater")
                except Exception as e:
                    st.error(f"⚠️ Panel debate failed: {e}")

        else:
            st.markdown("#### 💬 Interactive Debate")
            for entry in st.session_state.debate_history[-8:]:
                role = entry.get("role", "ai")
                css_class = "debate-bubble user-bubble" if role == "user" else "debate-bubble"
                persona_label = entry.get("persona", "AI")
                p_badge = ""
                if persona_label in PERSONAS:
                    pi = PERSONAS[persona_label]
                    p_badge = f"""<span class="persona-badge {pi['badge']}">{pi['icon']} {persona_label}</span><br><br>"""
                elif role == "user":
                    p_badge = f"""<span class="persona-badge badge-teacher">👤 {st.session_state.user_name}</span><br><br>"""
                st.markdown(
                    f"""<div class="{css_class}">{p_badge}{entry['content']}</div>""",
                    unsafe_allow_html=True
                )

            user_msg = st.text_input("Your response:", key="debate_input", placeholder="Type your answer or question...")

            col_send, col_new = st.columns([3, 1])
            with col_send:
                send_clicked = st.button("📤 Send", use_container_width=True)
            with col_new:
                new_clicked = st.button("🆕 New Topic", use_container_width=True)

            if new_clicked:
                try:
                    snippet = st.session_state.doc_text[:2000]
                    persona = st.session_state.active_persona
                    p_info = PERSONAS[persona]
                    mode_info = st.session_state.thinking_mode
                    lang = st.session_state.language
                    prompt = (
                        f"You are a {persona} AI mentor using {mode_info} thinking. "
                        f"Respond in {lang}. Start a challenging debate about this content. "
                        "Ask a powerful opening question that forces critical thinking. "
                        "Do not summarize — challenge the student.\n\n"
                        f"Content: {snippet}"
                    )
                    with st.spinner("Opening debate..."):
                        ai_msg = smart_call(prompt)
                    st.session_state.debate_history.append({"role": "ai", "persona": persona, "content": ai_msg})
                    add_xp(10, "New debate started")
                    st.rerun()
                except Exception as e:
                    st.error(f"⚠️ Could not start a new debate topic: {e}")

            if send_clicked and user_msg.strip():
                try:
                    st.session_state.debate_history.append({"role": "user", "persona": "User", "content": user_msg})
                    history_text = ""
                    for h in st.session_state.debate_history[-6:]:
                        r = h.get("role", "ai")
                        prefix = "Student" if r == "user" else h.get("persona", "AI")
                        history_text += prefix + ": " + h["content"] + "\n"
                    relevant = retrieve_chunks(
                        user_msg,
                        st.session_state.faiss_index,
                        st.session_state.doc_chunks
                    )
                    context = " ".join(relevant)[:1000]
                    persona = st.session_state.active_persona
                    p_info = PERSONAS[persona]
                    mode_info = st.session_state.thinking_mode
                    lang = st.session_state.language
                    if debate_mode == "Socratic Dialogue":
                        style = "Ask only questions, never give answers directly. Use Socratic method."
                    elif debate_mode == "Devil's Advocate":
                        style = "Always argue the opposite of what the student says. Find flaws."
                    else:
                        style = f"Use {mode_info} thinking. Challenge and extend the student's ideas."
                    prompt = (
                        f"You are a {persona} AI mentor. {style} "
                        f"Respond in {lang}. Keep response to 4-6 sentences. "
                        "Debate history:\n" + history_text + "\n"
                        f"Relevant context: {context}"
                    )
                    with st.spinner(f"{p_info['icon']} {persona} responding..."):
                        ai_reply = smart_call(prompt)
                    st.session_state.debate_history.append({"role": "ai", "persona": persona, "content": ai_reply})
                    add_xp(15, "Debate round")
                    st.rerun()
                except Exception as e:
                    st.error(f"⚠️ Could not send your message: {e}")

# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 — SCENARIO SIMULATOR
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### 🌍 Real-World Scenario Simulator")
    if not st.session_state.doc_text:
        st.info("Upload a document to generate scenarios.")
    else:
        scenario_types = [
            "Business Crisis", "Medical Diagnosis", "Engineering Problem",
            "Startup Founder", "Government Policy", "Crisis Management",
            "Courtroom Argument", "Research Lab", "Space Mission", "AI Ethics"
        ]
        s_type = st.selectbox("Scenario Type", scenario_types)
        lang = st.session_state.language

        if st.button("🎲 Generate New Scenario", use_container_width=True):
            try:
                snippet = st.session_state.doc_text[:2000]
                persona = st.session_state.active_persona
                p_info = PERSONAS[persona]
                prompt = (
                    f"Create a rich, challenging {s_type} scenario in {lang} "
                    f"based on this content. Persona: {persona}. "
                    "Format:\n"
                    "**SITUATION:** (2-3 sentences describing the crisis)\n"
                    "**YOUR ROLE:** (who the student plays)\n"
                    "**THE CHALLENGE:** (what decision must be made)\n"
                    "**KEY CONSTRAINTS:** (3 bullet points)\n"
                    "**QUESTION:** (one powerful question to the student)\n\n"
                    f"Content: {snippet}"
                )
                with st.spinner("Generating scenario..."):
                    scenario = smart_call(prompt)
                st.session_state.scenario_history.append({"type": s_type, "scenario": scenario, "response": ""})
                add_xp(20, "Scenario generated")
                st.rerun()
            except Exception as e:
                st.error(f"⚠️ Could not generate scenario: {e}")

        if st.session_state.scenario_history:
            current = st.session_state.scenario_history[-1]
            st.markdown(
                f"""<div class="card">
                <span class="persona-badge badge-scientist">🌍 {current['type']}</span><br><br>
                {current['scenario'].replace(chr(10), '<br>')}
                </div>""",
                unsafe_allow_html=True
            )

            user_decision = st.text_area("Your Decision & Reasoning:", height=120, placeholder="Explain your decision...")
            if st.button("📤 Submit Decision") and user_decision.strip():
                try:
                    persona = st.session_state.active_persona
                    p_info = PERSONAS[persona]
                    prompt = (
                        f"As a {persona} expert, evaluate this decision in {lang}: "
                        f"'{user_decision}'. "
                        f"Scenario: {current['scenario'][:600]}. "
                        "Give: 1) What is good about this decision 2) What is wrong or missing "
                        "3) What an expert would actually do 4) Give a score out of 100."
                    )
                    with st.spinner("Evaluating your decision..."):
                        feedback = smart_call(prompt)
                    st.session_state.scenario_history[-1]["response"] = feedback
                    add_xp(30, "Scenario decision")
                    if "🌍 Scenario Master" not in st.session_state.badges:
                        st.session_state.badges.append("🌍 Scenario Master")
                    st.markdown(
                        f"""<div class="debate-bubble">
                        <span class="persona-badge {p_info['badge']}">{p_info['icon']} {persona} Feedback</span><br><br>
                        {feedback}
                        </div>""",
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"⚠️ Could not evaluate your decision: {e}")

# ════════════════════════════════════════════════════════════════════════════
#  TAB 5 — THINKING LAB
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 🧪 Thinking Laboratory")
    if not st.session_state.doc_text:
        st.info("Upload a document to activate the Thinking Lab.")
    else:
        lab_mode = st.selectbox(
            "Select Lab Mode",
            [
                "First Principles Breakdown",
                "Assumption Analyzer",
                "Contradiction Finder",
                "Bias Detector",
                "Counterargument Generator",
                "Decision Tree Builder",
                "Hypothesis Testing",
                "Systems Thinking Map",
                "Einstein Mode",
                "Elon Musk Reasoning",
                "Socrates Questioning"
            ]
        )

        user_input = st.text_area(
            "Topic or claim to analyze:",
            height=100,
            placeholder="Enter a concept, idea, or claim from your document..."
        )
        lang = st.session_state.language

        if st.button("⚡ Activate Lab", use_container_width=True) and user_input.strip():
            try:
                relevant = retrieve_chunks(
                    user_input,
                    st.session_state.faiss_index,
                    st.session_state.doc_chunks
                )
                context = " ".join(relevant)[:1200]

                mode_prompts = {
                    "First Principles Breakdown": (
                        "Break this down to absolute first principles. "
                        "What are the fundamental truths? What can we know for certain? "
                        "Strip away assumptions one by one."
                    ),
                    "Assumption Analyzer": (
                        "List every hidden assumption in this idea. "
                        "Rate each assumption 1-10 for validity. "
                        "Which assumptions, if wrong, would collapse the entire idea?"
                    ),
                    "Contradiction Finder": (
                        "Find every internal contradiction, paradox, and logical inconsistency. "
                        "Be relentless. What cannot be true at the same time?"
                    ),
                    "Bias Detector": (
                        "Identify cognitive biases, cultural biases, and selection biases present. "
                        "What perspectives are missing? Who benefits from this framing?"
                    ),
                    "Counterargument Generator": (
                        "Generate the 5 strongest counterarguments to this idea. "
                        "Make each counterargument as powerful as possible. Do not hold back."
                    ),
                    "Decision Tree Builder": (
                        "Build a decision tree for this topic. "
                        "Show: If X then Y, If not X then Z. "
                        "Cover all major decision branches and their consequences."
                    ),
                    "Hypothesis Testing": (
                        "Formulate 3 testable hypotheses. "
                        "For each: state the hypothesis, how to test it, what would falsify it, "
                        "and what experiment would prove it."
                    ),
                    "Systems Thinking Map": (
                        "Map all system components, feedback loops, delays, and emergent behaviors. "
                        "Identify leverage points. What small change has the biggest impact?"
                    ),
                    "Einstein Mode": (
                        "Think like Einstein. Use thought experiments. "
                        "Imagine extreme edge cases. What happens at the limits? "
                        "What does this look like from the perspective of light/time/space?"
                    ),
                    "Elon Musk Reasoning": (
                        "Apply Elon Musk thinking: first principles, 10x thinking, physics constraints. "
                        "How would you build this from scratch? What is the minimum viable version? "
                        "How do you make it 100x better?"
                    ),
                    "Socrates Questioning": (
                        "Use the Socratic method. Generate 8 increasingly deep questions "
                        "that would make a student realize what they do not know. "
                        "Never give answers, only questions."
                    ),
                }

                mode_instruction = mode_prompts.get(lab_mode, "Analyze this deeply.")
                prompt = (
                    f"Respond in {lang}. Mode: {lab_mode}.\n"
                    f"Instruction: {mode_instruction}\n"
                    f"Topic: {user_input}\n"
                    f"Context from document: {context}"
                )
                with st.spinner(f"Running {lab_mode}..."):
                    result = smart_call(prompt, max_tokens=1200)
                add_xp(25, lab_mode)
                if "🧪 Lab Scientist" not in st.session_state.badges:
                    st.session_state.badges.append("🧪 Lab Scientist")
                st.markdown(
                    f"""<div class="debate-bubble">
                    <span class="persona-badge badge-scientist">🧪 {lab_mode}</span><br><br>
                    {result}
                    </div>""",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"⚠️ Thinking Lab encountered an error: {e}")

# ════════════════════════════════════════════════════════════════════════════
#  TAB 6 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### 📊 Cognitive Analytics Dashboard")

    col_a, col_b = st.columns(2)
    with col_a:
        try:
            xp_data = list(range(0, st.session_state.xp + 10, max(1, st.session_state.xp // 10 + 1)))
            fig_xp = go.Figure()
            fig_xp.add_trace(go.Scatter(
                x=list(range(len(xp_data))),
                y=xp_data,
                fill="tozeroy",
                fillcolor="rgba(86,74,150,0.25)",
                line=dict(color="#B75F67", width=2),
                name="XP Growth"
            ))
            fig_xp.update_layout(
                title=dict(text="XP Growth", font=dict(color="white")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,22,40,0.6)",
                font=dict(color="white"),
                xaxis=dict(gridcolor="rgba(86,74,150,0.2)", showgrid=True, zeroline=False),
                yaxis=dict(gridcolor="rgba(86,74,150,0.2)", showgrid=True, zeroline=False),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_xp, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Could not render XP chart: {e}")

    with col_b:
        try:
            categories = ["Debates", "Scenarios", "Lab Modes", "Concepts", "XP"]
            vals = [
                min(len(st.session_state.debate_history), 10),
                min(len(st.session_state.scenario_history), 10),
                min(st.session_state.xp // 25, 10),
                min(len(st.session_state.concepts), 10),
                min(st.session_state.level * 2, 10)
            ]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor="rgba(183,95,103,0.2)",
                line=dict(color="#B75F67"),
                name="Activity"
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(26,22,40,0.6)",
                    radialaxis=dict(visible=True, range=[0, 10], gridcolor="rgba(86,74,150,0.3)", color="white"),
                    angularaxis=dict(gridcolor="rgba(86,74,150,0.3)", color="white")
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                title=dict(text="Skill Radar", font=dict(color="white")),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Could not render skill radar: {e}")

    if st.session_state.concepts:
        try:
            st.markdown("#### 🎯 Concept Mastery Heatmap")
            concepts_list = st.session_state.concepts[:12]
            mastery_vals = [random.randint(20, 95) for _ in concepts_list]
            fig_heat = go.Figure(go.Bar(
                x=concepts_list,
                y=mastery_vals,
                marker=dict(
                    color=mastery_vals,
                    colorscale=[[0, "#564A96"], [0.5, "#8b5cf6"], [1, "#B75F67"]],
                    showscale=True,
                    colorbar=dict(tickfont=dict(color="white"), title=dict(text="Mastery %", font=dict(color="white")))
                )
            ))
            fig_heat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,22,40,0.6)",
                font=dict(color="white"),
                xaxis=dict(tickangle=-35, gridcolor="rgba(86,74,150,0.2)", zeroline=False),
                yaxis=dict(gridcolor="rgba(86,74,150,0.2)", zeroline=False,
                           title=dict(text="Mastery %", font=dict(color="white"))),
                height=350,
                margin=dict(l=40, r=40, t=20, b=80)
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Could not render mastery heatmap: {e}")

# ════════════════════════════════════════════════════════════════════════════
#  TAB 7 — EXAM MODE
# ════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### 🎓 Exam Mode")
    if not st.session_state.doc_text:
        st.info("Upload a document to generate exam questions.")
    else:
        exam_type = st.selectbox(
            "Exam Style",
            [
                "Multiple Choice", "Short Answer", "Essay Questions",
                "True/False with Explanation", "Case Study Analysis",
                "Interview Questions", "Research Defense"
            ]
        )
        difficulty = st.select_slider(
            "Difficulty",
            options=["Beginner", "Intermediate", "Advanced", "Expert", "PhD Level"]
        )
        num_q = st.slider("Number of Questions", 3, 10, 5)
        lang = st.session_state.language

        if st.button("📝 Generate Exam", use_container_width=True):
            try:
                snippet = st.session_state.doc_text[:2500]
                prompt = (
                    f"Create {num_q} {exam_type} questions at {difficulty} difficulty in {lang}. "
                    "Number each question. For multiple choice include A/B/C/D options. "
                    "Make questions that require deep understanding, not just memorization. "
                    f"Content: {snippet}"
                )
                with st.spinner("Generating exam..."):
                    exam_content = smart_call(prompt, max_tokens=1500)
                st.session_state["agoraai_exam_content"] = exam_content
                add_xp(30, "Exam generated")
            except Exception as e:
                st.error(f"⚠️ Could not generate exam: {e}")

        exam_content = st.session_state.get("agoraai_exam_content", "")
        if exam_content:
            st.markdown(
                f"""<div class="card">{exam_content}</div>""",
                unsafe_allow_html=True
            )

            answer_area = st.text_area("Your Answers:", height=200, key="exam_answers", placeholder="Write your answers here...")
            if st.button("📊 Grade My Answers") and answer_area.strip():
                try:
                    persona = st.session_state.active_persona
                    p_info = PERSONAS[persona]
                    grade_prompt = (
                        f"You are a strict {persona} examiner. "
                        f"Grade these answers in {lang}: '{answer_area}'. "
                        f"Questions: {exam_content[:800]}. "
                        "For each answer give: score out of 10, what is right, what is missing. "
                        "Give total score and overall feedback."
                    )
                    with st.spinner("Grading..."):
                        grade = smart_call(grade_prompt, max_tokens=1000)
                    add_xp(40, "Exam completed")
                    if "🎓 Exam Taker" not in st.session_state.badges:
                        st.session_state.badges.append("🎓 Exam Taker")
                    st.markdown(
                        f"""<div class="debate-bubble">
                        <span class="persona-badge {p_info['badge']}">{p_info['icon']} {persona} Grading</span><br><br>
                        {grade}
                        </div>""",
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"⚠️ Could not grade your answers: {e}")

# ════════════════════════════════════════════════════════════════════════════
#  TAB 8 — RESEARCH
# ════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("### 🔬 Research Intelligence")
    if not st.session_state.doc_text:
        st.info("Upload a document to activate research features.")
    else:
        research_mode = st.selectbox(
            "Research Tool",
            [
                "Research Gap Finder",
                "Hypothesis Generator",
                "Literature Review Draft",
                "Scientific Claim Verifier",
                "Paper Comparison",
                "Academic Writing Assistant",
                "Experiment Designer",
                "AI Peer Reviewer",
                "Citation Recommender",
                "Research Timeline Builder"
            ]
        )
        lang = st.session_state.language
        research_input = st.text_area("Your Research Question or Topic:", height=100)

        if st.button("🔭 Run Research Tool", use_container_width=True):
            try:
                snippet = st.session_state.doc_text[:2000]
                combined = (research_input or "") + "\n\nDocument content: " + snippet
                prompt = (
                    f"You are a senior research professor. Run the '{research_mode}' tool in {lang}. "
                    "Be thorough, academic, and rigorous. Use structured output with clear sections. "
                    f"Input: {combined[:2500]}"
                )
                with st.spinner(f"Running {research_mode}..."):
                    result = smart_call(prompt, max_tokens=1500)
                add_xp(35, research_mode)
                if "🔬 Researcher" not in st.session_state.badges:
                    st.session_state.badges.append("🔬 Researcher")
                st.markdown(
                    f"""<div class="debate-bubble">
                    <span class="persona-badge badge-scientist">🔬 {research_mode}</span><br><br>
                    {result}
                    </div>""",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"⚠️ Research tool encountered an error: {e}")

# ════════════════════════════════════════════════════════════════════════════
#  TAB 9 — PROGRESS
# ════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown("### 🏆 Your Progress & Achievements")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown(
            f"""<div class="stat-box">
            <div class="stat-number">{st.session_state.xp}</div>
            <div style="color:#ccc0f0;">Total XP</div>
            </div>""", unsafe_allow_html=True)
    with col_p2:
        st.markdown(
            f"""<div class="stat-box">
            <div class="stat-number">{st.session_state.level}</div>
            <div style="color:#ccc0f0;">Level</div>
            </div>""", unsafe_allow_html=True)
    with col_p3:
        st.markdown(
            f"""<div class="stat-box">
            <div class="stat-number">{len(st.session_state.badges)}</div>
            <div style="color:#ccc0f0;">Badges</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🏅 Earned Badges")
    if st.session_state.badges:
        badge_cols = st.columns(min(len(st.session_state.badges), 4))
        for i, badge in enumerate(st.session_state.badges):
            with badge_cols[i % 4]:
                st.markdown(
                    f"""<div class="card" style="text-align:center;">
                    <div style="font-size:1.5rem;">{badge}</div>
                    </div>""",
                    unsafe_allow_html=True
                )
    else:
        st.info("Complete activities to earn badges!")

    st.subheader("📈 Learning Summary")
    try:
        summary_data = {
            "Metric": ["Document Chunks", "Concepts Extracted", "Debate Rounds", "Scenarios Done", "XP Earned"],
            "Value": [
                len(st.session_state.doc_chunks),
                len(st.session_state.concepts),
                len(st.session_state.debate_history),
                len(st.session_state.scenario_history),
                st.session_state.xp
            ]
        }
        fig_summary = go.Figure(go.Bar(
            x=summary_data["Metric"],
            y=summary_data["Value"],
            marker=dict(
                color=["#564A96", "#6d5fad", "#B75F67", "#c97de0", "#8b5cf6"],
                line=dict(color="rgba(255,255,255,0.2)", width=1)
            )
        ))
        fig_summary.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,22,40,0.6)",
            font=dict(color="white"),
            xaxis=dict(gridcolor="rgba(86,74,150,0.2)", zeroline=False),
            yaxis=dict(gridcolor="rgba(86,74,150,0.2)", zeroline=False),
            height=300,
            margin=dict(l=20, r=20, t=20, b=60)
        )
        st.plotly_chart(fig_summary, use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Could not render learning summary: {e}")

    if st.button("🔄 Reset Session"):
        for k in list(defaults.keys()):
            st.session_state[k] = defaults[k]
        st.success("Session reset!")
        st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """<div style="text-align:center;padding:32px 0 16px 0;color:rgba(168,156,200,0.5);font-size:0.78rem;
    letter-spacing:2px;text-transform:uppercase;">
    AgoraAi &nbsp;|&nbsp; AI Knowledge &amp; Debate Engine &nbsp;|&nbsp;
    Powered by Groq LLaMA-3.3
    </div>""",
    unsafe_allow_html=True
)
