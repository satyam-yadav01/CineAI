"""
CineAI - Streamlit Web App (Merged: new.py UI + app.py history system)
Run: streamlit run app.py
"""

import os
import json
import time
import uuid
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai


# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CineAI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CONFIG ──────────────────────────────────────────────────────────────────

FAISS_INDEX_PATH = "faiss_index"
CSV_PATH         = "movies.csv"
HISTORY_FILE     = "chat_history.json"
TOP_K            = 6
MAX_HISTORY      = 10

# ─── CSS (new.py's clean blue-purple theme) ──────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: #070711;
    color: #e2e8f0;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background:
        radial-gradient(ellipse at 20% 20%, rgba(99,102,241,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(139,92,246,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(59,130,246,0.04) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stSidebar"] {
    background: #0d0d1a !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span { color: #94a3b8 !important; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

[data-testid="stChatMessage"] {
    background: rgba(15,15,30,0.8) !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    border-radius: 16px !important;
    margin-bottom: 12px !important;
}
[data-testid="stChatMessage"]:hover {
    border-color: rgba(99,102,241,0.25) !important;
}

[data-testid="stChatInput"] {
    background: rgba(13,13,26,0.9) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.08) !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(15,15,35,0.9), rgba(20,20,45,0.9)) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    transition: all 0.3s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(139,92,246,0.4) !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.1) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #818cf8, #a78bfa) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── All sidebar buttons ── */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(15,15,35,0.6) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    transition: all 0.2s ease !important;
    text-align: left !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99,102,241,0.12) !important;
    border-color: rgba(99,102,241,0.35) !important;
    color: #a5b4fc !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2)) !important;
    border-color: rgba(99,102,241,0.5) !important;
    color: #a5b4fc !important;
    box-shadow: 0 0 12px rgba(99,102,241,0.15) !important;
}

/* ── Main area buttons ── */
.main .stButton > button {
    background: rgba(15,15,35,0.7) !important;
    color: #a5b4fc !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    transition: all 0.25s ease !important;
}
.main .stButton > button:hover {
    background: rgba(99,102,241,0.15) !important;
    border-color: rgba(139,92,246,0.5) !important;
    color: #c4b5fd !important;
    box-shadow: 0 0 15px rgba(99,102,241,0.15) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stSelectbox"] > div > div {
    background: rgba(13,13,26,0.9) !important;
    border-color: rgba(99,102,241,0.25) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

[data-testid="stTabs"] [role="tablist"] {
    background: rgba(10,10,25,0.5);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid rgba(99,102,241,0.12);
    gap: 4px;
}
[data-testid="stTabs"] button[role="tab"] {
    color: #64748b !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #a5b4fc !important;
    border: none !important;
    box-shadow: 0 0 12px rgba(99,102,241,0.2) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] {
    background: rgba(13,13,26,0.6) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 12px !important;
}

hr { border-color: rgba(99,102,241,0.1) !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 2px; }

/* ── History group labels ── */
.hist-group {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #334155;
    margin: 14px 0 6px 0;
}
</style>
""", unsafe_allow_html=True)


# ─── CACHED RESOURCES ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

@st.cache_data(show_spinner=False)
def load_dataframe():
    return pd.read_csv(CSV_PATH, encoding="utf-8-sig")

@st.cache_resource(show_spinner=False)
def load_gemini_client():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    return genai.Client(api_key=api_key)


# ─── MULTI-SESSION HISTORY (from app.py) ─────────────────────────────────────

def load_all_sessions() -> dict:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_session(session_id: str, history: list, first_prompt: str):
    sessions = load_all_sessions()
    if session_id not in sessions:
        title = first_prompt[:28] + "..." if len(first_prompt) > 28 else first_prompt
        sessions[session_id] = {
            "title": title,
            "timestamp": time.time(),
            "history": history
        }
    else:
        sessions[session_id]["history"] = history
        sessions[session_id]["timestamp"] = time.time()
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)

def group_sessions_by_time(sessions: dict) -> dict:
    now   = datetime.now()
    today = now.date()
    groups = {"Today": [], "Yesterday": [], "Previous 7 Days": [], "Older": []}
    for s_id, s_data in sorted(sessions.items(), key=lambda x: x[1].get("timestamp", 0), reverse=True):
        d = datetime.fromtimestamp(s_data.get("timestamp", 0)).date()
        if d == today:
            groups["Today"].append((s_id, s_data))
        elif d == today - timedelta(days=1):
            groups["Yesterday"].append((s_id, s_data))
        elif d > today - timedelta(days=7):
            groups["Previous 7 Days"].append((s_id, s_data))
        else:
            groups["Older"].append((s_id, s_data))
    return groups


# ─── RAG + GEMINI ────────────────────────────────────────────────────────────

def retrieve_context(vectorstore, query: str, k: int = TOP_K) -> str:
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join([doc.page_content.strip() for doc in results])

SYSTEM_PROMPT = """You are CineAI, a passionate and knowledgeable movie and web series recommendation assistant. You have a personal, opinionated voice — like a cinephile friend who has seen everything.

Your job is to help users find movies and series they'll love, using ONLY the provided context (a curated personal dataset of ~376 titles hand-picked by the creator).

RESPONSE FORMAT (always follow this structure):
1. A short, engaging intro (1–2 sentences) that directly addresses what the user asked.
2. A "Recommendations" section with 3–5 picks. For each:
   - Title + Year + Type (Movie/Series)
   - Genre | Mood
   - IMDb: X.X | My Rating: X.X
   - A 2-sentence reason why this fits the user's request — be specific, not generic.
3. A brief closing note (1 sentence) offering a follow-up angle.

RULES:
- ONLY recommend titles that appear in the provided context. Never invent titles.
- Match mood/vibe first, genre second.
- Flag hidden gems where My Rating > IMDb Rating.
- Keep the tone warm, direct, and confident.
- Use past conversation history for personalized recommendations.
"""

def build_prompt(history, user_query, context):
    parts = [SYSTEM_PROMPT, "\n\n"]
    if history:
        parts.append("=== Conversation so far ===\n")
        for turn in history:
            role = "User" if turn["role"] == "user" else "CineAI"
            parts.append(f"{role}: {turn['content']}\n\n")
        parts.append("=== End of history ===\n\n")
    parts += [f"User: {user_query}\n\n", "---\nCONTEXT:\n\n", context, "\n---\n\nCineAI:"]
    return "".join(parts)

def ask_gemini(client, history, user_query, context):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=build_prompt(history, user_query, context)
    )
    return response.text

def trim_history(history):
    max_messages = MAX_HISTORY * 2
    return history[-max_messages:] if len(history) > max_messages else history


# ─── SESSION STATE ────────────────────────────────────────────────────────────

if "session_id"   not in st.session_state: st.session_state.session_id   = str(uuid.uuid4())
if "messages"     not in st.session_state: st.session_state.messages     = []
if "history"      not in st.session_state: st.session_state.history      = []
if "page"         not in st.session_state: st.session_state.page         = "Chat"
if "quick_prompt" not in st.session_state: st.session_state.quick_prompt = None


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:

    # Logo
    st.markdown("""
    <div style="padding:16px 4px 20px 4px;">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;
                    letter-spacing:-0.5px;color:#e2e8f0;">
            🎬 Cine<span style="background:linear-gradient(135deg,#818cf8,#a78bfa);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">AI</span>
        </div>
        <div style="font-size:11px;color:#475569;margin-top:3px;">
            RAG-powered · Personal Dataset · 376 titles
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Navigation
    st.markdown('<p style="font-size:10px;font-weight:600;letter-spacing:0.1em;color:#475569;text-transform:uppercase;margin-bottom:8px;">Navigation</p>', unsafe_allow_html=True)
    for icon, label in [("💬","Chat"),("🎞️","Browse Dataset"),("📊","Stats"),("ℹ️","About")]:
        is_active = st.session_state.page == label
        if st.button(f"{icon}  {label}", use_container_width=True,
                     type="primary" if is_active else "secondary", key=f"nav_{label}"):
            st.session_state.page = label
            st.rerun()

    st.divider()

    # Quick prompts
    st.markdown('<p style="font-size:10px;font-weight:600;letter-spacing:0.1em;color:#475569;text-transform:uppercase;margin-bottom:8px;">Quick Prompts</p>', unsafe_allow_html=True)
    for icon, label in [
        ("⚡","Something intense & thrilling"), ("😂","Make me laugh hard"),
        ("💔","Emotional & heartbreaking"),     ("🌀","Mind-bending sci-fi"),
        ("💜","Romantic & feel-good"),           ("👁️","Scary & disturbing"),
        ("💎","Hidden gems only"),               ("📖","Based on true story"),
    ]:
        if st.button(f"{icon} {label}", use_container_width=True, key=f"qp_{label}"):
            st.session_state.quick_prompt = label
            st.session_state.page = "Chat"
            st.rerun()

    st.divider()

    # Chat History (from app.py)
    st.markdown('<p style="font-size:10px;font-weight:600;letter-spacing:0.1em;color:#475569;text-transform:uppercase;margin-bottom:8px;">Chat History</p>', unsafe_allow_html=True)

    if st.button("＋  New Chat", use_container_width=True, key="new_chat"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages   = []
        st.session_state.history    = []
        st.session_state.page       = "Chat"
        st.rerun()

    sessions = load_all_sessions()
    grouped  = group_sessions_by_time(sessions)

    for group_name, items in grouped.items():
        if items:
            st.markdown(f'<div class="hist-group">{group_name}</div>', unsafe_allow_html=True)
            for s_id, s_data in items:
                title = s_data.get("title", "New Conversation")
                if st.button(title, key=f"sess_{s_id}", use_container_width=True, type="secondary"):
                    st.session_state.session_id = s_id
                    st.session_state.history    = s_data["history"]
                    st.session_state.messages   = s_data["history"].copy()
                    st.session_state.page       = "Chat"
                    st.rerun()

    st.divider()
    if st.button("🗑️  Clear All History", use_container_width=True, key="clear_hist"):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.history    = []
        st.session_state.messages   = []
        st.success("All history cleared!")
        st.rerun()


# ─── LOAD RESOURCES ──────────────────────────────────────────────────────────

with st.spinner("Loading CineAI..."):
    try:
        vectorstore  = load_vectorstore()
        df           = load_dataframe()
        client       = load_gemini_client()
        resources_ok = True
    except Exception as e:
        st.error(f"Failed to load: {e}")
        resources_ok = False
        st.stop()


# ════════════════════════════════════════════════════════════
#  PAGE: CHAT
# ════════════════════════════════════════════════════════════

if st.session_state.page == "Chat":

    if not st.session_state.messages:

        # Hero (from new.py)
        st.markdown("""
        <div style="text-align:center;padding:50px 20px 30px;">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:52px;font-weight:700;
                        letter-spacing:-2px;line-height:1.1;margin-bottom:14px;">
                <span style="color:#e2e8f0;">What should you</span><br>
                <span style="background:linear-gradient(135deg,#818cf8 0%,#a78bfa 50%,#60a5fa 100%);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">watch tonight?</span>
            </div>
            <div style="font-size:16px;color:#64748b;max-width:480px;margin:0 auto 36px;line-height:1.6;">
                Tell me your mood, a vibe, a genre, or a title you loved —
                I'll find exactly what you need from a curated collection of 376 titles.
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎥 Movies",        len(df[df["type"] == "movie"]))
        c2.metric("📺 Series",        len(df[df["type"] == "series"]))
        c3.metric("⭐ Avg IMDb",      f"{df['imdb_rating'].astype(float).mean():.1f}")
        c4.metric("💜 Avg My Rating", f"{df['my_rating'].astype(float).mean():.1f}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p style="text-align:center;color:#475569;font-size:13px;font-weight:500;'
                    'letter-spacing:0.05em;text-transform:uppercase;margin-bottom:14px;">Try one of these</p>',
                    unsafe_allow_html=True)

        examples = [
            "I want something mind-bending",
            "Best movies to watch with family",
            "Dark and disturbing thrillers",
            "Something like Interstellar",
            "Funny but smart comedies",
            "Hidden gems with high my rating",
        ]
        r1, r2 = st.columns(3), st.columns(3)
        for i, ex in enumerate(examples):
            col = r1[i] if i < 3 else r2[i-3]
            with col:
                if st.button(f'"{ex}"', use_container_width=True, key=f"ex_{i}"):
                    st.session_state.quick_prompt = ex
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"]=="assistant" else "🧑"):
                st.markdown(msg["content"])

    # Handle quick prompt
    if st.session_state.quick_prompt:
        prompt = st.session_state.quick_prompt
        st.session_state.quick_prompt = None
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Finding the perfect picks..."):
                response = ask_gemini(client, st.session_state.history, prompt,
                                      retrieve_context(vectorstore, prompt))
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.history.append({"role": "user",      "content": prompt})
        st.session_state.history.append({"role": "assistant", "content": response})
        st.session_state.history = trim_history(st.session_state.history)
        save_session(st.session_state.session_id, st.session_state.history, prompt)
        st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask CineAI anything about movies or series..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Finding the perfect picks..."):
                response = ask_gemini(client, st.session_state.history, prompt,
                                      retrieve_context(vectorstore, prompt))
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.history.append({"role": "user",      "content": prompt})
        st.session_state.history.append({"role": "assistant", "content": response})
        st.session_state.history = trim_history(st.session_state.history)
        save_session(st.session_state.session_id, st.session_state.history, prompt)
        st.rerun()


# ════════════════════════════════════════════════════════════
#  PAGE: BROWSE DATASET
# ════════════════════════════════════════════════════════════

elif st.session_state.page == "Browse Dataset":

    st.markdown('<div style="font-family:\'Space Grotesk\',sans-serif;font-size:26px;font-weight:700;'
                'color:#e2e8f0;margin-bottom:4px;">Browse Dataset '
                '<span style="font-size:13px;color:#475569;font-family:Inter;">Your curated collection</span></div>',
                unsafe_allow_html=True)
    st.divider()

    with st.expander("🔍  Filters & Sorting", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1: type_filter  = st.selectbox("Type", ["All","movie","series"])
        with c2:
            all_genres = sorted(set(g.strip() for genres in df["genre"] for g in genres.split(",")))
            genre_filter = st.selectbox("Genre", ["All"] + all_genres)
        with c3: sort_by = st.selectbox("Sort by", ["My Rating ↓","IMDb Rating ↓","Year ↓","Title A–Z"])
        c4, c5 = st.columns(2)
        with c4: year_range = st.slider("Year", int(df["year"].min()), int(df["year"].max()),
                                         (int(df["year"].min()), int(df["year"].max())))
        with c5: min_imdb = st.slider("Min IMDb", 0.0, 10.0, 0.0, 0.1)

    filtered = df.copy()
    if type_filter  != "All": filtered = filtered[filtered["type"]  == type_filter]
    if genre_filter != "All": filtered = filtered[filtered["genre"].str.contains(genre_filter, case=False, na=False)]
    filtered = filtered[(filtered["year"].astype(int) >= year_range[0]) & (filtered["year"].astype(int) <= year_range[1])]
    filtered = filtered[filtered["imdb_rating"].astype(float) >= min_imdb]
    sc, sa = {"My Rating ↓":("my_rating",False),"IMDb Rating ↓":("imdb_rating",False),
              "Year ↓":("year",False),"Title A–Z":("title",True)}[sort_by]
    filtered = filtered.sort_values(sc, ascending=sa)

    st.markdown(f'<p style="color:#475569;font-size:13px;margin-bottom:16px;">Showing '
                f'<b style="color:#818cf8;">{len(filtered)}</b> of {len(df)} titles</p>', unsafe_allow_html=True)

    for i in range(0, len(filtered), 3):
        cols = st.columns(3)
        for col, (_, item) in zip(cols, filtered.iloc[i:i+3].iterrows()):
            with col:
                imdb = float(item["imdb_rating"])
                my_r = float(item["my_rating"])
                gem  = "💎 " if my_r > imdb else ""
                type_icon = "🎥" if item["type"] == "movie" else "📺"
                val_color = "#34d399" if my_r > imdb else "#818cf8"
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,rgba(13,13,26,0.95),rgba(18,18,40,0.95));
                    border:1px solid rgba(99,102,241,0.15);border-radius:14px;padding:16px;
                    margin-bottom:14px;min-height:190px;">
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:600;
                        color:#e2e8f0;margin-bottom:6px;line-height:1.3;">
                        {gem}{item['title']}
                        <span style="color:#334155;font-size:12px;font-weight:400;"> {item['year']}</span>
                    </div>
                    <div style="margin-bottom:8px;display:flex;flex-wrap:wrap;gap:5px;">
                        <span style="font-size:11px;padding:2px 8px;border-radius:6px;
                            background:rgba(99,102,241,0.12);color:#818cf8;
                            border:1px solid rgba(99,102,241,0.2);">{type_icon} {item['type']}</span>
                        <span style="font-size:11px;padding:2px 8px;border-radius:6px;
                            background:rgba(139,92,246,0.1);color:#a78bfa;
                            border:1px solid rgba(139,92,246,0.2);">{item['genre'].split(',')[0].strip()}</span>
                    </div>
                    <div style="font-size:12px;color:#475569;margin-bottom:8px;line-height:1.5;">
                        🎭 {str(item['mood'])[:45]}{'...' if len(str(item['mood']))>45 else ''}
                    </div>
                    <div style="font-size:12px;color:#475569;margin-bottom:10px;line-height:1.5;">
                        {str(item['description'])[:90]}{'...' if len(str(item['description']))>90 else ''}
                    </div>
                    <div style="display:flex;gap:8px;">
                        <span style="font-size:12px;font-weight:600;padding:3px 10px;border-radius:6px;
                            background:rgba(251,191,36,0.1);color:#fbbf24;
                            border:1px solid rgba(251,191,36,0.2);">⭐ {imdb}</span>
                        <span style="font-size:12px;font-weight:600;padding:3px 10px;border-radius:6px;
                            background:rgba(99,102,241,0.1);color:{val_color};
                            border:1px solid rgba(99,102,241,0.2);">💜 {my_r}</span>
                    </div>
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: STATS
# ════════════════════════════════════════════════════════════

elif st.session_state.page == "Stats":

    st.markdown('<div style="font-family:\'Space Grotesk\',sans-serif;font-size:26px;font-weight:700;'
                'color:#e2e8f0;margin-bottom:4px;">Dataset Stats '
                '<span style="font-size:13px;color:#475569;font-family:Inter;">A breakdown of your collection</span></div>',
                unsafe_allow_html=True)
    st.divider()

    df_n = df.copy()
    df_n["imdb_rating"] = df_n["imdb_rating"].astype(float)
    df_n["my_rating"]   = df_n["my_rating"].astype(float)
    gems_n = len(df_n[df_n["my_rating"] > df_n["imdb_rating"]])

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Titles", len(df))
    c2.metric("🎥 Movies",    len(df[df["type"]=="movie"]))
    c3.metric("📺 Series",    len(df[df["type"]=="series"]))
    c4.metric("⭐ Avg IMDb",  f"{df_n['imdb_rating'].mean():.2f}")
    c5.metric("💎 Hidden Gems", gems_n)

    st.divider()

    tab1, tab2, tab3 = st.tabs(["  🎭  Genre Breakdown  ", "  📅  By Year  ", "  💎  Hidden Gems  "])

    with tab1:
        gc = {}
        for genres in df["genre"]:
            for g in genres.split(","):
                g = g.strip(); gc[g] = gc.get(g, 0) + 1
        st.bar_chart(pd.DataFrame(sorted(gc.items(), key=lambda x:-x[1]),
                     columns=["Genre","Count"]).set_index("Genre"), color="#818cf8")

    with tab2:
        yc = df.groupby("year").size().reset_index(name="Count")
        yc["year"] = yc["year"].astype(int)
        st.bar_chart(yc.set_index("year"), color="#a78bfa")

    with tab3:
        st.markdown('<p style="color:#64748b;font-size:13px;margin-bottom:16px;">'
                    'Titles where your rating beats IMDb — your hidden gems 💎</p>', unsafe_allow_html=True)
        gems = df_n[df_n["my_rating"] > df_n["imdb_rating"]].copy()
        gems["Difference"] = (gems["my_rating"] - gems["imdb_rating"]).round(1)
        st.dataframe(
            gems.sort_values("Difference", ascending=False)
                [["title","type","genre","imdb_rating","my_rating","Difference","year"]]
                .rename(columns={"title":"Title","type":"Type","genre":"Genre",
                                  "imdb_rating":"IMDb","my_rating":"My Rating","year":"Year"}),
            use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ════════════════════════════════════════════════════════════

elif st.session_state.page == "About":

    st.markdown('<div style="font-family:\'Space Grotesk\',sans-serif;font-size:26px;'
                'font-weight:700;color:#e2e8f0;margin-bottom:4px;">About CineAI</div>',
                unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
### What is CineAI?

CineAI is a personal movie and series recommendation chatbot built using
**RAG (Retrieval Augmented Generation)** — a technique that combines
semantic search with a large language model.

Instead of relying on generic databases, CineAI works from a **hand-curated
personal dataset** of 376 titles with personal ratings, moods, themes,
and descriptions written by the creator.

---

### How it works

1. **You ask** — mood, genre, vibe, or a title you loved
2. **FAISS semantic search** finds the most relevant titles from the dataset
3. **Gemini 2.5 Flash** reads those titles and generates a natural response
4. **Persistent memory** keeps context across sessions

---

### Tech Stack

| Component | Technology |
|---|---|
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Database | FAISS |
| LLM | Google Gemini 2.5 Flash |
| UI Framework | Streamlit |
| Dataset | Custom CSV · 376 titles · Hand-curated |
| Memory | JSON persistent storage |

---

### Why the dataset matters

Most systems use scraped IMDb or TMDB data. CineAI uses a **personally
rated** dataset — the `My Rating` field reflects the creator's actual
opinion, not crowd scores. This makes hidden gem detection meaningful.
        """)

    with col2:
        df_a = df.copy()
        df_a["imdb_rating"] = df_a["imdb_rating"].astype(float)
        df_a["my_rating"]   = df_a["my_rating"].astype(float)
        st.metric("Total Titles",  len(df))
        st.metric("Movies",        len(df[df["type"]=="movie"]))
        st.metric("Series",        len(df[df["type"]=="series"]))
        st.metric("Hidden Gems 💎", len(df_a[df_a["my_rating"] > df_a["imdb_rating"]]))
        st.divider()
        st.markdown("""
<div style="color:#64748b;font-size:13px;line-height:2.2;">
🐍 Python 3.x<br>
🤗 HuggingFace Transformers<br>
⚡ FAISS Vector Search<br>
✨ Google Gemini 2.5 Flash<br>
🎈 Streamlit
</div>""", unsafe_allow_html=True)
