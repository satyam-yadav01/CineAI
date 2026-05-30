"""
CineAI - Movie & Series Recommendation Chatbot
Uses: FAISS for semantic search (RAG) + Google Gemini API for responses
Run: python cineai.py
"""

import os
import json
from datetime import datetime
from google import genai
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ─── CONFIG ──────────────────────────────────────────────────────────────────

FAISS_INDEX_PATH = "faiss_index"
CSV_PATH         = "movies.csv"
HISTORY_FILE     = "terminal_history.json" # persistent memory file
TOP_K            = 6
MAX_HISTORY      = 10                    # turns to keep in memory (and save)


# ─── PERSISTENT MEMORY ───────────────────────────────────────────────────────

def load_history() -> list:
    """Load chat history from file. Returns empty list if file doesn't exist."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            history = data.get("history", [])
            last_seen = data.get("last_seen", "unknown")
            if history:
                print(f"💾 Memory loaded — {len(history) // 2} previous turn(s) found.")
                print(f"   Last session: {last_seen}\n")
            return history
    return []


def save_history(history: list):
    """Save chat history to file with timestamp."""
    data = {
        "last_seen": datetime.now().strftime("%d %B %Y, %I:%M %p"),
        "total_turns": len(history) // 2,
        "history": history
    }
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clear_history() -> list:
    """Wipe history file and return empty list."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    print("\n🔄 Memory cleared. Fresh start!\n")
    return []


# ─── LOAD MODELS ─────────────────────────────────────────────────────────────

def load_vectorstore():
    print("⏳ Loading FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ FAISS index loaded.\n")
    return vectorstore


def load_stats(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    movies = len(df[df["type"] == "movie"])
    series = len(df[df["type"] == "series"])
    return movies, series


# ─── RAG RETRIEVAL ───────────────────────────────────────────────────────────

def retrieve_context(vectorstore, query: str, k: int = TOP_K) -> str:
    results = vectorstore.similarity_search(query, k=k)
    context_parts = [doc.page_content.strip() for doc in results]
    return "\n\n---\n\n".join(context_parts)


# ─── PROMPT ──────────────────────────────────────────────────────────────────

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
- If the context has fewer than 3 relevant matches, be honest about it.
- Match mood/vibe first, genre second — users feel more than they categorize.
- Use the "My Rating" field to flag hidden gems (My Rating > IMDb Rating).
- Keep the tone warm, direct, and confident. No corporate AI voice.
- If the user asks something outside movies/series, politely redirect them.
- You have access to past conversation history — use it to give better, more personalized recommendations over time.
"""


def build_prompt(history: list, user_query: str, context: str) -> str:
    prompt_parts = [SYSTEM_PROMPT, "\n\n"]

    if history:
        prompt_parts.append("=== Conversation so far ===\n")
        for turn in history:
            role = "User" if turn["role"] == "user" else "CineAI"
            prompt_parts.append(f"{role}: {turn['content']}\n\n")
        prompt_parts.append("=== End of history ===\n\n")

    prompt_parts.append(f"User: {user_query}\n\n")
    prompt_parts.append("---\n")
    prompt_parts.append("CONTEXT (retrieved from personal movie dataset — use only these):\n\n")
    prompt_parts.append(context)
    prompt_parts.append("\n---\n\n")
    prompt_parts.append("CineAI:")

    return "".join(prompt_parts)


# ─── GEMINI API CALL ─────────────────────────────────────────────────────────

def ask_gemini(client, history: list, user_query: str, context: str) -> str:
    prompt = build_prompt(history, user_query, context)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


# ─── CONVERSATION MEMORY ─────────────────────────────────────────────────────

def trim_history(history: list, max_turns: int = MAX_HISTORY) -> list:
    max_messages = max_turns * 2
    if len(history) > max_messages:
        history = history[-max_messages:]
    return history


# ─── DISPLAY ─────────────────────────────────────────────────────────────────

def print_banner(movies: int, series: int, has_memory: bool):
    print("\n" + "═" * 60)
    print("  🎬  CineAI — Your Personal Movie & Series Recommender")
    print("═" * 60)
    print(f"  Dataset : {movies} movies  |  {series} series  |  {movies+series} total")
    print(f"  Memory  : {'✅ Persistent (loaded from last session)' if has_memory else '🆕 Fresh session'}")
    print("  Powered : FAISS (RAG) + Gemini 2.5 Flash (Google AI)")
    print("═" * 60)
    print("  Ask anything — mood, genre, vibe, similar titles, etc.")
    print('  Commands: "clear" = wipe memory  |  "exit" = quit')
    print("═" * 60 + "\n")


def print_response(text: str):
    print("\n" + "─" * 60)
    print("🤖 CineAI:\n")
    print(text)
    print("─" * 60 + "\n")


# ─── MAIN LOOP ───────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n❌ Error: GEMINI_API_KEY not set.")
        print("   Windows: $env:GEMINI_API_KEY='your-key-here'")
        print("   Mac/Linux: export GEMINI_API_KEY='your-key-here'\n")
        return

    client = genai.Client(api_key=api_key)
    vectorstore = load_vectorstore()
    movies, series = load_stats(CSV_PATH)

    # Load persistent memory
    history = load_history()
    print_banner(movies, series, has_memory=len(history) > 0)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            save_history(history)
            print("\n\n💾 Memory saved. See you next time!\n")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            save_history(history)
            print("\n💾 Memory saved. See you next time! 👋\n")
            break

        if user_input.lower() == "clear":
            history = clear_history()
            continue

        # 1. Retrieve relevant context from FAISS
        context = retrieve_context(vectorstore, user_input)

        # 2. Call Gemini
        print("\n⏳ Thinking...", end="\r")
        try:
            response_text = ask_gemini(client, history, user_input, context)
        except Exception as e:
            print(f"\n❌ API Error: {e}\n")
            continue

        # 3. Display response
        print_response(response_text)

        # 4. Update and save history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response_text})
        history = trim_history(history)
        save_history(history)   # save after every message


if __name__ == "__main__":
    main()
