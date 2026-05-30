"""
CineAI - Movie & Series Recommendation Chatbot
Uses: FAISS for semantic search (RAG) + Google Gemini API for responses
Run: python cineai.py
"""

import os
from google import genai
from google.genai import types
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ─── CONFIG ──────────────────────────────────────────────────────────────────

FAISS_INDEX_PATH = "faiss_index"   # folder created by build_index.py
CSV_PATH         = "movies.csv"
TOP_K            = 6               # number of similar titles to retrieve
MAX_HISTORY      = 10              # conversation turns to keep in memory


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
    """Load basic stats from CSV to show at startup."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    movies = len(df[df["type"] == "movie"])
    series = len(df[df["type"] == "series"])
    return movies, series


# ─── RAG RETRIEVAL ───────────────────────────────────────────────────────────

def retrieve_context(vectorstore, query: str, k: int = TOP_K) -> str:
    """Run semantic search and return formatted context string."""
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
"""


def build_prompt(history: list, user_query: str, context: str) -> str:
    """Build a single prompt string with history + context injected."""
    prompt_parts = [SYSTEM_PROMPT, "\n\n"]

    # Add conversation history
    if history:
        prompt_parts.append("=== Conversation so far ===\n")
        for turn in history:
            role = "User" if turn["role"] == "user" else "CineAI"
            prompt_parts.append(f"{role}: {turn['content']}\n\n")
        prompt_parts.append("=== End of history ===\n\n")

    # Add current user message with retrieved context
    prompt_parts.append(f"User: {user_query}\n\n")
    prompt_parts.append("---\n")
    prompt_parts.append("CONTEXT (retrieved from personal movie dataset — use only these):\n\n")
    prompt_parts.append(context)
    prompt_parts.append("\n---\n\n")
    prompt_parts.append("CineAI:")

    return "".join(prompt_parts)


# ─── GEMINI API CALL ─────────────────────────────────────────────────────────

def ask_gemini(client, history: list, user_query: str, context: str) -> str:
    """Send prompt to Gemini and return the response text."""
    prompt = build_prompt(history, user_query, context)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text


# ─── CONVERSATION MEMORY ─────────────────────────────────────────────────────

def trim_history(history: list, max_turns: int = MAX_HISTORY) -> list:
    """Keep only the last N turns (each turn = 1 user + 1 assistant message)."""
    max_messages = max_turns * 2
    if len(history) > max_messages:
        history = history[-max_messages:]
    return history


# ─── DISPLAY ─────────────────────────────────────────────────────────────────

def print_banner(movies: int, series: int):
    print("\n" + "═" * 60)
    print("  🎬  CineAI — Your Personal Movie & Series Recommender")
    print("═" * 60)
    print(f"  Dataset: {movies} movies  |  {series} series  |  {movies+series} total")
    print("  Powered by: FAISS (RAG) + Gemini 2.0 Flash (Google AI)")
    print("═" * 60)
    print("  Ask anything — mood, genre, vibe, similar titles, etc.")
    print('  Type "clear" to reset memory  |  "exit" to quit')
    print("═" * 60 + "\n")


def print_response(text: str):
    print("\n" + "─" * 60)
    print("🤖 CineAI:\n")
    print(text)
    print("─" * 60 + "\n")


# ─── MAIN LOOP ───────────────────────────────────────────────────────────────

def main():
    # --- Setup ---
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n❌ Error: GEMINI_API_KEY environment variable not set.")
        print("   Get your free key at: https://aistudio.google.com/app/apikey")
        print("   Then set it with:")
        print("     Windows PowerShell: $env:GEMINI_API_KEY='your-key-here'")
        print("     Mac/Linux:          export GEMINI_API_KEY='your-key-here'\n")
        return

    # Configure Gemini with new SDK
    client = genai.Client(api_key=api_key)

    vectorstore = load_vectorstore()
    movies, series = load_stats(CSV_PATH)
    print_banner(movies, series)

    history = []  # conversation memory: [{"role": "user"/"assistant", "content": "..."}]

    # --- Chat loop ---
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Bye! Happy watching.\n")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("\n👋 Bye! Happy watching.\n")
            break

        if user_input.lower() == "clear":
            history = []
            print("\n🔄 Memory cleared. Fresh start!\n")
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

        # 4. Update history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response_text})
        history = trim_history(history)


if __name__ == "__main__":
    main()