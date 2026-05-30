# 🎬 CineAI — Movie & Series Recommendation Chatbot

A RAG-powered chatbot that recommends movies and series from a 
hand-curated personal dataset of 376 titles.

## Tech Stack
- **Embeddings**: HuggingFace all-MiniLM-L6-v2
- **Vector DB**: FAISS
- **LLM**: Google Gemini 2.5 Flash
- **UI**: Streamlit

## Setup
1. Clone the repo
2. Create virtual environment: `python -m venv cineai-env`
3. Activate: `cineai-env\Scripts\activate`
4. Install: `pip install -r requirements.txt`
5. Build index: `python build_index.py`
6. Add your key to `.streamlit/secrets.toml`
7. Run: `streamlit run app.py`

## Project Structure
```
CineAI/
├── app.py              # Streamlit UI
├── build_index.py      # FAISS index builder
├── cineai.py           # Terminal chatbot
├── movies.csv          # Hand-curated dataset
├── requirements.txt
└── .streamlit/
    └── secrets.toml    # (not pushed to GitHub)
```