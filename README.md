# SHL Assessment Recommender

**Live Demo:** [https://rag-system-frontend-nine.vercel.app/](https://rag-system-frontend-nine.vercel.app/)

An AI-powered conversational agent that helps hiring managers find the right SHL assessments for any role. Built with a RAG (Retrieval-Augmented Generation) pipeline on top of the full SHL product catalog.

---

## The Problem

SHL offers **hundreds of assessments** spanning cognitive ability, personality, situational judgement, simulations, knowledge tests, and more. A hiring manager looking to build an assessment battery for a specific role — say, a senior backend engineer or a contact-centre agent — has no easy way to navigate that catalog. They would need to:

- Know what assessment types exist (OPQ, MQ, Verify, etc.)
- Understand which ones map to their job level and domain
- Manually cross-reference job families, languages, duration constraints, and remote-testing support

This is slow, error-prone, and requires deep SHL product knowledge most hiring managers simply don't have.

---

## My Approach

Instead of building a static filter UI, I designed a **conversational recommender** that behaves like an SHL expert sitting across the table.

1. **Scrape the full SHL catalog** — automated scraper extracts every Individual Test Solution: name, URL, description, test type, job levels, languages, duration, and remote-testing support.

2. **Embed the catalog into a vector store** — each assessment is embedded using Gemini's embedding model and stored in ChromaDB for semantic similarity search.

3. **RAG + LLM reasoning** — on every user turn, the agent:
   - Builds a retrieval query from the last 4 user messages (with domain-specific boosts for sales, leadership, safety, etc.)
   - Fetches the top-20 semantically closest assessments from ChromaDB
   - Always injects OPQ32r (the flagship personality questionnaire) into the context
   - Sends the retrieved catalog context + conversation history to Gemini 2.5 Flash
   - Validates every recommended URL back against ChromaDB to eliminate hallucinations

4. **Structured conversation flow** — the agent follows four explicit behaviors: **Clarify** (gather seniority/domain/job-family if missing), **Recommend**, **Refine** (add/drop/swap), and **Compare**. All responses are strict JSON.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client / Frontend                     │
│                    (POST /chat, GET /health)                 │
└───────────────────────────┬─────────────────────────────────┘
                            │  HTTP
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               FastAPI  (AWS EC2 — port 8000)                │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │                   /chat  endpoint                   │  │
│   │                                                     │  │
│   │  1. build_retrieval_query()                         │  │
│   │       └─ last 4 user msgs + domain boosts           │  │
│   │                                                     │  │
│   │  2. embed_query()  ──► Gemini Embedding API         │  │
│   │       └─ gemini-embedding-2                         │  │
│   │                                                     │  │
│   │  3. ChromaDB.query()  (cosine similarity, top-20)   │  │
│   │       └─ always inject OPQ32r                       │  │
│   │                                                     │  │
│   │  4. call_llm()  ──► Gemini 2.5 Flash                │  │
│   │       └─ system prompt + catalog context            │  │
│   │          + conversation history                     │  │
│   │                                                     │  │
│   │  5. validate_and_clean()                            │  │
│   │       └─ verify every URL against ChromaDB          │  │
│   │          drop hallucinated entries                  │  │
│   │                                                     │  │
│   │  6. return ChatResponse (JSON)                      │  │
│   └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
          ┌─────────────────┴──────────────────┐
          ▼                                    ▼
┌──────────────────┐                ┌─────────────────────┐
│    ChromaDB      │                │   Google Gemini API  │
│  (local persist) │                │                     │
│                  │                │  gemini-embedding-2  │
│  ~shl_assessments│                │  gemini-2.5-flash    │
│  collection      │                └─────────────────────┘
└──────────────────┘


────────────────────── Offline Pipeline ──────────────────────

  Scraping.py                generate_embeddings.py
  ───────────                ──────────────────────
  SHL catalog pages          shl_catalog.csv
  → shl_catalog.csv    ───►  → Gemini embeddings
                             → ChromaDB upsert
```

---

## Data Pipeline (Offline)

### Step 1 — Scrape (`Scraping.py`)
- Paginates through `shl.com/products/product-catalog/?type=1`
- Extracts: name, URL, test type codes, remote testing, adaptive IRT from the listing table
- Visits each product detail page to extract: description, job levels, languages, duration
- Outputs `shl_catalog.csv`

### Step 2 — Embed & Index (`generate_embeddings.py`)
- Loads `shl_catalog.csv` via pandas
- Builds a rich embedding text per assessment (name + type + job levels + description)
- Calls `gemini-embedding-2` for each row (with checkpoint to `embeddings.json` for resumability)
- Bulk-inserts into ChromaDB with cosine similarity index (`hnsw:space=cosine`)

---

## API

### `GET /health`
```json
{ "status": "ok" }
```

### `POST /chat`
**Request:**
```json
{
  "messages": [
    { "role": "user", "content": "I need assessments for a senior backend engineer" }
  ]
}
```

**Response:**
```json
{
  "reply": "Could you confirm the seniority and domain — backend, data science, DevOps?",
  "recommendations": [],
  "end_of_conversation": false
}
```

Once enough context is gathered:
```json
{
  "reply": "Here are the recommended assessments for a senior backend engineer...",
  "recommendations": [
    {
      "name": "Verify - Coding Pro",
      "url": "https://www.shl.com/products/product-catalog/view/verify-coding-pro/",
      "test_type": "K"
    },
    {
      "name": "Occupational Personality Questionnaire OPQ32r",
      "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
      "test_type": "P"
    }
  ],
  "end_of_conversation": false
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI |
| **LLM** | Google Gemini 2.5 Flash (`gemini-2.5-flash`) |
| **Embeddings** | Google Gemini Embedding (`gemini-embedding-2`) |
| **Vector Store** | ChromaDB (local persistent, cosine similarity) |
| **Data Processing** | pandas |
| **Web Scraping** | requests + BeautifulSoup4 |
| **Deployment** | AWS EC2 (`13.233.138.155:8000`) |
| **Server** | Uvicorn (ASGI) |
| **Config** | python-dotenv |
| **Validation** | Pydantic v2 |

---

## Project Structure

```
SHL/
├── main.py                  # FastAPI app — chat endpoint, RAG pipeline, LLM agent
├── Scraping.py              # SHL catalog scraper → shl_catalog.csv
├── generate_embeddings.py   # Embedding pipeline → ChromaDB
├── shl_catalog.csv          # Scraped catalog (gitignored)
├── embeddings.json          # Embedding checkpoint (gitignored)
├── chroma_db/               # ChromaDB persistent store (gitignored)
├── .env                     # API keys (gitignored)
└── README.md
```

---

## Running Locally

### Prerequisites
```
Python 3.11+
pip install fastapi uvicorn chromadb google-genai pandas python-dotenv pydantic requests beautifulsoup4 tqdm
```

### Environment
Create a `.env` file:
```
Gemini_Api_Key = "your-gemini-api-key"
```

### Build the catalog (first time only)
```bash
# Step 1 — scrape SHL catalog
python Scraping.py

# Step 2 — generate embeddings and populate ChromaDB
python generate_embeddings.py
```

### Start the API
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API is live at `http://localhost:8000`. The deployed instance runs on AWS EC2 at `http://13.233.138.155:8000`.

---

## Key Design Decisions

- **OPQ32r always included** — the flagship personality questionnaire is force-injected into every retrieval context and recommended by default unless the user explicitly drops it. Personality assessment is almost always relevant.
- **Domain boosting** — the retrieval query is augmented with domain-specific terms (sales, leadership, safety, contact-centre, graduate) to improve recall on niche roles where the user's wording might not naturally match catalog language.
- **URL validation** — every URL the LLM recommends is verified back against ChromaDB before being returned. This hard-blocks hallucinated assessments from ever reaching the client.
- **Conversation state is stateless** — the full message history is sent on every request. No server-side session storage needed, making horizontal scaling trivial.
- **Structured JSON-only LLM output** — the system prompt enforces strict JSON responses. A robust `clean_json_response()` parser strips thinking blocks and markdown fences before parsing, with a fallback regex extractor.
- **Checkpoint-based embedding pipeline** — embeddings are saved to `embeddings.json` after each row. If the pipeline is interrupted (rate limits, network failures), it resumes from where it left off without re-embedding anything.
