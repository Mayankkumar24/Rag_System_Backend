

import os, time, logging, sys, re, json
import pandas as pd
import chromadb
from google import genai
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("embedding_log.txt", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

load_dotenv()

GEMINI_API_KEY   = os.getenv("Gemini_Api_Key")
CSV_PATH         = "shl_catalog.csv"
CHROMA_DB_PATH   = "./chroma_db"
COLLECTION_NAME  = "shl_assessments"
EMBEDDING_MODEL  = "gemini-embedding-2"
EMBEDDINGS_JSON  = "embeddings.json" 
DELAY_SECONDS    = 5                   

if not GEMINI_API_KEY:
    log.error("Gemini_Api_Key not found in .env file. Exiting.")
    sys.exit(1)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)



def load_catalog(path: str) -> pd.DataFrame:
    log.info("Loading CSV from %s ...", path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    text_cols = ["name", "url", "description", "test_type_codes",
                 "test_type_labels", "job_levels", "languages"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    for col in ["remote_testing", "adaptive_irt"]:
        if col in df.columns:
            df[col] = df[col].fillna(False)

    if "duration_minutes" in df.columns:
        df["duration_minutes"] = df["duration_minutes"].fillna("").astype(str).str.strip()

    df = df[df["name"].str.len() > 0]
    df = df[df["url"].str.len() > 0]
    df = df.drop_duplicates(subset=["url"])
    log.info("Clean rows: %d", len(df))
    return df.reset_index(drop=True)


def build_chroma_id(row: pd.Series) -> str:
    url   = row.get("url", "")
    parts = [p for p in url.rstrip("/").split("/") if p]
    if parts:
        slug = re.sub(r"[^a-z0-9\-]", "-", parts[-1].lower())
        slug = re.sub(r"-+", "-", slug).strip("-")
        if slug:
            return slug
    name     = row.get("name", f"assessment-{row.name}")
    fallback = re.sub(r"[^a-z0-9\-]", "-", name.lower())
    fallback = re.sub(r"-+", "-", fallback).strip("-")
    return fallback[:100]


def build_embedding_text(row: pd.Series) -> str:
    parts = [f"Assessment Name: {row['name']}"]
    if row.get("test_type_labels"):
        parts.append(f"Test Type: {row['test_type_labels']}")
    if row.get("job_levels"):
        parts.append(f"Job Levels: {row['job_levels']}")
    if row.get("languages"):
        parts.append(f"Languages: {row['languages']}")
    if row.get("description"):
        parts.append(f"Description: {row['description']}")
    return "\n".join(parts)


def build_metadata(row: pd.Series) -> dict:
    try:
        duration = int(float(row.get("duration_minutes", 0) or 0))
    except (ValueError, TypeError):
        duration = 0

    def to_bool_str(val) -> str:
        if isinstance(val, bool):
            return str(val)
        if isinstance(val, str):
            return str(val.strip().lower() in ("true", "yes", "1"))
        return str(bool(val))

    return {
        "name":             str(row.get("name", "")),
        "url":              str(row.get("url", "")),
        "test_type_codes":  str(row.get("test_type_codes", "")),
        "test_type_labels": str(row.get("test_type_labels", "")),
        "job_levels":       str(row.get("job_levels", "")),
        "languages":        str(row.get("languages", "")),
        "duration_minutes": duration,
        "remote_testing":   to_bool_str(row.get("remote_testing", False)),
        "adaptive_irt":     to_bool_str(row.get("adaptive_irt", False)),
    }


def get_embedding(text: str, retries: int = 3) -> list[float]:
    for attempt in range(1, retries + 1):
        try:
            result = gemini_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=f"task: search result | query: {text}",
            )
            return list(result.embeddings[0].values)
        except Exception as exc:
            log.warning("Attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                wait = DELAY_SECONDS * attempt
                log.info("Retrying in %ds ...", wait)
                time.sleep(wait)
    raise RuntimeError(f"Embedding failed after {retries} attempts.")



def phase1_generate_embeddings(df: pd.DataFrame) -> list[dict]:
    log.info("=" * 65)
    log.info("  PHASE 1 - Generating Embeddings")
    log.info("=" * 65)

    checkpoint: dict[str, dict] = {}
    if os.path.exists(EMBEDDINGS_JSON):
        with open(EMBEDDINGS_JSON, "r", encoding="utf-8") as f:
            saved = json.load(f)
        checkpoint = {rec["id"]: rec for rec in saved}
        log.info("Checkpoint loaded - %d already embedded.", len(checkpoint))
    else:
        log.info("No checkpoint found - starting fresh.")

    total   = len(df)
    success = 0
    failed  = []

    for idx, row in df.iterrows():
        chroma_id      = build_chroma_id(row)
        embedding_text = build_embedding_text(row)
        metadata       = build_metadata(row)

        if chroma_id in checkpoint:
            log.info("[%d/%d] SKIP (checkpoint)  %s", idx + 1, total, row["name"])
            success += 1
            continue

        log.info("[%d/%d] Embedding: %s", idx + 1, total, row["name"])

        try:
            embedding = get_embedding(embedding_text)

            checkpoint[chroma_id] = {
                "id":        chroma_id,
                "text":      embedding_text,
                "metadata":  metadata,
                "embedding": embedding,
            }

            with open(EMBEDDINGS_JSON, "w", encoding="utf-8") as f:
                json.dump(list(checkpoint.values()), f)

            success += 1
            log.info("  OK  embedded and saved (%d/%d)", success, total)

        except RuntimeError as exc:
            log.error("  FAIL  %s - %s", row["name"], exc)
            failed.append(row["name"])

        if idx < total - 1:
            log.info("  Waiting %ds ...", DELAY_SECONDS)
            time.sleep(DELAY_SECONDS)

    log.info("Phase 1 complete - %d embedded, %d failed.", success, len(failed))
    if failed:
        log.warning("Failed: %s", failed)

    return list(checkpoint.values())


def phase2_insert_chromadb(records: list[dict], reset: bool = False) -> None:
    log.info("=" * 65)
    log.info("  PHASE 2 - Bulk Insert into ChromaDB")
    log.info("=" * 65)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    existing_names = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME in existing_names and reset:
        log.warning("Deleting existing collection for fresh insert.")
        chroma_client.delete_collection(COLLECTION_NAME)

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    already_inserted = set()
    if collection.count() > 0:
        existing = collection.get(include=[])
        already_inserted = set(existing["ids"])
        log.info("%d already in ChromaDB - skipping.", len(already_inserted))

    new_records = [r for r in records if r["id"] not in already_inserted]
    log.info("%d new records to insert.", len(new_records))

    if not new_records:
        log.info("Nothing new to insert. ChromaDB is up to date.")
        log.info("Total in ChromaDB: %d", collection.count())
        return

    BATCH_SIZE = 377
    total_inserted = 0

    for batch_start in range(0, len(new_records), BATCH_SIZE):
        batch = new_records[batch_start: batch_start + BATCH_SIZE]

        ids        = [r["id"]        for r in batch]
        embeddings = [r["embedding"] for r in batch]
        documents  = [r["text"]      for r in batch]
        metadatas  = [r["metadata"]  for r in batch]

        log.info(
            "Inserting batch %d-%d of %d ...",
            batch_start + 1,
            batch_start + len(batch),
            len(new_records),
        )

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        total_inserted += len(batch)
        log.info("  OK  batch done. Total in DB: %d", collection.count())

    log.info("Phase 2 complete - %d records inserted.", total_inserted)
    log.info("ChromaDB final count: %d", collection.count())
    log.info("DB location: %s", CHROMA_DB_PATH)



def main():
    log.info("=" * 65)
    log.info("  SHL Embedding Pipeline (Two-Phase)")
    log.info("=" * 65)

    df = load_catalog(CSV_PATH)

    records = phase1_generate_embeddings(df)

    if not records:
        log.error("No embeddings generated. Check API key and internet connection.")
        sys.exit(1)

    log.info("Total embeddings ready: %d", len(records))
    phase2_insert_chromadb(records, reset=False)

    log.info("=" * 65)
    log.info("  ALL DONE")
    log.info("  Checkpoint file : %s", EMBEDDINGS_JSON)
    log.info("  ChromaDB folder : %s", CHROMA_DB_PATH)
    log.info("=" * 65)


if __name__ == "__main__":
    main()