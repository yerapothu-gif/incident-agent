import os
import json
from pathlib import Path
from dotenv import load_dotenv
import psycopg
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    task_type="RETRIEVAL_DOCUMENT",
    output_dimensionality=768
)

knowledge_base = [
    {
        "content": "Auth Service 504 Gateway Timeouts: Triggered by Redis cache evictions or token refresh lock contention. Remediation: Flush expired token blacklist in Redis or restart auth pod replicas.",
        "metadata": {"service": "auth", "category": "runbook"}
    },
    {
        "content": "Database High Latency: When replica CPU exceeds 80%, kill idle transactions and inspect query locks using pg_stat_activity before failing over.",
        "metadata": {"service": "database", "category": "runbook"}
    },
    {
        "content": "Payment Gateway Glitches: Verify Stripe webhook idempotency keys. If failure persists beyond 5 minutes, reroute traffic to the Adyen backup provider.",
        "metadata": {"service": "payments", "category": "runbook"}
    }
]


def seed():
    print("Generating Gemini embeddings (768 dimensions)...")
    texts = [doc["content"] for doc in knowledge_base]
    vectors = embeddings.embed_documents(texts)

    db_uri = os.getenv("DATABASE_URL")
    print("Connecting directly to Supabase Postgres pooler...")

    with psycopg.connect(db_uri, connect_timeout=15) as conn:
        with conn.cursor() as cur:
            for doc, vec in zip(knowledge_base, vectors):
                vec_str = f"[{','.join(str(x) for x in vec)}]"
                cur.execute(
                    """
                    INSERT INTO incident_docs (content, metadata, embedding)
                    VALUES (%s, %s, %s::vector)
                    """,
                    (doc["content"], json.dumps(doc["metadata"]), vec_str)
                )
        conn.commit()

    print(f"Success! Seeded {len(knowledge_base)} documents into Supabase pgvector.")


if __name__ == "__main__":
    seed()