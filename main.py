import os
import re
import time
import json
import logging
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# ---------------- LOGGING ----------------

os.makedirs("/var/log/progresskb", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/var/log/progresskb/app.log"),
        logging.StreamHandler()  # also print to console
    ]
)
logger = logging.getLogger(__name__)

# ---------------- CONFIG ----------------

PERSIST_DIR   = "/home/progress/progresskb/chroma_db"
ARTICLE_DIR   = "/home/progress/progresskb/chroma_articles"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MAX_CONTEXT_TOKENS = 6000
TOP_ARTICLES       = 8
BATCH_SIZE         = 2000
DEDUP_THRESHOLD    = 0.92
BASE_URL           = "https://progress.my.salesforce-sites.com/ProgressKB/articles/Article/"

# ---------------- OPENROUTER CLIENT ----------------

# ── DUAL CLIENT SETUP ──────────────────────────────────────
# Cloud client (OpenRouter)
cloud_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Local client (llama-cpp-python server)
local_client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="local"
)

LOCAL_MODELS = ["gguf", "local", "llama", "mistral-local", "qwen"]

def get_client(model_name: str):
    """Return local or cloud client based on model name."""
    if any(k in model_name.lower() for k in LOCAL_MODELS):
        logger.info(f"Using LOCAL LLM client for model: {model_name}")
        return local_client
    logger.info(f"Using CLOUD LLM client for model: {model_name}")
    return cloud_client

# ---------------- FASTAPI APP ----------------

app = FastAPI(
    title="Progress KB – Advanced RAG",
    description="Production RAG API for Progress Knowledge Base",
    version="1.0.0"
)

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- REQUEST MODELS ----------------

class QueryRequest(BaseModel):
    question: str
    product_group: Optional[str] = "All Products"
    model_name: Optional[str] = "deepseek-r1-7b-q4.gguf"
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 1500
    vector_top_k: Optional[int] = 40
    bm25_k: Optional[int] = 15
    rerank_limit: Optional[int] = 15

# ---------------- LOAD MODELS (once at startup) ----------------

logger.info("Loading embedding model...")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

logger.info("Loading vector stores...")
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings_model,
    collection_name="progress_kb"
)
article_store = Chroma(
    persist_directory=ARTICLE_DIR,
    embedding_function=embeddings_model,
    collection_name="progress_articles"
)

logger.info("Loading reranker model...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

logger.info("Loading product groups...")
def load_groups():
    groups = set()
    all_ids = vectorstore._collection.get(include=[])["ids"]
    for batch_start in range(0, len(all_ids), BATCH_SIZE):
        batch_ids = all_ids[batch_start: batch_start + BATCH_SIZE]
        results = vectorstore._collection.get(ids=batch_ids, include=["metadatas"])
        for m in results["metadatas"]:
            if m.get("group"):
                groups.add(m["group"])
    groups = sorted(list(groups))
    groups.insert(0, "All Products")
    return groups

PRODUCT_GROUPS = load_groups()
logger.info(f"Loaded {len(PRODUCT_GROUPS)} product groups")
logger.info(f"Chunk store: {vectorstore._collection.count():,} chunks")
logger.info(f"Article store: {article_store._collection.count():,} articles")
logger.info("✅ All models loaded — ready to serve requests")

# ---------------- PROMPT ----------------

prompt = ChatPromptTemplate.from_template(
"""You are a senior Progress technical support engineer.

Use ONLY the KB context below.

Rules:
- Do not invent commands
- Cite KB article IDs inline using the format (Article <ID>) immediately after any claim derived from that article
- If information is missing say "KB does not contain enough info"

Context:
{context}

Question:
{question}

Answer directly without any preamble:"""
)

# ---------------- RAG FUNCTIONS ----------------

def call_llm(prompt_text, model_name, temperature, max_tokens):
    """Non-streaming LLM call — used for query expansion."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


def call_llm(prompt_text, model_name, temperature, max_tokens):
    """Non-streaming LLM call — used for query expansion."""
    c = get_client(model_name)
    response = c.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


def call_llm_streaming(prompt_text, model_name, temperature, max_tokens):
    """Stream the LLM response — strips DeepSeek R1 thinking tokens."""
    c = get_client(model_name)
    stream = c.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )

    inside_think = False
    think_buffer = ""

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if not delta or not delta.content:
            continue

        token = delta.content
        think_buffer += token

        if "<think>" in think_buffer:
            inside_think = True
            before = think_buffer.split("<think>")[0]
            if before.strip():
                yield before
            think_buffer = ""
            continue

        if inside_think and "</think>" in think_buffer:
            inside_think = False
            after = think_buffer.split("</think>")[-1]
            think_buffer = ""
            if after.strip():
                yield after
            continue

        if not inside_think and len(think_buffer) > 10:
            yield think_buffer
            think_buffer = ""

    if think_buffer and not inside_think:
        yield think_buffer

def expand_queries(question, model_name, temperature):
    prompt_text = f"""Generate 4 search queries for troubleshooting technical issues.
Return each query on a new line with no numbering, bullets, or reasoning.
Do not explain. Just output the queries directly.

Question:
{question}

Return each query on a new line with no numbering or bullets."""

    response = call_llm(prompt_text, model_name, temperature, max_tokens=300)
    queries = []
    for line in response.split("\n"):
        q = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
        if len(q) > 5:
            queries.append(q)
    queries.insert(0, question)
    return list(set(queries))


def cosine_similarity(a, b):
    """Compute cosine similarity between two numpy vectors."""
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def semantic_dedup(docs, threshold=DEDUP_THRESHOLD):
    """Remove semantically duplicate chunks using cosine similarity."""
    if not docs:
        return docs
    texts = [d.page_content for d in docs]
    embeddings = embeddings_model.embed_documents(texts)
    kept_indices = []
    kept_embeddings = []
    for i, emb in enumerate(embeddings):
        is_duplicate = any(
            cosine_similarity(emb, kept_emb) >= threshold
            for kept_emb in kept_embeddings
        )
        if not is_duplicate:
            kept_indices.append(i)
            kept_embeddings.append(emb)
    return [docs[i] for i in kept_indices]


def rerank_articles(question, article_hits):
    """Rerank candidate articles using CrossEncoder before chunk retrieval."""
    if not article_hits:
        return []
    seen = {}
    for doc in article_hits:
        aid = doc.metadata.get("article_id")
        if aid and aid not in seen:
            seen[aid] = doc
    unique_articles = list(seen.values())
    pairs = [(question, doc.page_content) for doc in unique_articles]
    scores = reranker.predict(pairs)
    scored = sorted(zip(unique_articles, scores), key=lambda x: x[1], reverse=True)
    return [doc.metadata["article_id"] for doc, _ in scored]


def run_rag_pipeline(req: QueryRequest):
    """
    Full RAG pipeline — returns (selected_docs, context_text, prompt_text, trace_log)
    """
    trace_log = []
    start = time.time()

    # -------- QUERY EXPANSION --------
    queries = expand_queries(req.question, req.model_name, req.temperature)
    trace_log.append(f"Queries used: {queries}")

    # -------- ARTICLE SEARCH --------
    article_hits = []
    for q in queries:
        if req.product_group == "All Products":
            retriever = article_store.as_retriever(
                search_kwargs={"k": TOP_ARTICLES}
            )
        else:
            retriever = article_store.as_retriever(
                search_kwargs={"k": TOP_ARTICLES, "filter": {"group": req.product_group}}
            )
        docs = retriever.invoke(q)
        trace_log.append(f"Query '{q}' → {len(docs)} articles")
        article_hits.extend(docs)

    # -------- ARTICLE-LEVEL RERANKING --------
    article_ids = rerank_articles(req.question, article_hits)
    trace_log.append(f"Articles after rerank ({len(article_ids)}): {article_ids}")

    # -------- CHUNK RETRIEVAL --------
    filtered_docs = []
    for aid in article_ids:
        results = vectorstore._collection.get(
            where={"article_id": str(aid)},
            include=["documents", "metadatas"]
        )
        for text, meta in zip(results["documents"], results["metadatas"]):
            filtered_docs.append(Document(page_content=text, metadata=meta))

    trace_log.append(f"Chunks from article filter: {len(filtered_docs)}")

    # -------- FALLBACK 1: similarity search with product filter --------
    if len(filtered_docs) == 0:
        trace_log.append("⚠️ Fallback 1: similarity search with product filter")
        search_kwargs = {"k": req.vector_top_k}
        if req.product_group != "All Products":
            search_kwargs["filter"] = {"group": req.product_group}
        filtered_docs = vectorstore.similarity_search(req.question, **search_kwargs)
        trace_log.append(f"Fallback 1 → {len(filtered_docs)} chunks")

    # -------- FALLBACK 2: no product filter --------
    if len(filtered_docs) == 0 and req.product_group != "All Products":
        trace_log.append("⚠️ Fallback 2: ignoring product group filter")
        filtered_docs = vectorstore.similarity_search(req.question, k=req.vector_top_k)
        trace_log.append(f"Fallback 2 → {len(filtered_docs)} chunks")

    if len(filtered_docs) == 0:
        raise HTTPException(status_code=404, detail="No documents found in KB.")

    trace_log.append(f"Total chunks before dedup/rerank: {len(filtered_docs)}")

    # -------- BM25 --------
    bm25 = BM25Retriever.from_documents(filtered_docs)
    bm25.k = req.bm25_k
    bm25_docs = bm25.invoke(req.question)
    docs = list({d.page_content: d for d in (filtered_docs + bm25_docs)}.values())

    # -------- SEMANTIC DEDUP --------
    before = len(docs)
    docs = semantic_dedup(docs, threshold=DEDUP_THRESHOLD)
    trace_log.append(f"Semantic dedup: {before} → {len(docs)} chunks (removed {before - len(docs)})")

    # -------- CHUNK RERANK --------
    pairs = [(req.question, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:req.rerank_limit]

    # -------- DYNAMIC CONTEXT --------
    approx_tokens = 0
    selected_docs = []
    for doc, score in scored_docs:
        tokens = len(doc.page_content) // 6
        if approx_tokens + tokens > MAX_CONTEXT_TOKENS:
            break
        selected_docs.append(doc)
        approx_tokens += tokens

    trace_log.append(f"Chunks sent to LLM: {len(selected_docs)}")
    trace_log.append(f"Approx tokens used: {approx_tokens}")

    # -------- CONTEXT BUILD --------
    context_blocks = []
    for doc in selected_docs:
        aid   = doc.metadata.get("article_id")
        title = doc.metadata.get("title")
        group = doc.metadata.get("group")
        context_blocks.append(
            f"KB Article: {aid}\nTitle: {title}\nGroup: {group}\n\n{doc.page_content}"
        )
    context_text = "\n\n---\n\n".join(context_blocks)

    # -------- BUILD FINAL PROMPT --------
    final_prompt = prompt.invoke({"context": context_text, "question": req.question})
    prompt_text  = final_prompt.to_string()

    elapsed = round(time.time() - start, 2)
    trace_log.append(f"Pipeline time (excl. LLM): {elapsed}s")

    return selected_docs, prompt_text, trace_log


# ---------------- API ENDPOINTS ----------------

@app.get("/health")
def health():
    """Health check endpoint — used by Nginx and monitoring."""
    return {
        "status": "ok",
        "chunks": vectorstore._collection.count(),
        "articles": article_store._collection.count(),
        "groups": len(PRODUCT_GROUPS)
    }


@app.get("/groups")
def get_groups():
    """Return list of product groups for the frontend dropdown."""
    return {"groups": PRODUCT_GROUPS}


@app.post("/query")
def query(req: QueryRequest):
    """
    Main RAG endpoint — streams the answer token by token.
    Frontend reads this as a Server-Sent Events stream.
    """
    logger.info(f"Query: '{req.question}' | Group: {req.product_group} | Model: {req.model_name}")
    start = time.time()

    try:
        selected_docs, prompt_text, trace_log = run_rag_pipeline(req)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Build sources list
    sources = []
    seen = set()
    for doc in selected_docs:
        aid   = doc.metadata.get("article_id", "")
        title = doc.metadata.get("title", "Untitled")
        if aid and aid not in seen:
            seen.add(aid)
            sources.append({
                "article_id": aid,
                "title": title,
                "url": f"{BASE_URL}{aid}"
            })

    def stream_response():
        """Generator that streams answer + metadata as SSE."""
        full_answer = ""
        try:
            for token in call_llm_streaming(
                prompt_text,
                req.model_name,
                req.temperature,
                req.max_tokens
            ):
                full_answer += token
                # Send each token as SSE
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        # After streaming — send sources and trace
        elapsed = round(time.time() - start, 2)
        logger.info(f"Query completed in {elapsed}s | Sources: {len(sources)}")

        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        yield f"data: {json.dumps({'type': 'trace', 'trace': trace_log})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'elapsed': elapsed})}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # tell Nginx not to buffer stream
        }
    )


@app.post("/rebuild-index")
def rebuild_index():
    """Trigger article index rebuild — admin use only."""
    logger.info("Article index rebuild triggered via API")
    try:
        # Clear existing
        all_ids = article_store._collection.get(include=[])["ids"]
        for i in range(0, len(all_ids), 500):
            article_store._collection.delete(ids=all_ids[i:i + 500])

        # Rebuild
        all_ids = vectorstore._collection.get(include=[])["ids"]
        total_ids = len(all_ids)
        article_map = {}

        for batch_start in range(0, total_ids, BATCH_SIZE):
            batch_ids = all_ids[batch_start: batch_start + BATCH_SIZE]
            results = vectorstore._collection.get(
                ids=batch_ids,
                include=["documents", "metadatas"]
            )
            for text, meta in zip(results["documents"], results["metadatas"]):
                aid = meta.get("article_id")
                if not aid:
                    continue
                if aid not in article_map:
                    article_map[aid] = {
                        "title": meta.get("title", ""),
                        "group": meta.get("group", ""),
                        "content": []
                    }
                if sum(len(c) for c in article_map[aid]["content"]) < 1500:
                    article_map[aid]["content"].append(text)

        ids, docs, metas = [], [], []
        for aid, data in article_map.items():
            ids.append(str(aid))
            docs.append(" ".join(data["content"])[:1500])
            metas.append({
                "article_id": str(aid),
                "title": data["title"],
                "group": data["group"]
            })

        INSERT_BATCH = 500
        for i in range(0, len(ids), INSERT_BATCH):
            article_store._collection.add(
                ids=ids[i:i + INSERT_BATCH],
                documents=docs[i:i + INSERT_BATCH],
                metadatas=metas[i:i + INSERT_BATCH]
            )

        logger.info(f"Article index rebuilt: {len(ids):,} articles")
        return {"status": "ok", "articles_indexed": len(ids)}

    except Exception as e:
        logger.error(f"Rebuild error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
