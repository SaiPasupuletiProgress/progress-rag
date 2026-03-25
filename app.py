import streamlit as st
import os
import re
import time
import numpy as np

from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# ---------------- CONFIG ----------------

PERSIST_DIR = "./chroma_db"
ARTICLE_DIR = "./chroma_articles"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MAX_CONTEXT_TOKENS = 6000
TOP_ARTICLES = 8
BATCH_SIZE = 2000

# Semantic dedup cosine similarity threshold (higher = stricter dedup)
DEDUP_THRESHOLD = 0.92

# Base URL for KB articles
BASE_URL = "https://progress.my.salesforce-sites.com/ProgressKB/articles/Article/"

# ---------------- OPENROUTER ----------------

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# ---------------- PAGE ----------------

st.set_page_config(page_title="Progress KB – Advanced RAG")
st.title("Progress Knowledge Base – Advanced RAG")

# ---------------- TRACE PANEL ----------------

trace_panel = st.expander("🔍 Retrieval Trace", expanded=False)

def trace(msg):
    trace_panel.write(msg)

# ---------------- SIDEBAR ----------------

with st.sidebar:

    temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
    max_tokens = st.slider("Max Tokens", 200, 1200, 800)

    vector_top_k = st.slider("Vector Top-K", 10, 60, 40)
    bm25_k = st.slider("BM25 Top-K", 5, 20, 15)
    rerank_limit = st.slider("Reranker Limit", 5, 25, 15)

    model_name = st.selectbox(
        "LLM Model",
        [
            "deepseek/deepseek-r1",
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "mistralai/mistral-large",
            "openai/gpt-oss-120b:free"
        ]
    )

    st.markdown("---")

    # ---- Article store rebuild button ----
    st.markdown("**🗄️ Article Index**")
    if st.button("🔄 Rebuild Article Index"):
        st.session_state["rebuild_article_index"] = True

# ---------------- VECTOR STORES ----------------

os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(ARTICLE_DIR, exist_ok=True)

@st.cache_resource
def load_vectorstores():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    chunk_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="progress_kb"
    )
    article_store = Chroma(
        persist_directory=ARTICLE_DIR,
        embedding_function=embeddings,
        collection_name="progress_articles"
    )
    return chunk_store, article_store, embeddings


vectorstore, article_store, embeddings_model = load_vectorstores()

# ---------------- BUILD ARTICLE INDEX ----------------

def clear_article_store():
    """Delete all documents from article store in safe batches."""
    all_ids = article_store._collection.get(include=[])["ids"]
    if not all_ids:
        return
    for i in range(0, len(all_ids), 500):
        article_store._collection.delete(ids=all_ids[i:i + 500])


def build_article_index(force=False):

    chunk_total = vectorstore._collection.count()
    article_total = article_store._collection.count()

    if chunk_total == 0:
        st.error("❌ Chunk store is empty.")
        return

    # If article store already has a healthy number of articles and not forcing rebuild, skip
    if not force and article_total > 50000:
        return

    # Clear existing article store before rebuild
    if article_total > 0:
        with st.spinner("Clearing old article index..."):
            clear_article_store()

    st.write(f"⚙️ Building article index from {chunk_total:,} chunks...")
    progress_bar = st.progress(0)

    # Step 1: Fetch all IDs (lightweight — no documents/embeddings)
    all_ids = vectorstore._collection.get(include=[])["ids"]
    total_ids = len(all_ids)

    article_map = {}

    # Step 2: Fetch in batches of BATCH_SIZE
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

            # Cap content per article to avoid huge strings
            if sum(len(c) for c in article_map[aid]["content"]) < 1500:
                article_map[aid]["content"].append(text)

        progress_bar.progress(min((batch_start + BATCH_SIZE) / total_ids, 1.0))

    progress_bar.empty()

    if not article_map:
        st.error("❌ No article_id found in metadata. Check chunk store.")
        return

    # Step 3: Insert into article store in batches
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

    st.success(f"✅ Article index built: {len(ids):,} articles")


# Handle rebuild button trigger from sidebar
force_rebuild = st.session_state.get("rebuild_article_index", False)
if force_rebuild:
    st.session_state["rebuild_article_index"] = False
build_article_index(force=force_rebuild)

st.write(f"Article store size: **{article_store._collection.count():,}**")
st.write(f"Total KB chunks indexed: **{vectorstore._collection.count():,}**")

# ---------------- PRODUCT GROUPS ----------------

@st.cache_data
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


groups = load_groups()
product_group = st.selectbox("Select Product Group", groups)

# ---------------- RERANKER ----------------

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

reranker = load_reranker()

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

Answer:"""
)

# ---------------- LLM CALL (non-streaming) ----------------

def call_llm(prompt_text):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

# ---------------- LLM CALL (streaming) ----------------

def call_llm_streaming(prompt_text):
    """Stream the LLM response token by token, yielding text chunks."""
    stream = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )
    for chunk in stream:
        # Guard against empty choices (e.g. final [DONE] chunk from OpenRouter)
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content

# ---------------- QUERY EXPANSION ----------------

def expand_queries(question):
    prompt_text = f"""Generate 4 search queries for troubleshooting technical issues.

Question:
{question}

Return each query on a new line with no numbering or bullets."""

    response = call_llm(prompt_text)

    queries = []
    for line in response.split("\n"):
        q = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
        if len(q) > 5:
            queries.append(q)

    queries.insert(0, question)
    return list(set(queries))

# ---------------- SEMANTIC DUPLICATE REMOVAL ----------------

def cosine_similarity(a, b):
    """Compute cosine similarity between two numpy vectors."""
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def semantic_dedup(docs, threshold=DEDUP_THRESHOLD):
    """
    Remove semantically duplicate chunks using cosine similarity on embeddings.
    Keeps the first occurrence when two chunks exceed the similarity threshold.
    """
    if not docs:
        return docs

    texts = [d.page_content for d in docs]
    embeddings = embeddings_model.embed_documents(texts)

    kept_indices = []
    kept_embeddings = []

    for i, emb in enumerate(embeddings):
        is_duplicate = False
        for kept_emb in kept_embeddings:
            if cosine_similarity(emb, kept_emb) >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept_indices.append(i)
            kept_embeddings.append(emb)

    return [docs[i] for i in kept_indices]

# ---------------- ARTICLE-LEVEL RERANKING ----------------

def rerank_articles(question, article_hits):
    """
    Rerank candidate articles using the cross-encoder before chunk retrieval.
    Each article is scored by its representative summary text vs. the question.
    Returns article_ids sorted by descending relevance score.
    """
    if not article_hits:
        return []

    # Deduplicate articles — keep one doc per article_id
    seen = {}
    for doc in article_hits:
        aid = doc.metadata.get("article_id")
        if aid and aid not in seen:
            seen[aid] = doc

    unique_articles = list(seen.values())

    pairs = [(question, doc.page_content) for doc in unique_articles]
    scores = reranker.predict(pairs)

    scored = sorted(zip(unique_articles, scores), key=lambda x: x[1], reverse=True)

    trace("**Article-level rerank scores (top 10):**")
    for doc, score in scored[:10]:
        aid = doc.metadata.get("article_id", "?")
        title = doc.metadata.get("title", "")
        trace(f"  [{score:.3f}] {aid} — {title}")

    return [doc.metadata["article_id"] for doc, _ in scored]

# ---------------- USER QUERY ----------------

question = st.text_input("Ask KB Question")

if question:

    start = time.time()

    # -------- QUERY EXPANSION --------

    with st.spinner("Expanding queries..."):
        queries = expand_queries(question)

    trace("**Queries used:**")
    trace(queries)

    # -------- ARTICLE SEARCH --------

    article_hits = []

    with st.spinner("Searching article index..."):
        for q in queries:
            if product_group == "All Products":
                retriever = article_store.as_retriever(
                    search_kwargs={"k": TOP_ARTICLES}
                )
            else:
                retriever = article_store.as_retriever(
                    search_kwargs={
                        "k": TOP_ARTICLES,
                        "filter": {"group": product_group}
                    }
                )
            docs = retriever.invoke(q)
            trace(f"Query '{q}' → {len(docs)} articles")
            article_hits.extend(docs)

    # -------- ARTICLE-LEVEL RERANKING --------

    with st.spinner("Reranking articles..."):
        article_ids = rerank_articles(question, article_hits)

    trace(f"**Articles selected after rerank ({len(article_ids)}):** {article_ids}")

    # -------- CHUNK RETRIEVAL --------

    filtered_docs = []

    with st.spinner(f"Fetching chunks from {len(article_ids)} articles..."):
        for aid in article_ids:
            results = vectorstore._collection.get(
                where={"article_id": str(aid)},
                include=["documents", "metadatas"]
            )
            for text, meta in zip(results["documents"], results["metadatas"]):
                filtered_docs.append(Document(page_content=text, metadata=meta))

    trace(f"Chunks from article filter: {len(filtered_docs)}")

    # -------- FALLBACK 1: similarity search with product filter --------

    if len(filtered_docs) == 0:
        trace("⚠️ Fallback 1: similarity search with product filter")
        search_kwargs = {"k": vector_top_k}
        if product_group != "All Products":
            search_kwargs["filter"] = {"group": product_group}
        filtered_docs = vectorstore.similarity_search(question, **search_kwargs)
        trace(f"Fallback 1 → {len(filtered_docs)} chunks")

    # -------- FALLBACK 2: no product filter --------

    if len(filtered_docs) == 0 and product_group != "All Products":
        trace("⚠️ Fallback 2: ignoring product group filter")
        filtered_docs = vectorstore.similarity_search(question, k=vector_top_k)
        trace(f"Fallback 2 → {len(filtered_docs)} chunks")

    if len(filtered_docs) == 0:
        st.error("❌ No documents found even after fallback. Check Retrieval Trace.")
        st.stop()

    trace(f"**Total chunks before dedup/rerank: {len(filtered_docs)}**")

    # -------- BM25 --------

    bm25 = BM25Retriever.from_documents(filtered_docs)
    bm25.k = bm25_k
    bm25_docs = bm25.invoke(question)

    docs = list({d.page_content: d for d in (filtered_docs + bm25_docs)}.values())

    # -------- SEMANTIC DUPLICATE REMOVAL --------

    with st.spinner("Removing semantic duplicates..."):
        docs_before_dedup = len(docs)
        docs = semantic_dedup(docs, threshold=DEDUP_THRESHOLD)
        docs_after_dedup = len(docs)

    trace(f"**Semantic dedup:** {docs_before_dedup} → {docs_after_dedup} chunks (removed {docs_before_dedup - docs_after_dedup})")

    # -------- RERANK --------

    with st.spinner("Reranking chunks..."):
        pairs = [(question, d.page_content) for d in docs]
        scores = reranker.predict(pairs)
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:rerank_limit]

    # -------- DYNAMIC CONTEXT --------

    approx_tokens = 0
    selected_docs = []

    for doc, score in scored_docs:
        tokens = len(doc.page_content) // 6
        if approx_tokens + tokens > MAX_CONTEXT_TOKENS:
            break
        selected_docs.append(doc)
        approx_tokens += tokens

    trace(f"**Chunks sent to LLM: {len(selected_docs)}**")
    trace(f"Approx tokens used: {approx_tokens}")

    # -------- CONTEXT BUILD --------

    context_blocks = []
    for doc in selected_docs:
        aid = doc.metadata.get("article_id")
        title = doc.metadata.get("title")
        group = doc.metadata.get("group")
        context_blocks.append(
            f"KB Article: {aid}\nTitle: {title}\nGroup: {group}\n\n{doc.page_content}"
        )

    context_text = "\n\n---\n\n".join(context_blocks)

    # -------- LLM (STREAMING) --------

    with st.spinner("Generating answer..."):
        final_prompt = prompt.invoke({"context": context_text, "question": question})
        prompt_text = final_prompt.to_string()

    st.markdown("### Answer")
    answer_placeholder = st.empty()
    full_answer = ""

    # Stream tokens into the placeholder
    for token in call_llm_streaming(prompt_text):
        full_answer += token
        answer_placeholder.markdown(full_answer + "▌")  # blinking cursor effect

    # Final render without cursor
    answer_placeholder.markdown(full_answer)

    st.caption(f"⏱ Total time: {round(time.time() - start, 2)} sec")

    # -------- SOURCES (clean format: ArticleID <Title>) --------

    st.markdown("### Sources")
    shown = set()

    for doc in selected_docs:
        aid = doc.metadata.get("article_id", "")
        title = doc.metadata.get("title", "Untitled")
        if aid and aid not in shown:
            shown.add(aid)
            url = f"{BASE_URL}{aid}"
            st.markdown(f"🔗 [{aid} &lt;{title}&gt;]({url})")
