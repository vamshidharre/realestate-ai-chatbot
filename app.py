"""Streamlit RAG chatbot over scraped DarGlobal and Wasalt pages.

Flow for each question: embed it -> top-k search in ChromaDB (optionally filtered by source)
-> put the retrieved chunks in the system prompt -> stream the answer from a free OpenRouter model.
"""

import os

import chromadb
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from scraper_indexer import CHROMA_DIR, COLLECTION, embedder

load_dotenv()

# Free models change often on OpenRouter. The first is used; if it is rate-limited or down,
# OpenRouter falls through to the next ones (its "models" fallback list).
MODELS = [
    m.strip()
    for m in os.getenv(
        "OPENROUTER_MODELS",
        "google/gemma-4-31b-it:free,google/gemma-4-26b-a4b-it:free,nvidia/nemotron-3-ultra-550b-a55b:free",
    ).split(",")
    if m.strip()
]
TOP_K = 12  # ~2.5K tokens of context; long SEO text otherwise crowds out listings and project lists
HISTORY_TURNS = 6  # past messages sent to the model, to keep requests small

SYSTEM_PROMPT = """You are a helpful real estate assistant for two sources:
- DarGlobal: an international luxury developer (branded residences in the UAE, Oman, Qatar, Saudi Arabia, Spain, UK).
- Wasalt: a Saudi Arabian property portal with listings for sale and rent. Wasalt prices are in Saudi riyals (SAR).

Answer ONLY from the context below. If the context does not contain the answer, say you don't have
that information in the scraped data and suggest what the user could ask instead. Never invent
prices, sizes, dates or availability. Name the source (DarGlobal or Wasalt) for the facts you use.
Keep answers short and use bullet points for lists of properties.

Context:
{context}"""

st.set_page_config(page_title="Real Estate AI Chatbot", page_icon="🏠")


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # The collection embeds query text with the same ONNX model used at index time.
    return client.get_collection(COLLECTION, embedding_function=embedder())


@st.cache_resource
def load_llm():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None
    # OpenRouter speaks the OpenAI API, so the official SDK works with a different base_url.
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


@st.cache_resource
def known_cities():
    metas = load_collection().get(include=["metadatas"])["metadatas"]
    return sorted({m["city"] for m in metas if m.get("city")})


def retrieve(query, source):
    def search(where, k):
        res = load_collection().query(query_texts=[query], n_results=k, where=where)
        return list(zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]))

    hits = search(None if source == "All" else {"source": source}, TOP_K)
    # Metadata filtering: if the question names a Wasalt city, also pull that city's listing
    # chunks. Pure similarity ranks them below the same page's long SEO text.
    city = next((c for c in known_cities() if c.lower() in query.lower()), None)
    if city and source != "DarGlobal":
        seen = {h[0] for h in hits}
        listings = search({"$and": [{"city": city}, {"has_listings": True}]}, 8)
        hits += [h for h in listings if h[0] not in seen]
    return [{"text": doc, "meta": meta, "score": 1 - dist} for _, doc, meta, dist in hits]


def stream_answer(llm, messages):
    stream = llm.chat.completions.create(
        model=MODELS[0],
        messages=messages,
        stream=True,
        temperature=0.2,
        extra_body={"models": MODELS},
    )
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content


# ---------------------------------------------------------------- UI

with st.sidebar:
    st.header("Settings")
    source = st.radio("Search in", ["All", "DarGlobal", "Wasalt"], help="Filters retrieval by data source.")
    if st.button("Clear chat"):
        st.session_state.messages = []
    st.caption(f"Model: `{MODELS[0]}`")
    try:
        st.caption(f"Index: {load_collection().count()} chunks")
    except Exception as exc:
        st.error(f"Vector store not found. Run `python scraper_indexer.py` first. ({exc})")
        st.stop()

st.title("🏠 Real Estate AI Chatbot")
st.caption("Ask about DarGlobal's luxury projects or Wasalt listings in Saudi Arabia. "
           "Answers come only from pages scraped from darglobal.co.uk and wasalt.sa.")

llm = load_llm()
if llm is None:
    st.error("OPENROUTER_API_KEY is not set. Add it to `.env` locally, or as a Secret in the Hugging Face Space settings.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("**Try:** *Which DarGlobal projects are in Oman?* · *Apartments for rent in Jeddah under 30,000 a year?* · "
                "*What amenities does DG1 have?*")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about a property, project or city..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Short follow-ups like "and for rent?" need the previous question to retrieve the right
    # chunks; full questions are searched alone so an earlier topic does not swamp a new one.
    user_turns = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    query = " ".join(user_turns[-2:]) if len(question.split()) <= 5 else question
    hits = retrieve(query, source)
    context = "\n\n---\n\n".join(f"(source: {h['meta']['url']})\n{h['text']}" for h in hits)

    history = st.session_state.messages[-HISTORY_TURNS:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}, *history]

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_answer(llm, messages))
        except Exception as exc:
            answer = (f"Sorry, the language model request failed: `{exc}`. "
                      "If this is a rate limit (429), free models allow a few requests per minute; try again shortly.")
            st.error(answer)
        with st.expander("Sources"):
            for h in hits:
                st.markdown(f"- [{h['meta']['title']}]({h['meta']['url']}) · {h['meta']['source']} · similarity {h['score']:.2f}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
