"""ETL pipeline: scrape DarGlobal + Wasalt pages, chunk them, embed them, store them in ChromaDB.

Two steps, so the Docker build never depends on a headless browser or on the target sites
letting a datacenter IP through their bot protection:

  python scraper_indexer.py --scrape   # run locally: crawl pages -> data/pages.json (needs crawl4ai)
  python scraper_indexer.py            # index data/pages.json -> ./chroma_store (runs in Docker build)

If data/pages.json does not exist yet, the plain command scrapes first, then indexes.

DarGlobal (darglobal.co.uk) sits behind Imperva/Incapsula bot protection, which blocks automated
crawlers. We do not try to get around it. Instead, pages saved from a normal browser
(Ctrl+S -> "Webpage, HTML only") into data/manual/darglobal/ are run through the same crawl4ai
markdown pipeline.
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
PAGES_FILE = ROOT / "data" / "pages.json"
MANUAL_DIR = ROOT / "data" / "manual"
CHROMA_DIR = ROOT / "chroma_store"
COLLECTION = "real_estate"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# A small, hard-coded sample keeps the crawl polite and fast. Wasalt listing pages each carry
# ~15 listings with price, size and district. Its single-property pages answer automated
# browsers with a Cloudflare check, so we use listing pages only.
URLS = {
    "DarGlobal": [
        "https://darglobal.co.uk/",
        "https://darglobal.co.uk/projects",
        "https://darglobal.co.uk/one-of-one",
        "https://darglobal.co.uk/why-invest",
        "https://darglobal.co.uk/faq",
        "https://darglobal.co.uk/projects/properties-in-spain",
        "https://darglobal.co.uk/projects/properties-in-oman",
        "https://darglobal.co.uk/dg1",
        "https://darglobal.co.uk/trump-international-hotel-and-tower-dubai",
        "https://darglobal.co.uk/the-astera",
        "https://darglobal.co.uk/davinci-tower-by-pagani",
        "https://darglobal.co.uk/urban-oasis-by-missoni",
        "https://darglobal.co.uk/w-residences",
        "https://darglobal.co.uk/tierra-viva",
        "https://darglobal.co.uk/one-of-one/fendi-casa-azure-oceanfront-villas",
        "https://darglobal.co.uk/trump-tower-jeddah",
        "https://darglobal.co.uk/les-vagues",
        "https://darglobal.co.uk/marea",
    ],
    "Wasalt": [
        "https://wasalt.sa/en",
        "https://wasalt.sa/en/properties-for-sale-in-riyadh",
        "https://wasalt.sa/en/villas-for-sale-in-riyadh",
        "https://wasalt.sa/en/apartments-for-sale-in-riyadh",
        "https://wasalt.sa/en/lands-for-sale-in-riyadh",
        "https://wasalt.sa/en/properties-for-rent-in-riyadh",
        "https://wasalt.sa/en/apartments-for-rent-in-riyadh",
        "https://wasalt.sa/en/furnished-apartments-for-rent-in-riyadh",
        "https://wasalt.sa/en/properties-for-sale-in-jeddah",
        "https://wasalt.sa/en/properties-for-rent-in-jeddah",
        "https://wasalt.sa/en/properties-for-sale-in-makkah",
        "https://wasalt.sa/en/properties-for-sale-in-madinah",
        "https://wasalt.sa/en/properties-for-sale-in-dammam",
        "https://wasalt.sa/en/properties-for-rent-in-dammam",
    ],
}

# Text that means we got a bot-check page instead of content.
BLOCK_MARKERS = ("Performing security verification", "Incapsula", "Access Denied", "Just a moment")
MIN_CHARS = 400

# UI labels that carry no meaning for a chatbot (buttons, filter widgets).
NOISE_LINES = {
    "WhatsApp", "Call", "Apply", "ClearApply", "View Details", "Open app", "Wasalt app",
    "More Filters", "Budget", "Bedroom", "Get Matched", "Post Property",
}

CHUNK_CHARS = 800
OVERLAP_CHARS = 100


# ---------------------------------------------------------------- scraping (local only)

def _crawl_config():
    from crawl4ai import CacheMode, CrawlerRunConfig, DefaultMarkdownGenerator, PruningContentFilterLXML

    # PruningContentFilter scores DOM blocks by text density and drops boilerplate;
    # the result is exposed as markdown.fit_markdown.
    md = DefaultMarkdownGenerator(
        content_filter=PruningContentFilterLXML(threshold=0.45),
        options={"ignore_links": True, "ignore_images": True},
    )
    return CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=60_000,
        delay_before_return_html=3,  # let client-side rendering finish
        excluded_tags=["nav", "footer", "header", "script", "style", "form"],
        markdown_generator=md,
    )


def _page_record(source, url, result):
    """Turn a crawl4ai result into a stored page, or None if it is empty or a bot-check page."""
    if not result.success or not result.markdown:
        print(f"  skip {url}: {result.error_message or 'no content'}")
        return None
    text = result.markdown.fit_markdown or result.markdown.raw_markdown
    if len(text) < MIN_CHARS or any(m in text for m in BLOCK_MARKERS):
        print(f"  skip {url}: blocked or too little content ({len(text)} chars)")
        return None
    title = (result.metadata or {}).get("title") or url
    print(f"  ok   {url}: {len(text)} chars")
    return {
        "source": source,
        "url": url,
        "title": title.strip(),
        "markdown": text,
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _manual_pages():
    """Yield (source, url, html) for browser-saved pages in data/manual/<source>/*.html."""
    for path in sorted(MANUAL_DIR.glob("*/*.htm*")):
        source = {"darglobal": "DarGlobal", "wasalt": "Wasalt"}.get(path.parent.name.lower())
        if not source:
            continue
        html = path.read_text(encoding="utf-8", errors="ignore")
        # Chrome writes "<!-- saved from url=(0034)https://... -->" at the top of saved pages;
        # otherwise fall back to the page's canonical link.
        m = (re.search(r"saved from url=\(\d+\)(\S+?)\s*-->", html[:2000])
             or re.search(r'<link[^>]+rel="canonical"[^>]+href="([^"]+)"', html))
        url = m.group(1) if m else f"file://{path.name}"
        yield source, url, html


async def scrape(only=None, delay=10):
    from crawl4ai import AsyncWebCrawler, BrowserConfig

    # Start from the last snapshot, so a page that is blocked this run keeps its last good copy.
    pages = json.loads(PAGES_FILE.read_text(encoding="utf-8")) if PAGES_FILE.exists() else []
    cfg = _crawl_config()
    async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False)) as crawler:
        for source, urls in URLS.items():
            if only and source != only:
                continue
            print(f"Crawling {source} ({len(urls)} URLs)")
            for url in urls:
                try:
                    result = await crawler.arun(url, config=cfg)
                except Exception as exc:  # one bad page must not stop the run
                    print(f"  skip {url}: {exc}")
                    continue
                if rec := _page_record(source, url, result):
                    pages.append(rec)
                await asyncio.sleep(delay)  # be polite: one page every `delay` seconds

        manual = list(_manual_pages())
        if manual:
            print(f"Converting {len(manual)} browser-saved page(s)")
        for source, url, html in manual:
            # "raw:" tells crawl4ai to process the given HTML instead of fetching a URL.
            result = await crawler.arun("raw:" + html, config=cfg)
            if rec := _page_record(source, url, result):
                pages.append(rec)

    # De-duplicate by URL, keeping the newest copy.
    pages = list({p["url"]: p for p in pages}.values())
    PAGES_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAGES_FILE.write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = {s: sum(p["source"] == s for p in pages) for s in URLS}
    print(f"Saved {len(pages)} pages to {PAGES_FILE} {counts}")


# ---------------------------------------------------------------- chunking + indexing

def clean(markdown, source):
    lines = []
    for line in markdown.splitlines():
        s = line.strip()
        if not s or s in NOISE_LINES or s.lstrip("*- ").isdigit():
            continue
        lines.append(s)
    text = "\n".join(lines)
    if source == "DarGlobal":
        # DarGlobal project grids print the location ABOVE the project name
        # ("Dubai, UAE\n### DG1"). Move it into the heading so it stays with its project.
        text = re.sub(r"^([A-Z][^#*\n.]{1,50}[,|][^#*\n.]{1,30})\n(#{2,4} )(.+)$", r"\2\3 (\1)", text, flags=re.M)
    if source == "Wasalt":
        # A Wasalt listing card is "### 30,000 / yearly / ## Apartment 175 SQM / ###### Al Nuzha, Jeddah".
        # Join it into one line so a chunk boundary can never split a price from its property.
        text = re.sub(r"^#{1,6} ?([\d,]{3,})\n(?:(yearly|monthly|weekly|daily)\n)?## (.+)(?:\n#{6} (.+))?$",
                      lambda m: f"## {m[3]} | {m[4] or 'location not shown'} | Price: {m[1]}"
                                + (f" ({m[2]})" if m[2] else ""),
                      text, flags=re.M)
    return text


def chunk(text, size=CHUNK_CHARS, overlap=OVERLAP_CHARS):
    """Split on markdown headings first (one Wasalt listing or one DarGlobal section per block),
    then pack blocks into chunks of about `size` characters. A block longer than `size`
    is cut into windows that overlap by `overlap` characters."""
    blocks = re.split(r"\n(?=#{1,6} )", text)
    chunks, current = [], ""
    for block in blocks:
        if len(block) > size:
            if current:
                chunks.append(current)
                current = ""
            step = size - overlap
            chunks.extend(block[i:i + size] for i in range(0, len(block), step))
        elif len(current) + len(block) + 1 > size:
            chunks.append(current)
            current = block
        else:
            current = f"{current}\n{block}" if current else block
    if current:
        chunks.append(current)
    return [c.strip() for c in chunks if len(c.strip()) > 40]


def metadata_for(page):
    """Metadata used for filtering at query time. Chroma metadata values cannot be None."""
    meta = {"source": page["source"], "url": page["url"], "title": page["title"][:200]}
    m = re.search(r"/(?:(\w[\w-]*?)-)?for-(sale|rent)-in-([a-z-]+)$", page["url"])
    if m:
        meta["property_type"] = (m.group(1) or "properties").replace("-", " ")
        meta["listing"] = m.group(2)
        meta["city"] = m.group(3).replace("-", " ").title()
    return meta


def embedder():
    """all-MiniLM-L6-v2 run through ONNX Runtime (bundled with chromadb). Same weights, pooling and
    normalisation as the sentence-transformers version, without PyTorch, so the app fits in 512 MB."""
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    return ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])


def build_index():
    import chromadb

    pages = json.loads(PAGES_FILE.read_text(encoding="utf-8"))
    ids, docs, metas = [], [], []
    for p_i, page in enumerate(pages):
        meta = metadata_for(page)
        for c_i, text in enumerate(chunk(clean(page["markdown"], page["source"]))):
            # Prefix each chunk with where it came from so the embedding and the LLM both see it.
            header = f"[{page['source']}] {meta['title']}"
            ids.append(f"p{p_i}-c{c_i}")
            docs.append(f"{header}\n{text}")
            # has_listings marks chunks holding Wasalt listing cards (merged in clean()),
            # so the app can fetch a city's listings by metadata, not only by similarity.
            metas.append({**meta, "has_listings": "| Price:" in text})
    if not docs:
        sys.exit("No chunks to index: data/pages.json is empty.")

    print(f"Embedding {len(docs)} chunks from {len(pages)} pages with {EMBED_MODEL} (ONNX)")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)  # rebuild from scratch so removed pages disappear
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=embedder())
    for i in range(0, len(docs), 100):  # Chroma embeds the documents on upsert
        col.upsert(ids=ids[i:i + 100], documents=docs[i:i + 100], metadatas=metas[i:i + 100])
    counts = {s: sum(m["source"] == s for m in metas) for s in URLS}
    print(f"Indexed {col.count()} chunks into {CHROMA_DIR} {counts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scrape", action="store_true", help="crawl the sites and rewrite data/pages.json first")
    ap.add_argument("--only", choices=list(URLS), help="with --scrape: crawl only this source")
    ap.add_argument("--delay", type=int, default=10, help="with --scrape: seconds between pages (default 10)")
    args = ap.parse_args()
    if args.scrape or not PAGES_FILE.exists():
        asyncio.run(scrape(args.only, args.delay))
    build_index()
