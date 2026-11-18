# Real Estate AI Chatbot (DarGlobal + Wasalt)

A retrieval-augmented (RAG) chatbot that answers questions from public pages of
[DarGlobal](https://darglobal.co.uk) (international luxury developer) and
[Wasalt](https://wasalt.sa) (Saudi property portal), using a free LLM on OpenRouter.
It runs as a single Docker container.

**Live demo:** LIVE_URL

Try: *Which DarGlobal projects are in Oman?* · *Apartments for rent in Jeddah under 30,000 a year?* ·
*What amenities does DG1 have?*

## How it works

| Step | File | Tool |
|---|---|---|
| Scrape public pages to clean Markdown | `scraper_indexer.py --scrape` | crawl4ai (headless Chromium + PruningContentFilter) |
| Clean, chunk (~800 chars, split on headings), embed | `scraper_indexer.py` | `all-MiniLM-L6-v2` via ONNX Runtime (bundled with ChromaDB) |
| Store vectors + metadata (source, url, city, sale/rent, has_listings) | `chroma_store/` | ChromaDB `PersistentClient`, cosine distance |
| Retrieve top 12 chunks, plus a city's listings when the question names a city; answer and stream | `app.py` | Streamlit + OpenAI SDK pointed at OpenRouter |

Design choices:

- **Scraping is split from indexing.** The scrape runs locally and is saved to `data/pages.json`, which is
  committed. The Docker build only indexes that snapshot, so the build needs no browser and does not depend
  on the sites letting a cloud IP through.
- **Listing-aware chunking.** Each Wasalt listing card (price, title, district) is merged into one line
  before chunking, so a price is never split from its property.
- **Metadata filtering.** Pure similarity ranks Wasalt listing chunks below the same page's long SEO text,
  so when a question names Riyadh, Jeddah, Makkah or Dammam the app also fetches that city's listing chunks
  by metadata.
- **Grounded answers.** The system prompt allows answers only from the retrieved context, so the bot says
  when the data has no answer (for example, DG1 prices are not published) instead of inventing one.
- **Small footprint.** The embedding model runs on ONNX Runtime instead of PyTorch: about 280 MB of RAM
  under load, so it fits free 512 MB hosting.

## Data

25 pages, 302 chunks, scraped 18 Nov 2026:

- **DarGlobal (19 pages, live crawl):** home, projects, One of One, Why Invest, FAQ, Spain and Oman
  portfolio pages, and project pages (DG1, Trump International Dubai, The Astera, Da Vinci Tower,
  Urban Oasis, W Residences, Tierra Viva, Azure Oceanfront Villas, Trump Tower Jeddah, Les Vagues, Marea).
- **Wasalt (6 pages, about 150 listings):** the home page was crawled live. Wasalt's listing pages answer
  automated browsers with a Cloudflare challenge, which the scraper does not try to get around, so five
  listing pages (Riyadh sale and rent, Jeddah rent, Makkah sale, Dammam sale) were saved from a normal
  browser into `data/manual/wasalt/` and converted by the same crawl4ai pipeline.

The scraper crawls politely (one page every 10 s by default), skips bot-check pages, and keeps the last
good copy of each page between runs.

## Run locally

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
copy .env.example .env                                   # then put your OpenRouter key in .env
python scraper_indexer.py                                # builds ./chroma_store from data/pages.json
streamlit run app.py
```

To re-scrape: `pip install -r requirements-scraper.txt`, `python -m playwright install chromium`,
then `python scraper_indexer.py --scrape` (options: `--only DarGlobal|Wasalt`, `--delay SECONDS`).

## Run with Docker

```bash
docker build -t realestate-bot .
docker run -p 7860:7860 -e OPENROUTER_API_KEY=sk-or-... realestate-bot
```

Open http://localhost:7860. The container listens on `$PORT` if set (Render sets it), otherwise 7860.

## Deploy (Render, free plan)

1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, pick the repo. `render.yaml` defines a free Docker web service.
3. Enter `OPENROUTER_API_KEY` when Render asks for it, then **Apply**. The build takes about 5 minutes.

Free-plan services sleep after 15 minutes without traffic; the first visit after that takes about a
minute to wake the app.

## Model

A free OpenRouter model, `google/gemma-4-31b-it:free`, with free fallbacks
(`google/gemma-4-26b-a4b-it:free`, `nvidia/nemotron-3-ultra-550b-a55b:free`) that OpenRouter uses when the
first is busy. Override with `OPENROUTER_MODELS`. Free models allow 20 requests per minute and 50 per day
per account (1,000 per day once the account has bought at least $10 of credits).
