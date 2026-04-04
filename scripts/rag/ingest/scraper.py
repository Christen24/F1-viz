"""
scraper.py — Content scrapers for the F1 RAG knowledge base.

Supports three scraping strategies:
  - Wikipedia articles (Playwright, extracts article body, strips boilerplate)
  - FIA regulation pages (Playwright, extracts main content)
  - General HTML pages (Playwright, extracts article/main body)
  - PDF documents (pdfplumber, extracts raw text page by page)

All scrapers return a ScrapedDocument dataclass with the raw text
and metadata, ready to pass to the chunker.
"""

import asyncio
import logging
import re
from html import unescape
from dataclasses import dataclass, field
from typing import Literal

log = logging.getLogger(__name__)

SourceType = Literal["wikipedia", "fia", "html", "pdf"]


@dataclass
class ScrapedDocument:
    url: str
    title: str
    category: str
    text: str
    source_type: SourceType
    season: int | None = None
    event_name: str | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HTTP fallback scraper (no Playwright browser binary required)
# ---------------------------------------------------------------------------

async def _scrape_http_fallback(url: str) -> str:
    import httpx

    async with httpx.AsyncClient(follow_redirects=True, timeout=45.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

    # Remove noise blocks first
    html = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", html)
    html = re.sub(r"(?is)<!--.*?-->", " ", html)

    # Try common content containers in order
    candidate = ""
    for pattern in (
        r'(?is)<main[^>]*>(.*?)</main>',
        r'(?is)<article[^>]*>(.*?)</article>',
        r'(?is)<div[^>]*id="mw-content-text"[^>]*>(.*?)</div>',
        r'(?is)<body[^>]*>(.*?)</body>',
    ):
        m = re.search(pattern, html)
        if m:
            candidate = m.group(1)
            if len(candidate) > 200:
                break

    if not candidate:
        candidate = html

    # Remove HTML tags and decode entities
    text = re.sub(r"(?is)<[^>]+>", "\n", candidate)
    text = unescape(text)
    return _clean_text(text)


# ---------------------------------------------------------------------------
# Wikipedia scraper
# ---------------------------------------------------------------------------

async def scrape_wikipedia(url: str, title: str, category: str, **kwargs) -> ScrapedDocument:
    """
    Scrape a Wikipedia article. Extracts the article body, strips navboxes,
    infoboxes, reference lists, and edit links to get clean prose.
    """
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)

            # Extract just the article content, excluding sidebars and navboxes
            text = await page.evaluate("""
                () => {
                    const content = document.querySelector('#mw-content-text .mw-parser-output');
                    if (!content) return '';

                    // Remove unwanted elements
                    const remove = [
                        '.navbox', '.vertical-navbox', '.infobox', '.ambox',
                        '.reflist', '.references', '.mw-editsection',
                        '.sidebar', '.hatnote', 'table', '.toc',
                        'style', 'script', '.noprint'
                    ];
                    remove.forEach(sel => {
                        content.querySelectorAll(sel).forEach(el => el.remove());
                    });

                    // Get text from paragraphs and headings only
                    const elements = content.querySelectorAll('p, h2, h3, h4');
                    return Array.from(elements)
                        .map(el => el.innerText.trim())
                        .filter(t => t.length > 0)
                        .join('\\n\\n');
                }
            """)

            await browser.close()
    except Exception as exc:
        log.warning("Wikipedia Playwright path failed for %s; using HTTP fallback (%s)", url, exc)
        text = await _scrape_http_fallback(url)

    text = _clean_text(text)
    log.info("Wikipedia scraped: %s (%d chars)", title, len(text))
    return ScrapedDocument(url=url, title=title, category=category, text=text,
                           source_type="wikipedia", **kwargs)


# ---------------------------------------------------------------------------
# FIA regulation scraper
# ---------------------------------------------------------------------------

async def scrape_fia(url: str, title: str, category: str, **kwargs) -> ScrapedDocument:
    """
    Scrape an FIA regulation page. Extracts the main article content.
    Falls back to general HTML scraper if FIA-specific selectors fail.
    """
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # FIA may require accepting cookies
            await page.goto(url, wait_until="networkidle", timeout=60_000)
            try:
                await page.click("button:has-text('Accept')", timeout=3_000)
            except Exception:
                pass

            text = await page.evaluate("""
                () => {
                    const selectors = [
                        '.field-body', '.regulation-content', 'article',
                        'main', '.content-body', '#content'
                    ];
                    for (const sel of selectors) {
                        const el = document.querySelector(sel);
                        if (el) {
                            ['nav', 'header', 'footer', '.breadcrumb'].forEach(s => {
                                el.querySelectorAll(s).forEach(n => n.remove());
                            });
                            return el.innerText.trim();
                        }
                    }
                    return document.body.innerText.trim();
                }
            """)

            await browser.close()
    except Exception as exc:
        log.warning("FIA Playwright path failed for %s; using HTTP fallback (%s)", url, exc)
        text = await _scrape_http_fallback(url)

    text = _clean_text(text)
    log.info("FIA scraped: %s (%d chars)", title, len(text))
    return ScrapedDocument(url=url, title=title, category=category, text=text,
                           source_type="fia", **kwargs)


# ---------------------------------------------------------------------------
# General HTML scraper
# ---------------------------------------------------------------------------

async def scrape_html(url: str, title: str, category: str, **kwargs) -> ScrapedDocument:
    """
    General-purpose Playwright scraper. Tries common article selectors
    (article, main, .article-body, etc.) and falls back to body text.
    """
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)

            text = await page.evaluate("""
                () => {
                    const selectors = [
                        'article', 'main', '.article-body', '.article__body',
                        '.story-body', '.content', '#content', '.post-content'
                    ];
                    for (const sel of selectors) {
                        const el = document.querySelector(sel);
                        if (el && el.innerText.length > 200) {
                            ['nav','header','footer','aside','.ad','.advertisement',
                             '.related','script','style'].forEach(s => {
                                el.querySelectorAll(s).forEach(n => n.remove());
                            });
                            return el.innerText.trim();
                        }
                    }
                    return document.body.innerText.trim().slice(0, 50000);
                }
            """)

            await browser.close()
    except Exception as exc:
        log.warning("HTML Playwright path failed for %s; using HTTP fallback (%s)", url, exc)
        text = await _scrape_http_fallback(url)

    text = _clean_text(text)
    log.info("HTML scraped: %s (%d chars)", title, len(text))
    return ScrapedDocument(url=url, title=title, category=category, text=text,
                           source_type="html", **kwargs)


# ---------------------------------------------------------------------------
# PDF scraper
# ---------------------------------------------------------------------------

async def scrape_pdf(url: str, title: str, category: str, **kwargs) -> ScrapedDocument:
    """
    Download a PDF and extract its text using pdfplumber.
    Works for FIA regulation PDFs.
    """
    import io
    import httpx
    import pdfplumber

    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=60, follow_redirects=True)
        response.raise_for_status()
        pdf_bytes = response.content

    pages_text: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text.strip())

    text = _clean_text("\n\n".join(pages_text))
    log.info("PDF scraped: %s (%d pages, %d chars)", title, len(pages_text), len(text))
    return ScrapedDocument(url=url, title=title, category=category, text=text,
                           source_type="pdf", **kwargs)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

async def scrape(source: dict) -> ScrapedDocument | None:
    """
    Dispatch to the correct scraper based on source dict from sources.json.
    Returns None on failure (logged but not re-raised, so the pipeline continues).
    """
    url = source["url"]
    title = source.get("title", url)
    category = source.get("category", "general")
    source_type = source.get("type", "html")
    extra = {
        k: source[k]
        for k in ("season", "event_name")
        if k in source
    }

    try:
        if "wikipedia.org" in url:
            return await scrape_wikipedia(url, title, category, **extra)
        elif source_type == "pdf" or url.endswith(".pdf"):
            return await scrape_pdf(url, title, category, **extra)
        elif "fia.com" in url:
            return await scrape_fia(url, title, category, **extra)
        else:
            return await scrape_html(url, title, category, **extra)
    except Exception as exc:
        log.error("Scrape failed for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalise whitespace, strip citation markers [1], [2] etc."""
    text = re.sub(r"\[\d+\]", "", text)         # remove [1] citation refs
    text = re.sub(r"\[edit\]", "", text)         # remove Wikipedia edit links
    text = re.sub(r"\n{3,}", "\n\n", text)       # collapse excessive blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)       # collapse horizontal whitespace
    return text.strip()
