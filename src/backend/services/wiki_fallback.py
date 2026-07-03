"""
Live Wikipedia + DuckDuckGo web-scrape fallback for any unanswered F1 query.
"""
from __future__ import annotations

import html as _html_mod
import re
from typing import Any

import httpx

# ── DuckDuckGo instant-answer + HTML scraper ─────────────────────────────────
_DDG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


async def web_search_f1(query: str, max_results: int = 3) -> dict[str, Any]:
    """
    Multi-stage web scrape for any F1 query. Returns RAG payload {context, sources}.

    Priority:
    1. Wikipedia REST summary (direct slug guess from query)
    2. DuckDuckGo instant-answer API abstract
    3. DuckDuckGo HTML snippet scrape
    4. Wikipedia opensearch → fetch extract of top result
    """
    context_parts: list[str] = []
    sources: list[dict[str, Any]] = []

    # ── Step 0: Championship-winner queries get a dedicated, verified lookup ──
    # (avoids Step 1 below dumping a generic season-summary paragraph that
    # never actually names the champion).
    champ_req = _extract_championship_request(query)
    if champ_req:
        champ_result = await _lookup_championship_winner(*champ_req)
        if champ_result:
            return champ_result
        # Couldn't confidently extract a winner — don't let Step 1 below guess
        # by returning an unrelated season-summary paragraph for this query.

    async with httpx.AsyncClient(
        headers=_DDG_HEADERS, timeout=20.0, follow_redirects=True
    ) as client:

        # ── Step 1: Direct Wikipedia REST summary guess ────────────────────────
        # Try constructing a Wikipedia page title directly from the query.
        # e.g. "who won the 2024 championship" → "2024 Formula One World Championship"
        # Skipped for championship-winner queries (handled, or deliberately left
        # unanswered, by Step 0 above) so we never surface an irrelevant intro
        # paragraph as if it answered "who won".
        year_match = re.search(r"\b(20\d{2})\b", query)
        wiki_candidates: list[str] = []
        if year_match and not champ_req:
            yr = year_match.group(1)
            wiki_candidates = [
                f"{yr} Formula One World Championship",
                f"{yr} FIA Formula One World Championship",
                f"{yr} Formula One season",
            ]

        async def _fetch_wiki_extract(title: str) -> tuple[str, str]:
            """Returns (extract_text, page_url). Empty strings on failure."""
            slug = title.replace(" ", "_")
            try:
                resp = await client.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}",
                    headers={"User-Agent": "f1-viz/1.0", "Accept": "application/json"},
                )
                if resp.status_code >= 400:
                    return "", ""
                data = resp.json()
                extract = str(data.get("extract") or "").strip()
                url = str(
                    ((data.get("content_urls") or {}).get("desktop") or {}).get("page")
                    or f"https://en.wikipedia.org/wiki/{slug}"
                )
                return extract, url
            except Exception:
                return "", ""

        for candidate in wiki_candidates:
            extract, url = await _fetch_wiki_extract(candidate)
            if extract and len(extract) > 80:
                # Take first 600 chars to keep it focused
                context_parts.append(f"Wikipedia — {candidate}:\n{extract[:600]}")
                sources.append({
                    "id": "wiki-direct-1",
                    "rank": 1,
                    "title": candidate,
                    "source": url,
                    "category": "wikipedia_live",
                    "score": 1.0,
                })
                break

        # ── Step 2: DuckDuckGo instant-answer API ─────────────────────────────
        if not context_parts:
            search_query = f"Formula 1 F1 {query}"
            try:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": search_query, "format": "json", "no_redirect": 1, "no_html": 1},
                )
                if resp.status_code < 400:
                    ddg = resp.json()
                    abstract = str(ddg.get("AbstractText") or "").strip()
                    abstract_url = str(ddg.get("AbstractURL") or "").strip()
                    abstract_source = str(ddg.get("AbstractSource") or "").strip()
                    if abstract:
                        context_parts.append(f"DuckDuckGo Instant Answer ({abstract_source}):\n{abstract}")
                        sources.append({
                            "id": "ddg-instant",
                            "rank": 1,
                            "title": f"{abstract_source} — {query[:60]}",
                            "source": abstract_url or "https://duckduckgo.com",
                            "category": "web_search",
                            "score": 1.0,
                        })
                    # RelatedTopics often has season-winner sentences
                    related: list[str] = []
                    for topic in (ddg.get("RelatedTopics") or [])[:6]:
                        if isinstance(topic, dict):
                            txt = str(topic.get("Text") or "").strip()
                            if txt and len(txt) > 20:
                                related.append(txt)
                    if related:
                        context_parts.append("Related info:\n" + "\n".join(related[:3]))
            except Exception:
                pass

        # ── Step 3: DDG HTML snippet scrape ───────────────────────────────────
        if not context_parts:
            search_query = f"Formula 1 F1 {query}"
            try:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": search_query},
                )
                if resp.status_code < 400:
                    body = resp.text
                    snippets = re.findall(
                        r'class="result__snippet"[^>]*>([^<]+(?:<[^/][^>]*>[^<]*</[^>]+>)*[^<]*)',
                        body,
                    )
                    titles = re.findall(r'class="result__a"[^>]*>([^<]+)', body)
                    links = re.findall(r'class="result__url"[^>]*>\s*([^<\s]+)', body)

                    good_snippets = 0
                    for i, snippet in enumerate(snippets[:max_results + 3]):
                        clean = re.sub(r"<[^>]+>", " ", snippet)
                        clean = _html_mod.unescape(re.sub(r"\s+", " ", clean)).strip()
                        if not clean or len(clean) < 30:
                            continue
                        title = _html_mod.unescape(titles[i]) if i < len(titles) else f"Result {i+1}"
                        url = f"https://{links[i]}" if i < len(links) else "https://duckduckgo.com"
                        context_parts.append(f"[{good_snippets+1}] {title}\n{clean}")
                        sources.append({
                            "id": f"ddg-web-{good_snippets+1}",
                            "rank": good_snippets + 1,
                            "title": title,
                            "source": url,
                            "category": "web_search",
                            "score": 1.0 - good_snippets * 0.1,
                        })
                        good_snippets += 1
                        if good_snippets >= max_results:
                            break
            except Exception:
                pass

        # ── Step 4: Wikipedia opensearch → fetch extract ──────────────────────
        if not context_parts:
            search_query = f"Formula 1 F1 {query}"
            try:
                resp = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "opensearch",
                        "search": search_query,
                        "limit": 3,
                        "namespace": 0,
                        "format": "json",
                    },
                    headers={"User-Agent": "f1-viz/1.0", "Accept": "application/json"},
                )
                if resp.status_code < 400:
                    data = resp.json()
                    wiki_titles_found = data[1] if len(data) > 1 else []
                    wiki_urls_found = data[3] if len(data) > 3 else []
                    for i, wt in enumerate(wiki_titles_found[:2]):
                        extract, url = await _fetch_wiki_extract(str(wt))
                        if extract and len(extract) > 60:
                            context_parts.append(f"Wikipedia — {wt}:\n{extract[:500]}")
                            sources.append({
                                "id": f"wiki-search-{i+1}",
                                "rank": i + 1,
                                "title": str(wt),
                                "source": url or (wiki_urls_found[i] if i < len(wiki_urls_found) else "https://en.wikipedia.org"),
                                "category": "wikipedia_live",
                                "score": 0.9,
                            })
                            break  # One good Wikipedia article is enough
            except Exception:
                pass

    if not context_parts:
        return {"context": "", "sources": []}

    return {
        "context": "\n\n".join(context_parts),
        "sources": sources,
    }


_BORN_QUERY_RE = re.compile(
    r"\bwhere\s+(?:was|is)\s+(?P<name>[a-zA-Z][a-zA-Z\s\.\-']{1,80}?)\s+(?:born|from)\b",
    flags=re.IGNORECASE,
)
_WHO_IS_RE = re.compile(
    r"\bwho\s+is\s+(?P<name>[a-zA-Z][a-zA-Z\s\.\-']{1,80})\b",
    flags=re.IGNORECASE,
)
_BIRTHPLACE_RE_1 = re.compile(
    r"\b(?:what(?:'s| is)|tell me)\s+(?P<name>[a-zA-Z][a-zA-Z\s\.\-']{1,80}?)\s+(?:birthplace|place of birth)\b",
    flags=re.IGNORECASE,
)
_BIRTHPLACE_RE_2 = re.compile(
    r"\b(?P<name>[a-zA-Z][a-zA-Z\s\.\-']{1,80}?)(?:'s|s)\s+(?:birthplace|place of birth)\b",
    flags=re.IGNORECASE,
)
_BIRTHPLACE_RE_3 = re.compile(
    r"\b(?P<name>[a-zA-Z][a-zA-Z\s\.\-']{1,80}?)\s+birthplace\b",
    flags=re.IGNORECASE,
)
_RACE_WINNER_RE_1 = re.compile(
    r"\bwho\s+won\s+(?:the\s+)?(?P<year>20\d{2})\s+(?P<event>.+)$",
    flags=re.IGNORECASE,
)
_RACE_WINNER_RE_2 = re.compile(
    r"\bwho\s+won\s+(?:the\s+)?(?P<event>.+?)\s+(?P<year>20\d{2})\b",
    flags=re.IGNORECASE,
)

# Championship-winner queries, e.g. "who won the drivers championship in 2025",
# "who won the 2025 constructors championship", "who was 2024 world champion".
# Distinct from _RACE_WINNER_RE_* because there's no specific Grand Prix event
# name here — the "event" is the championship itself, and a year can appear
# either before or after the championship phrase, sometimes with filler words
# like "in the year" in between.
_CHAMPIONSHIP_QUERY_RE_1 = re.compile(
    r"\bwho\s+won\s+(?:the\s+)?(?P<ctype>driver'?s?|drivers'?|constructor'?s?|constructors'?)?\s*"
    r"(?:world\s+)?championship\b.*?\b(?P<year>20\d{2})\b",
    flags=re.IGNORECASE,
)
_CHAMPIONSHIP_QUERY_RE_2 = re.compile(
    r"\bwho\s+(?:is|was)\s+(?:the\s+)?(?P<year>20\d{2})\s+"
    r"(?P<ctype>driver'?s?|drivers'?|constructor'?s?|constructors'?)?\s*(?:world\s+)?champion\b",
    flags=re.IGNORECASE,
)
_CHAMPIONSHIP_QUERY_RE_3 = re.compile(
    r"\bwho\s+won\s+(?:the\s+)?(?P<year>20\d{2})\s+"
    r"(?P<ctype>driver'?s?|drivers'?|constructor'?s?|constructors'?)?\s*(?:world\s+)?championship\b",
    flags=re.IGNORECASE,
)

_DRIVER_CHAMPION_PATTERNS = (
    r"(?P<name>[A-Z][\w.\-']+(?:\s+[A-Z][\w.\-']+){0,3})\s+"
    r"(?:clinched|won|secured|claimed)\s+(?:his\s+\S+\s+)?(?:the\s+)?"
    r"(?:World\s+)?Drivers['’]?\s+(?:World\s+)?Championship",
    r"Drivers['’]?\s+(?:World\s+)?Championship\s+(?:title\s+)?(?:was\s+)?won\s+by\s+"
    r"(?P<name>[A-Z][\w.\-']+(?:\s+[A-Z][\w.\-']+){0,3})",
    r"(?P<name>[A-Z][\w.\-']+(?:\s+[A-Z][\w.\-']+){0,3})\s+"
    r"(?:won|claimed|secured)\s+(?:his\s+\S+\s+)?(?:consecutive\s+)?(?:World\s+)?"
    r"(?:Drivers['’]?\s+)?(?:World\s+)?title\b",
)
_CONSTRUCTOR_CHAMPION_PATTERNS = (
    r"Constructors['’]?\s+(?:World\s+)?Championship\s+(?:title\s+)?(?:was\s+)?won\s+by\s+"
    r"(?P<name>[A-Z][\w.&\-' ]+?)(?:[.,;]|\s+with|\s+ahead|\s+for\b)",
    r"(?P<name>[A-Z][\w.&\-' ]+?)\s+"
    r"(?:won|clinched|secured|claimed)\s+(?:their\s+\S+\s+)?(?:consecutive\s+)?(?:the\s+)?"
    r"(?:World\s+)?Constructors['’]?\s+(?:World\s+)?Championship",
)


def _extract_championship_request(query: str) -> tuple[int, str] | None:
    """Returns (year, championship_type) where type is 'drivers' or 'constructors'."""
    q = (query or "").strip()
    m = (
        _CHAMPIONSHIP_QUERY_RE_1.search(q)
        or _CHAMPIONSHIP_QUERY_RE_2.search(q)
        or _CHAMPIONSHIP_QUERY_RE_3.search(q)
    )
    if not m:
        return None
    year = int(m.group("year"))
    ctype_raw = (m.group("ctype") or "").lower()
    ctype = "constructors" if ctype_raw.startswith("constructor") else "drivers"
    return year, ctype


def _extract_champion_name(text: str, championship_type: str) -> str | None:
    patterns = (
        _CONSTRUCTOR_CHAMPION_PATTERNS
        if championship_type == "constructors"
        else _DRIVER_CHAMPION_PATTERNS
    )
    for pattern in patterns:
        # No IGNORECASE here: these patterns rely on [A-Z] to anchor onto
        # proper nouns (driver/team names). Under IGNORECASE, [A-Z] also
        # matches lowercase letters, so common lowercase words ("the",
        # "reigning", etc.) get swept into the captured name. The keyword
        # alternatives below (won/clinched/...) are written in the casing
        # actually used in Wikipedia prose, so this is safe without it.
        match = re.search(pattern, text)
        if match:
            name = match.group("name").strip().rstrip(".,;")
            if len(name) > 1:
                return name
    return None


async def _lookup_championship_winner(year: int, championship_type: str) -> dict[str, Any] | None:
    """
    Fetch the season article and try to extract the actual champion (driver or
    constructor). Returns a RAG payload on success, None if we can't confidently
    identify a winner (e.g. the season hasn't concluded yet) — callers should
    treat None as "try the next fallback" rather than surfacing a guess.
    """
    title = f"{year} Formula One World Championship"
    slug = _slugify_title(title)
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=_WIKI_HEADERS) as client:
        try:
            resp = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "titles": title,
                    "format": "json",
                },
            )
            if resp.status_code >= 400:
                return None
            payload = resp.json()
            pages = ((payload.get("query") or {}).get("pages") or {})
            if not pages:
                return None
            page = next(iter(pages.values()))
            full_text = str(page.get("extract") or "")
        except Exception:
            return None

    if not full_text:
        return None

    champion = _extract_champion_name(full_text, championship_type)
    if not champion:
        # Season likely still in progress, or the article phrases it in a way
        # our patterns don't catch — don't guess, let the caller fall through.
        return None

    label = "Constructors' Champion" if championship_type == "constructors" else "Drivers' Champion"
    source_url = f"https://en.wikipedia.org/wiki/{slug}"
    return {
        "context": f"[1] Wikipedia — {title}\n{year} {label}: {champion}.\nSource: {source_url}",
        "sources": [
            {
                "id": "wiki-championship-1",
                "rank": 1,
                "title": title,
                "source": source_url,
                "category": "wikipedia_live",
                "score": 1.0,
            }
        ],
    }
_WIKI_HEADERS = {
    "User-Agent": "f1-viz/1.0 (local-dev; contact: local@f1viz)",
    "Accept": "application/json",
}

# Known F1 driver name disambiguations — maps common names to their
# Wikipedia article titles to avoid disambiguation pages.
_F1_DRIVER_DISAMBIGUATION: dict[str, str] = {
    "carlos sainz": "Carlos Sainz Jr.",
    "max verstappen": "Max Verstappen",
    "lewis hamilton": "Lewis Hamilton",
    "charles leclerc": "Charles Leclerc",
    "lando norris": "Lando Norris",
    "oscar piastri": "Oscar Piastri",
    "george russell": "George Russell (racing driver)",
    "fernando alonso": "Fernando Alonso",
    "pierre gasly": "Pierre Gasly",
    "esteban ocon": "Esteban Ocon",
    "daniel ricciardo": "Daniel Ricciardo",
    "valtteri bottas": "Valtteri Bottas",
    "nico hulkenberg": "Nico Hülkenberg",
    "kevin magnussen": "Kevin Magnussen",
    "alexander albon": "Alexander Albon",
    "yuki tsunoda": "Yuki Tsunoda",
    "lance stroll": "Lance Stroll",
    "logan sargeant": "Logan Sargeant",
    "zhou guanyu": "Zhou Guanyu",
    "sergio perez": "Sergio Pérez",
    "nyck de vries": "Nyck de Vries",
    "jack doohan": "Jack Doohan",
    "oliver bearman": "Oliver Bearman",
    "andrea kimi antonelli": "Andrea Kimi Antonelli",
    "kimi antonelli": "Andrea Kimi Antonelli",
    "gabriel bortoleto": "Gabriel Bortoleto",
    "isack hadjar": "Isack Hadjar",
    "liam lawson": "Liam Lawson",
}


def _is_disambiguation(text: str) -> bool:
    """Check if a Wikipedia extract is a disambiguation page."""
    low = (text or "").lower().strip()
    return (
        low.startswith("may refer to")
        or "may refer to:" in low
        or "can refer to:" in low
        or "most commonly refers to:" in low
        or "(disambiguation)" in low
    )


def _extract_person_name(query: str) -> str | None:
    q = (query or "").strip()
    m = _BORN_QUERY_RE.search(q)
    if m:
        return m.group("name").strip()
    for rx in (_BIRTHPLACE_RE_1, _BIRTHPLACE_RE_2, _BIRTHPLACE_RE_3):
        m = rx.search(q)
        if m:
            return m.group("name").strip()
    m = _WHO_IS_RE.search(q)
    if m:
        return m.group("name").strip()
    return None


def _is_birthplace_intent(query: str) -> bool:
    """
    True if the query specifically asks for a birthplace (as opposed to a
    general "who is X" bio question). Used so that when birthplace extraction
    fails, we don't silently substitute an unrelated bio sentence as if it
    answered the question that was actually asked.
    """
    q = (query or "").strip()
    if _BORN_QUERY_RE.search(q):
        return True
    return any(rx.search(q) for rx in (_BIRTHPLACE_RE_1, _BIRTHPLACE_RE_2, _BIRTHPLACE_RE_3))


def _normalize_name(name: str) -> str:
    name = re.sub(r"\s+", " ", (name or "").strip())
    name = re.sub(r"^(?:what(?:'s|s| is)|tell me)\s+", "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"(?:'s|s)$", "", name, flags=re.IGNORECASE).strip()
    # Remove trailing role words accidentally captured by loose queries
    name = re.sub(r"\b(?:birthplace|place of birth|born|from)\b.*$", "", name, flags=re.IGNORECASE).strip()
    return name


def _normalize_event(event: str) -> str:
    value = re.sub(r"\bgp\b", "grand prix", (event or ""), flags=re.IGNORECASE)
    value = re.sub(r"[^\w\s-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _slugify_title(value: str) -> str:
    return value.replace(" ", "_")


def _ascii_fold(value: str) -> str:
    # Minimal accent fold for title variants.
    import unicodedata

    txt = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in txt if not unicodedata.combining(ch))


def _extract_winner_request(query: str) -> tuple[int, str] | None:
    raw = (query or "").strip()
    m = _RACE_WINNER_RE_1.search(raw) or _RACE_WINNER_RE_2.search(raw)
    if not m:
        return None
    year = int(m.group("year"))
    event = _normalize_event(m.group("event"))
    if not event:
        return None
    return year, event


def _first_sentences(text: str, limit: int = 2) -> str:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
    return " ".join(parts[:limit]).strip()


def _extract_birthplace(intro: str) -> str | None:
    m = re.search(r"\bborn\b[^.]*?\bin\s+([A-Z][^.;]+)", intro or "")
    if not m:
        return None
    place = m.group(1).strip().rstrip(",")
    if len(place) < 2:
        return None
    return place


async def retrieve_wikipedia_context(query: str) -> dict[str, Any]:
    """
    Returns RAG-like payload: {context, sources}. Empty payload when unresolved.
    """
    person = _extract_person_name(query)
    if not person:
        champ_req = _extract_championship_request(query)
        if champ_req:
            champ_year, champ_type = champ_req
            champ_result = await _lookup_championship_winner(champ_year, champ_type)
            if champ_result:
                return champ_result
            # Couldn't confidently extract a champion (e.g. season still in
            # progress) — fall through to other handlers rather than guessing.

        winner_req = _extract_winner_request(query)
        if winner_req:
            year, event = winner_req
            base_title = f"{year} {event.title()}"
            candidates = [base_title]
            ascii_title = _ascii_fold(base_title)
            if ascii_title != base_title:
                candidates.append(ascii_title)
            if "brazil" in event.lower():
                candidates = [
                    f"{year} São Paulo Grand Prix",
                    f"{year} Sao Paulo Grand Prix",
                    f"{year} Brazilian Grand Prix",
                ] + candidates

            # preserve order, remove duplicates
            seen_titles: set[str] = set()
            deduped_candidates: list[str] = []
            for c in candidates:
                key = c.strip().lower()
                if key and key not in seen_titles:
                    seen_titles.add(key)
                    deduped_candidates.append(c)
            candidates = deduped_candidates

            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=_WIKI_HEADERS) as client:
                for title_guess in candidates:
                    slug = _slugify_title(title_guess)
                    try:
                        resp = await client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}")
                        if resp.status_code >= 400:
                            continue
                        payload = resp.json()
                        title = str(payload.get("title") or title_guess)
                        extract = str(payload.get("extract") or "").strip()
                        source_url = str(
                            ((payload.get("content_urls") or {}).get("desktop") or {}).get("page")
                            or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                        )
                        # Summary often omits winner; fetch intro extract as stronger fallback.
                        if not extract:
                            try:
                                ex_resp = await client.get(
                                    "https://en.wikipedia.org/w/api.php",
                                    params={
                                        "action": "query",
                                        "prop": "extracts",
                                        "explaintext": 1,
                                        "exintro": 1,
                                        "titles": title,
                                        "format": "json",
                                    },
                                )
                                if ex_resp.status_code < 400:
                                    ex_payload = ex_resp.json()
                                    pages = ((ex_payload.get("query") or {}).get("pages") or {})
                                    if pages:
                                        page = next(iter(pages.values()))
                                        extract = str(page.get("extract") or "").strip()
                            except Exception:
                                pass
                        if not extract:
                            continue
                        winner = None
                        winner_patterns = (
                            r"\bwon by\s+([A-Z][A-Za-z\.\-'\s]+?)(?:[.,;]|\s+driving|\s+after|\s+for\b)",
                            r"\bwinner(?: was|:)\s+([A-Z][A-Za-z\.\-'\s]+?)(?:[.,;]|\s+driving|\s+after|\s+for\b)",
                            r"\b([A-Z][A-Za-z\.\-'\s]+?)\s+won\s+the\s+\d{4}\s+[A-Za-z\s]+Grand Prix\b",
                            r"\bthe race was won by\s+([A-Z][A-Za-z\.\-'\s]+?)(?:[.,;]|\s+driving|\s+after|\s+for\b)",
                        )
                        for pattern in winner_patterns:
                            match = re.search(pattern, extract, flags=re.IGNORECASE)
                            if match:
                                winner = " ".join(part.capitalize() for part in match.group(1).strip().split())
                                break

                        if not winner:
                            # For winner-intent queries, avoid returning generic race description.
                            # Returning empty context allows upstream deterministic "couldn't verify" handling.
                            continue

                        answer_line = f"{year} {title}: winner was {winner}."
                        return {
                            "context": f"[1] Wikipedia live lookup\n{answer_line}\nSource: {source_url}",
                            "sources": [
                                {
                                    "id": "wiki-live-race-1",
                                    "rank": 1,
                                    "title": title,
                                    "source": source_url,
                                    "category": "wikipedia_live",
                                    "score": 1.0,
                                }
                            ],
                        }
                    except Exception:
                        continue
        return {"context": "", "sources": []}
    person = _normalize_name(person)

    # Check disambig mapping first (e.g. "carlos sainz" → "Carlos Sainz Jr.")
    wiki_search_name = _F1_DRIVER_DISAMBIGUATION.get(person.lower(), person)

    title: str | None = None
    source_url: str | None = None
    intro = ""

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=_WIKI_HEADERS) as client:
        # Try REST summary first (most reliable for specific article titles)
        async def _fetch_rest_summary(search_name: str) -> tuple[str | None, str | None, str]:
            slug = search_name.replace(" ", "_")
            try:
                rest_resp = await client.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
                )
                if rest_resp.status_code >= 400:
                    return None, None, ""
                rest = rest_resp.json()
                t = str(rest.get("title") or search_name)
                u = str(
                    ((rest.get("content_urls") or {}).get("desktop") or {}).get("page")
                    or f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}"
                )
                ext = str(rest.get("extract") or "").strip()
                return t, u, ext
            except Exception:
                return None, None, ""

        # Path A: Try the disambiguated/mapped name via REST first
        title, source_url, intro = await _fetch_rest_summary(wiki_search_name)

        # If we got a disambiguation page, retry with " (racing driver)" suffix
        if intro and _is_disambiguation(intro):
            title, source_url, intro = await _fetch_rest_summary(f"{wiki_search_name} (racing driver)")

        # Path B: MediaWiki search API fallback
        if not title or not intro or _is_disambiguation(intro):
            try:
                search_resp = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "opensearch",
                        "search": f"{wiki_search_name} formula one",
                        "limit": 5,
                        "namespace": 0,
                        "format": "json",
                    },
                )
                if search_resp.status_code < 400:
                    data = search_resp.json()
                    titles_list = data[1] if isinstance(data, list) and len(data) > 1 else []
                    urls_list = data[3] if isinstance(data, list) and len(data) > 3 else []
                    # Prefer results that are NOT disambiguation pages
                    for i, candidate_title in enumerate(titles_list):
                        if "(disambiguation)" not in str(candidate_title).lower():
                            title = str(candidate_title)
                            source_url = str(urls_list[i]) if i < len(urls_list) else None
                            break
            except Exception:
                pass

        # Fetch full intro if not already obtained
        if title and (not intro or _is_disambiguation(intro)):
            try:
                _, source_url_new, intro_new = await _fetch_rest_summary(title)
                if intro_new and not _is_disambiguation(intro_new):
                    intro = intro_new
                    if source_url_new:
                        source_url = source_url_new
            except Exception:
                pass

        # Path C: Full extract API as final fallback
        if title and (not intro or _is_disambiguation(intro)):
            try:
                extract_resp = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "prop": "extracts",
                        "explaintext": 1,
                        "exintro": 1,
                        "titles": title,
                        "format": "json",
                    },
                )
                extract_resp.raise_for_status()
                payload = extract_resp.json()
                pages = ((payload.get("query") or {}).get("pages") or {})
                if pages:
                    page = next(iter(pages.values()))
                    candidate_intro = str(page.get("extract") or "").strip()
                    if candidate_intro and not _is_disambiguation(candidate_intro):
                        intro = candidate_intro
            except Exception:
                pass

    if not intro or _is_disambiguation(intro):
        return {"context": "", "sources": []}

    birth_place = _extract_birthplace(intro)
    if birth_place:
        answer_line = f"{title} was born in {birth_place}."
    elif _is_birthplace_intent(query):
        # The question specifically asked for a birthplace and we couldn't
        # find one in the fetched intro — don't paper over that by returning
        # an unrelated bio sentence as if it were the answer. Return empty so
        # the caller escalates to the next fallback (e.g. a web search) that
        # might actually find it, rather than surfacing a non-answer.
        return {"context": "", "sources": []}
    else:
        answer_line = _first_sentences(intro, limit=2) or intro[:300]

    context = (
        f"[1] Wikipedia live lookup\n"
        f"{answer_line}\n"
        f"Source: {source_url}\n"
        f"Summary: {_first_sentences(intro, limit=3)}"
    )
    sources = [
        {
            "id": "wiki-live-1",
            "rank": 1,
            "title": title,
            "source": source_url or f"https://en.wikipedia.org/wiki/{(title or person).replace(' ', '_')}",
            "category": "wikipedia_live",
            "score": 1.0,
        }
    ]
    return {"context": context, "sources": sources}
