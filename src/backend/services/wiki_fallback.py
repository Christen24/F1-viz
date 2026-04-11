"""
Live Wikipedia fallback retrieval for biography-style F1 queries.
"""
from __future__ import annotations

import re
from typing import Any

import httpx

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
