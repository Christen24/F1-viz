"""
Race Intelligence Platform — Video Source Resolver

Maps race sessions to video sources:
  1. OvertakeFans.com — full race replays (primary)
  2. YouTube — auto-searches for highlight video IDs and caches them
"""
import json
import logging
import re
from pathlib import Path
from urllib.parse import quote_plus

from src.backend.config import settings
from src.backend.schemas import VideoSource

logger = logging.getLogger(__name__)

# GP name -> URL slug mapping for overtakefans.com
GP_SLUGS: dict[str, str] = {
    "bahrain": "bahrain-grand-prix",
    "saudi arabia": "saudi-arabian-grand-prix",
    "saudi arabian": "saudi-arabian-grand-prix",
    "australia": "australian-grand-prix",
    "australian": "australian-grand-prix",
    "japan": "japanese-grand-prix",
    "japanese": "japanese-grand-prix",
    "china": "chinese-grand-prix",
    "chinese": "chinese-grand-prix",
    "miami": "miami-grand-prix",
    "emilia romagna": "emilia-romagna-grand-prix",
    "monaco": "monaco-grand-prix",
    "canada": "canadian-grand-prix",
    "canadian": "canadian-grand-prix",
    "spain": "spanish-grand-prix",
    "spanish": "spanish-grand-prix",
    "austria": "austrian-grand-prix",
    "austrian": "austrian-grand-prix",
    "great britain": "british-grand-prix",
    "britain": "british-grand-prix",
    "british": "british-grand-prix",
    "hungary": "hungarian-grand-prix",
    "hungarian": "hungarian-grand-prix",
    "belgium": "belgian-grand-prix",
    "belgian": "belgian-grand-prix",
    "netherlands": "dutch-grand-prix",
    "dutch": "dutch-grand-prix",
    "italy": "italian-grand-prix",
    "italian": "italian-grand-prix",
    "azerbaijan": "azerbaijan-grand-prix",
    "singapore": "singapore-grand-prix",
    "united states": "united-states-grand-prix",
    "us": "united-states-grand-prix",
    "mexico": "mexico-city-grand-prix",
    "mexico city": "mexico-city-grand-prix",
    "são paulo": "sao-paulo-grand-prix",
    "sao paulo": "sao-paulo-grand-prix",
    "brazil": "sao-paulo-grand-prix",
    "las vegas": "las-vegas-grand-prix",
    "qatar": "qatar-grand-prix",
    "abu dhabi": "abu-dhabi-grand-prix",
}

# Disk cache for YouTube video IDs (survives restarts)
_YT_CACHE_FILE = settings.processed_dir / "_youtube_cache.json"


def _load_yt_cache() -> dict[str, str]:
    """Load cached YouTube video IDs from disk."""
    if _YT_CACHE_FILE.exists():
        try:
            with open(_YT_CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_yt_cache(cache: dict[str, str]) -> None:
    """Save YouTube video ID cache to disk."""
    _YT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_YT_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def _search_youtube_video_id(query: str) -> str:
    """
    Search YouTube and return the first video ID.

    Uses a simple HTTP request to YouTube's search results page
    and parses the video ID from the response HTML.
    No API key needed.
    """
    import urllib.request
    import urllib.error

    search_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"

    try:
        req = urllib.request.Request(
            search_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # YouTube embeds video IDs in the initial page data as "videoId":"XXXXXXXXXXX"
        matches = re.findall(r'"videoId"\s*:\s*"([a-zA-Z0-9_-]{11})"', html)
        if matches:
            # Return the first unique video ID (skip shorts/ads)
            seen = set()
            for vid in matches:
                if vid not in seen:
                    seen.add(vid)
                    return vid

    except (urllib.error.URLError, TimeoutError, Exception) as e:
        logger.warning("YouTube search failed for '%s': %s", query, e)

    return ""


def _find_youtube_highlight(year: int, gp: str) -> str:
    """
    Find YouTube video ID for race highlights.

    Searches with specific terms to find the official F1 highlight video.
    Results are cached to disk to avoid repeated searches.
    """
    cache_key = f"{year}_{gp.lower().replace(' ', '_')}"
    cache = _load_yt_cache()

    if cache_key in cache:
        return cache[cache_key]

    # Search YouTube for the official race highlights
    query = f"Formula 1 {year} {gp} Grand Prix Race Highlights"
    video_id = _search_youtube_video_id(query)

    if video_id:
        logger.info("Found YouTube highlight for %s %d: %s", gp, year, video_id)
        cache[cache_key] = video_id
        _save_yt_cache(cache)
    else:
        logger.info("No YouTube highlight found for %s %d", gp, year)

    return video_id


def _build_overtakefans_url(year: int, gp: str) -> str:
    """Build overtakefans.com watch URL from year and GP name.
    
    Format: https://overtakefans.com/f1-race-archive/watch/index.php?race=2025-australian-grand-prix-formula-1-race
    """
    gp_lower = gp.lower().strip()

    slug = GP_SLUGS.get(gp_lower, "")
    if not slug:
        for key, val in GP_SLUGS.items():
            if key in gp_lower or gp_lower in key:
                slug = val
                break
    if not slug:
        slug = gp_lower.replace(" ", "-") + "-grand-prix"

    race_param = f"{year}-{slug}-formula-1-race"
    return f"https://overtakefans.com/f1-race-archive/watch/index.php?race={race_param}"


def _build_youtube_search_url(year: int, gp: str) -> str:
    """Build a YouTube search URL for race highlights (fallback)."""
    query = f"Formula 1 {year} {gp} Grand Prix Race Highlights"
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}"


def _find_youtube_fullrace(year: int, gp: str) -> str:
    """
    Find YouTube video ID for full race replay.
    Searches with specific terms and caches result to disk.
    """
    cache_key = f"fullrace_{year}_{gp.lower().replace(' ', '_')}"
    cache = _load_yt_cache()

    if cache_key in cache:
        return cache[cache_key]

    # Try several search queries to find a full race replay
    queries = [
        f"Formula 1 {year} {gp} Grand Prix full race replay",
        f"F1 {year} {gp} Grand Prix race full",
        f"F1 {year} {gp} GP race",
    ]

    video_id = ""
    for query in queries:
        video_id = _search_youtube_video_id(query)
        if video_id:
            break

    if video_id:
        logger.info("Found YouTube full race for %s %d: %s", gp, year, video_id)
        cache[cache_key] = video_id
        _save_yt_cache(cache)
    else:
        logger.info("No YouTube full race found for %s %d", gp, year)

    return video_id


def resolve_video_source(year: int, gp: str, session_type: str = "R") -> VideoSource:
    """Resolve video source for a race session.

    Searches YouTube for both highlight and full race video IDs.
    Falls back to OvertakeFans embed URL and YouTube search links.
    """
    overtakefans_url = _build_overtakefans_url(year, gp)
    youtube_search_url = _build_youtube_search_url(year, gp)

    # Try to find YouTube video IDs (cached after first search)
    youtube_id = _find_youtube_highlight(year, gp)
    fullrace_id = _find_youtube_fullrace(year, gp)

    return VideoSource(
        provider="youtube",
        video_id=youtube_id,
        fullrace_video_id=fullrace_id,
        embed_url=overtakefans_url,
        thumbnail_url=f"https://img.youtube.com/vi/{youtube_id}/hqdefault.jpg" if youtube_id else "",
        youtube_search_url=youtube_search_url,
    )
