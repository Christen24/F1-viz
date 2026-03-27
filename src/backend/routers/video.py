"""
Backend video proxy — streams YouTube content through our server.

Uses yt-dlp + ffmpeg to download, merge, and serve YouTube videos
at multiple quality levels (360p, 480p, 720p, 1080p).

YouTube serves higher resolutions as separate video+audio DASH streams.
yt-dlp + ffmpeg merge them into a single mp4 file, which we serve via
FileResponse with full Range request support.
"""
import logging
import subprocess
import hashlib
import os
from pathlib import Path

import imageio_ffmpeg
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from src.backend.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["video"])

# Directory to cache downloaded videos
_CACHE_DIR = Path(settings.processed_dir) / "_video_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ffmpeg binary from imageio-ffmpeg
_FFMPEG_PATH = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)

# Quality presets — yt-dlp format selectors (merging video+audio)
QUALITY_FORMATS = {
    "360":  "bestvideo[height<=360][ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/best[height<=360][ext=mp4]/best[height<=360]",
    "480":  "bestvideo[height<=480][ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[height<=480]",
    "720":  "bestvideo[height<=720][ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
    "1080": "bestvideo[height<=1080][ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best[height<=1080]",
    "best": "bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/best[ext=mp4]/best",
}

# In-memory metadata cache
_META_CACHE: dict[tuple[str, str], dict] = {}


def _get_cache_path(video_id: str, quality: str) -> Path:
    """Get the filesystem path for a cached video."""
    return _CACHE_DIR / f"{video_id}_{quality}.mp4"


def _download_video(video_id: str, quality: str = "720") -> Path:
    """
    Download and merge a YouTube video at the specified quality.

    Uses yt-dlp with ffmpeg to merge separate DASH video+audio streams
    into a single mp4 file. Result is cached on disk.
    """
    cache_path = _get_cache_path(video_id, quality)

    # Already downloaded?
    if cache_path.exists() and cache_path.stat().st_size > 0:
        logger.info("Cache hit: %s [q=%s]", video_id, quality)
        return cache_path

    url = f"https://www.youtube.com/watch?v={video_id}"
    fmt = QUALITY_FORMATS.get(quality, QUALITY_FORMATS["720"])

    logger.info("Downloading %s at quality=%s ...", video_id, quality)

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--ffmpeg-location", _FFMPEG_PATH,
                "-f", fmt,
                "--merge-output-format", "mp4",
                "--no-playlist",
                "--no-warnings",
                "-o", str(cache_path),
                url,
            ],
            capture_output=True,
            text=True,
            timeout=120,  # Generous timeout for download+merge
        )

        if result.returncode != 0:
            logger.error("yt-dlp download failed: %s", result.stderr[:500])
            # Clean up partial file
            if cache_path.exists():
                cache_path.unlink()
            raise HTTPException(502, "Could not download video")

        if not cache_path.exists() or cache_path.stat().st_size == 0:
            raise HTTPException(502, "Download produced no file")

        size_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info("Downloaded %s [q=%s]: %.1f MB", video_id, quality, size_mb)
        return cache_path

    except subprocess.TimeoutExpired:
        if cache_path.exists():
            cache_path.unlink()
        raise HTTPException(504, "Download timed out")


def _get_video_meta(video_id: str) -> dict:
    """Get video metadata without downloading."""
    cache_key = (video_id, "meta")
    if cache_key in _META_CACHE:
        return _META_CACHE[cache_key]

    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--no-download",
                "--print-json",
                "-f", "best[ext=mp4]/best",
                "--no-playlist",
                "--no-warnings",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            meta = {
                "title": info.get("title", ""),
                "duration": info.get("duration", 0),
            }
            _META_CACHE[cache_key] = meta
            return meta
    except Exception:
        pass
    return {"title": "", "duration": 0}


@router.get("/video/stream/{video_id}")
async def stream_video(
    video_id: str,
    request: Request,
    q: str = Query("720", description="Quality: 360, 480, 720, 1080, best"),
):
    """
    Download (if needed) and serve a YouTube video at the specified quality.

    Uses FileResponse for full Range request support.
    First request for a new quality may take 10-30 seconds to download.
    Subsequent requests serve from disk cache instantly.
    """
    quality = q if q in QUALITY_FORMATS else "720"

    cache_path = _get_cache_path(video_id, quality)

    # Download if not cached
    if not cache_path.exists() or cache_path.stat().st_size == 0:
        _download_video(video_id, quality)

    # Serve the file — FileResponse handles Range requests automatically
    return FileResponse(
        path=str(cache_path),
        media_type="video/mp4",
        filename=f"{video_id}_{quality}.mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/video/download-status/{video_id}")
async def download_status(
    video_id: str,
    q: str = Query("720"),
):
    """Check if a video is already cached at a given quality."""
    quality = q if q in QUALITY_FORMATS else "720"
    cache_path = _get_cache_path(video_id, quality)
    cached = cache_path.exists() and cache_path.stat().st_size > 0
    size = cache_path.stat().st_size if cached else 0
    return {
        "video_id": video_id,
        "quality": quality,
        "cached": cached,
        "size_bytes": size,
        "size_mb": round(size / (1024 * 1024), 1) if size else 0,
    }


@router.get("/video/qualities/{video_id}")
async def video_qualities(video_id: str):
    """Return available quality options and their cache status."""
    qualities = []
    for val, label, desc in [
        ("360", "360p", "Low — fast"),
        ("480", "480p", "Medium"),
        ("720", "720p", "HD"),
        ("1080", "1080p", "Full HD"),
        ("best", "Best", "Highest"),
    ]:
        cache_path = _get_cache_path(video_id, val)
        cached = cache_path.exists() and cache_path.stat().st_size > 0
        qualities.append({
            "value": val,
            "label": label,
            "description": desc,
            "cached": cached,
        })
    return {"video_id": video_id, "qualities": qualities, "default": "720"}


@router.get("/video/info/{video_id}")
async def video_info(video_id: str):
    """Get metadata about a video."""
    meta = _get_video_meta(video_id)
    return {
        "video_id": video_id,
        "title": meta["title"],
        "duration": meta["duration"],
        "ready": True,
    }
