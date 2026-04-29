"""
Backend video proxy — streams YouTube content through our server.

Uses yt-dlp + ffmpeg to download, merge, and serve YouTube videos
at multiple quality levels (360p, 480p, 720p, 1080p).

YouTube serves higher resolutions as separate video+audio DASH streams.
yt-dlp + ffmpeg merge them into a single mp4 file, which we serve via
FileResponse with full Range request support.
"""
import logging
import asyncio
import subprocess
import os
import sys
import threading
from pathlib import Path

import imageio_ffmpeg
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from src.backend.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["video"])

# Directory to cache downloaded videos
_CACHE_DIR = Path(settings.processed_dir) / "_video_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ffmpeg binary from imageio-ffmpeg
_FFMPEG_PATH = str(imageio_ffmpeg.get_ffmpeg_exe())

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
_DOWNLOAD_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_MIN_PLAYABLE_BYTES = 1024 * 1024


def _get_download_lock(video_id: str, quality: str) -> threading.Lock:
    key = (video_id, quality)
    with _LOCKS_GUARD:
        if key not in _DOWNLOAD_LOCKS:
            _DOWNLOAD_LOCKS[key] = threading.Lock()
        return _DOWNLOAD_LOCKS[key]


def _get_cache_path(video_id: str, quality: str) -> Path:
    """Get the filesystem path for a cached video."""
    return _CACHE_DIR / f"{video_id}_{quality}.mp4"


def _is_cached_video_ready(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= _MIN_PLAYABLE_BYTES


def _download_video(video_id: str, quality: str = "720") -> Path:
    """
    Download and merge a YouTube video at the specified quality.

    Uses yt-dlp with ffmpeg to merge separate DASH video+audio streams
    into a single mp4 file. Result is cached on disk.
    """
    cache_path = _get_cache_path(video_id, quality)
    lock = _get_download_lock(video_id, quality)

    with lock:
        # Another request may have completed the same download while we waited.
        if _is_cached_video_ready(cache_path):
            logger.info("Cache hit: %s [q=%s]", video_id, quality)
            return cache_path
        if cache_path.exists():
            cache_path.unlink()

        url = f"https://www.youtube.com/watch?v={video_id}"
        fmt = QUALITY_FORMATS.get(quality, QUALITY_FORMATS["720"])

        logger.info("Downloading %s at quality=%s ...", video_id, quality)

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "yt_dlp",
                    "--ffmpeg-location", _FFMPEG_PATH,
                    "-f", fmt,
                    "--merge-output-format", "mp4",
                    "--no-playlist",
                    "--no-warnings",
                    "--force-ipv4",
                    "--retries", "5",
                    "--fragment-retries", "5",
                    "--concurrent-fragments", "4",
                    "--socket-timeout", "30",
                    "--extractor-args", "youtube:player_client=android,web",
                    "--user-agent",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    "--referer", "https://www.youtube.com/",
                    "-o", str(cache_path),
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                logger.error("yt-dlp download failed: %s", result.stderr[-1200:])
                if cache_path.exists():
                    cache_path.unlink()
                raise HTTPException(502, "Could not download video with yt-dlp")

            if not _is_cached_video_ready(cache_path):
                raise HTTPException(502, "Download produced no playable file")

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
                sys.executable,
                "-m",
                "yt_dlp",
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
    if not _is_cached_video_ready(cache_path):
        cache_path = await asyncio.to_thread(_download_video, video_id, quality)

    # Serve the file — FileResponse handles Range requests automatically
    file_size = os.path.getsize(str(cache_path))
    range_header = request.headers.get("range", 0)
    headers = {
        "content-type": "video/mp4",
        "accept-ranges": "bytes",
        "content-encoding": "identity",
        "content-length": str(file_size),
        "access-control-expose-headers": (
            "content-type, accept-ranges, content-length, content-range, content-encoding"
        ),
    }
    start = 0
    end = file_size - 1
    status_code = 200

    if range_header:
        range_match = str(range_header).strip().lower().replace("bytes=", "")
        range_parts = range_match.split("-")
        try:
            if range_parts[0]:
                start = int(range_parts[0])
            if len(range_parts) > 1 and range_parts[1]:
                end = min(int(range_parts[1]), file_size - 1)
        except ValueError:
            pass

        if start >= file_size:
            from fastapi import Response
            return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

        status_code = 206
        headers["content-length"] = str(end - start + 1)
        headers["content-range"] = f"bytes {start}-{end}/{file_size}"

    def chunk_generator():
        with open(str(cache_path), "rb") as f:
            f.seek(start)
            bytes_remaining = end - start + 1
            chunk_size = 1024 * 1024
            while bytes_remaining > 0:
                read_size = min(chunk_size, bytes_remaining)
                data = f.read(read_size)
                if not data:
                    break
                yield data
                bytes_remaining -= len(data)

    return StreamingResponse(
        chunk_generator(), headers=headers, status_code=status_code
    )


@router.post("/video/prepare/{video_id}")
async def prepare_video(
    video_id: str,
    q: str = Query("720", description="Quality: 360, 480, 720, 1080, best"),
):
    """Download and cache a YouTube video before the browser mounts <video>."""
    quality = q if q in QUALITY_FORMATS else "720"
    cache_path = await asyncio.to_thread(_download_video, video_id, quality)
    size = cache_path.stat().st_size if cache_path.exists() else 0
    return {
        "video_id": video_id,
        "quality": quality,
        "cached": size > 0,
        "stream_url": f"/api/video/stream/{video_id}?q={quality}",
        "size_bytes": size,
        "size_mb": round(size / (1024 * 1024), 1) if size else 0,
    }



@router.get("/video/download-status/{video_id}")
async def download_status(
    video_id: str,
    q: str = Query("720"),
):
    """Check if a video is already cached at a given quality."""
    quality = q if q in QUALITY_FORMATS else "720"
    cache_path = _get_cache_path(video_id, quality)
    cached = _is_cached_video_ready(cache_path)
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
        cached = _is_cached_video_ready(cache_path)
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
