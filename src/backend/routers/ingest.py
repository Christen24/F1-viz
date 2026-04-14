from fastapi import APIRouter, BackgroundTasks, HTTPException
# Triggering reload
from typing import Optional
import logging
import sys
import os

# Add the scripts directory to path to allow importing run_ingest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

router = APIRouter(prefix="/api/ingest", tags=["ingest"])
logger = logging.getLogger(__name__)

@router.post("/category/{category}")
async def ingest_category(category: str, background_tasks: BackgroundTasks):
    """
    Trigger ingestion for a specific category (e.g., 'driver', 'team').
    Runs as a background task.
    """
    if category not in ["driver", "team", "circuit", "regulation", "race_report"]:
        raise HTTPException(status_code=400, detail=f"Invalid category: {category}")

    # Import lazily so normal API/test startup does not require tokenizer downloads.
    from scripts.rag.ingest.run_ingest import run_pipeline, load_sources_file

    sources = load_sources_file()
    filtered_sources = [s for s in sources if s.get("category") == category]

    if not filtered_sources:
        raise HTTPException(status_code=404, detail=f"No sources found for category: {category}")

    background_tasks.add_task(run_pipeline, filtered_sources)

    return {
        "status": "ingestion_started",
        "category": category,
        "source_count": len(filtered_sources)
    }
