from pathlib import Path
from fastapi import APIRouter, Depends
from backend.app.core.config import Settings, get_settings

router = APIRouter(prefix="/api/v1", tags=["SAFE"])

@router.get("/safe/status")
async def safe_status(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    path = Path(settings.safe_model_path) if settings.safe_model_path else None
    return {"model":"SAFE", "ready": bool(path and path.exists()), "model_path_configured": bool(path), "reason": None if path and path.exists() else "SAFE checkpoint/STAMP weights are not supplied by the referenced repository; configure SAFE_MODEL_PATH after obtaining compatible weights."}
