from fastapi import FastAPI

from backend.app.api.burned_areas import router as burned_areas_router
from backend.app.api.dnbr import router as dnbr_router
from backend.app.api.events import router as events_router
from backend.app.api.hotspots import router as hotspots_router
from backend.app.api.safe import router as safe_router
from backend.app.api.forest_impact import router as forest_impact_router


app = FastAPI(title="Forest Fire Monitoring API", version="0.1.0")
app.include_router(hotspots_router)
app.include_router(events_router)
app.include_router(burned_areas_router)
app.include_router(dnbr_router)
app.include_router(forest_impact_router)
app.include_router(safe_router)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
