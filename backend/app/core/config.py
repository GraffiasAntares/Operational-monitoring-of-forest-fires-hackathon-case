from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    firms_map_key: str | None
    firms_base_url: str = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    copernicus_client_id: str | None = None
    copernicus_client_secret: str | None = None
    copernicus_token_url: str = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    )
    copernicus_process_url: str = "https://sh.dataspace.copernicus.eu/api/v1/process"
    copernicus_catalog_url: str = "https://sh.dataspace.copernicus.eu/catalog/v1/search"
    safe_model_path: str | None = None
    bs_model_path: str | None = None
    bs_preprocessing_path: str | None = None
    request_timeout_seconds: float = 20.0


def _load_local_env_file() -> None:
    """Load simple KEY=VALUE pairs without overwriting real environment variables."""
    env_file = Path(__file__).resolve().parents[3] / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_local_env_file()
    return Settings(
        firms_map_key=os.getenv("FIRMS_MAP_KEY"),
        copernicus_client_id=os.getenv("COPERNICUS_CLIENT_ID"),
        copernicus_client_secret=os.getenv("COPERNICUS_CLIENT_SECRET"),
        safe_model_path=os.getenv("SAFE_MODEL_PATH"),
        bs_model_path=os.getenv("BS_MODEL_PATH") or str(Path(__file__).resolve().parents[3] / "ml/artifacts/bs_unet_full_tuned_seed42/best.pt"),
        bs_preprocessing_path=os.getenv("BS_PREPROCESSING_PATH") or str(Path(__file__).resolve().parents[3] / "ml/artifacts/bs_preprocessing_seed42.npz"),
    )
