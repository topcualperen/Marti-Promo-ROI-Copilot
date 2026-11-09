"""Ortak ayarlar: veritabanı bağlantısı, sabitler, .env yükleme."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

SEED = 42
DEFAULT_DB_URL = f"sqlite:///{(ROOT / 'data' / 'promo.db').as_posix()}"

# Hikâyedeki kamuya açık rakamlar (demo anlatımı için; modelde kullanılmaz)
GERCEK_PROMOSYON_MUSD = 9.97
GERCEK_PROMOSYON_GELIR_PAYI = 0.28

# Ek yolculuk başına maliyet bu eşiği aşarsa kampanya yolculuk başına zarar ediyor sayılır
ORTALAMA_YOLCULUK_GELIRI_TL = 120


def db_url() -> str:
    """PostgreSQL'e geçmek için .env'de DATABASE_URL tanımlamak yeterli."""
    return os.getenv("DATABASE_URL", DEFAULT_DB_URL)


@lru_cache(maxsize=4)
def _engine_for(url: str) -> Engine:
    return create_engine(url)


def get_engine() -> Engine:
    return _engine_for(db_url())
