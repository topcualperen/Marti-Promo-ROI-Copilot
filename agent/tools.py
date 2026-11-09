"""Ajanın kullandığı 3 araç. Her araç JSON metni döner (LLM için kolay okunur)."""
import json
import re
import unicodedata

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sqlalchemy import text

from config import get_engine
from model.budget import campaign_stats, compare

MAX_ROWS = 200
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma|vacuum|"
    r"reindex|grant|revoke|copy|merge|call|exec|execute)\b",
    re.IGNORECASE,
)
HIDDEN_TABLES = re.compile(r"\bhidden_truth\b", re.IGNORECASE)


class UnsafeQueryError(ValueError):
    pass


def validate_select(query: str) -> str:
    """Sadece tek bir SELECT (veya WITH ... SELECT) sorgusuna izin verir."""
    q = query.strip().rstrip(";").strip()
    if not re.match(r"^(select|with)\b", q, re.IGNORECASE):
        raise UnsafeQueryError("Yalnızca SELECT sorgularına izin veriliyor.")
    if ";" in q:
        raise UnsafeQueryError("Tek seferde yalnızca bir sorgu çalıştırılabilir.")
    if m := FORBIDDEN.search(q):
        raise UnsafeQueryError(f"İzin verilmeyen komut: {m.group(0).upper()}")
    if HIDDEN_TABLES.search(q):
        raise UnsafeQueryError("Bu tablo doğrulama içindir, sorgulanamaz.")
    return q


def run_select(query: str) -> pd.DataFrame:
    q = validate_select(query)
    with get_engine().connect() as conn:
        result = conn.execute(text(q))
        rows = result.fetchmany(MAX_ROWS)
        conn.rollback()  # ek güvenlik: hiçbir değişiklik kalıcı olmaz
    return pd.DataFrame(rows, columns=list(result.keys()))


def _to_json(obj) -> str:
    def default(o):
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        return str(o)
    return json.dumps(obj, ensure_ascii=False, default=default)


def _normalize(s: str) -> str:
    s = s.casefold().replace("ı", "i")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if c.isalnum())


def _match_campaign(name: str, names: list[str]) -> str | None:
    key = _normalize(name)
    for n in names:
        if key and (key in _normalize(n) or _normalize(n) in key):
            return n
    # "hoş geldin kampanyası" gibi serbest ifadeler için kelime bazında eşleşme
    words = [_normalize(w) for w in name.split() if len(w) > 2]
    scored = [(sum(w in _normalize(n) for w in words), n) for n in names]
    best = max(scored)
    return best[1] if best[0] > 0 else None


def _round(d: dict) -> dict:
    return {k: (round(v, 3) if isinstance(v, float) and np.isfinite(v) else
                (None if isinstance(v, float) else v)) for k, v in d.items()}


@tool
def sql_query(query: str) -> str:
    """Veritabanında salt-okunur bir SELECT sorgusu çalıştırır (en fazla 200 satır döner).
    Hazır araçların cevaplamadığı sorular için kullan."""
    try:
        df = run_select(query)
    except UnsafeQueryError as e:
        return _to_json({"hata": str(e)})
    except Exception as e:  # SQL söz dizimi hatası vb. → ajan sorgusunu düzeltebilsin
        return _to_json({"hata": f"Sorgu çalıştırılamadı: {e}"})
    return _to_json({"satir_sayisi": len(df), "limit": MAX_ROWS, "veri": df.to_dict("records")})


@tool
def campaign_uplift(campaign_name: str, city: str | None = None) -> str:
    """Bir kampanyanın A/B testine göre gerçek etkisini döner: indirimli ve kontrol grubunun
    yolculuk oranı, uplift (ve %95 güven aralığı), harcama, ek yolculuk, ek yolculuk başına
    maliyet, boşa giden pay. campaign_name='tümü' verilirse tüm kampanyaları döner.
    city: İstanbul, Ankara, İzmir, Bursa, Antalya (opsiyonel)."""
    stats = campaign_stats(city)
    if stats.empty:
        return _to_json({"hata": f"'{city}' için veri bulunamadı."})
    if _normalize(campaign_name) in {"tumu", "hepsi", "all", "tum"}:
        rows = stats
    else:
        name = _match_campaign(campaign_name, stats["kampanya_adi"].tolist())
        if name is None:
            return _to_json({"hata": "Kampanya bulunamadı.", "mevcut_kampanyalar": stats["kampanya_adi"].tolist()})
        rows = stats[stats["kampanya_adi"] == name]
    return _to_json({"sehir": city or "Tümü", "kampanyalar": [_round(r) for r in rows.to_dict("records")]})


@tool
def budget_simulator(budget_tl: float, city: str | None = None) -> str:
    """Verilen bütçeyi uplift skoruna göre en verimli kullanıcı-kampanya çiftlerine dağıtır ve
    aynı bütçenin rastgele dağıtımıyla karşılaştırır. Ek yolculuk, ek yolculuk başına maliyet,
    aynı sonuç için gereken bütçe ve kampanya bazında dağılımı döner."""
    try:
        return _to_json(compare(budget=float(budget_tl), city=city))
    except ValueError as e:
        return _to_json({"hata": str(e)})


TOOLS = [sql_query, campaign_uplift, budget_simulator]
