"""Kampanya verimliliği ve bütçe senaryoları.

- campaign_stats: kampanya bazında A/B (indirim vs. kontrol) ölçümü — dürüst, modelden bağımsız.
- compare: aynı bütçeyle "mevcut" (rastgele) vs. "akıllı" (uplift'e göre) dağıtım.

Senaryo değerlendirmesi: hangi kullanıcıya indirim gideceğine YALNIZCA model skorları karar
verir. Sonuç (ek yolculuk, harcama) ise sentetik dünyanın gizli gerçeğiyle ölçülür; gerçek
hayatta bu adımın yerini yeni bir holdout testi alır.
"""
import numpy as np
import pandas as pd

from config import ORTALAMA_YOLCULUK_GELIRI_TL, get_engine

Z95 = 1.96

AB_SQL = """
SELECT c.kampanya_adi, u.sehir, a.treated, a.cost_tl, o.rode
FROM assignments a
JOIN outcomes o ON o.user_id = a.user_id AND o.campaign_id = a.campaign_id
JOIN users u ON u.user_id = a.user_id
JOIN campaigns c ON c.campaign_id = a.campaign_id
"""

POOL_SQL = """
SELECT s.user_id, s.campaign_id, c.kampanya_adi, u.sehir, c.birim_maliyet_tl,
       s.p_treated, s.uplift, h.p1 AS true_p1, h.true_uplift
FROM uplift_scores s
JOIN campaigns c ON c.campaign_id = s.campaign_id
JOIN users u ON u.user_id = s.user_id
JOIN hidden_truth h ON h.user_id = s.user_id AND h.campaign_id = s.campaign_id
"""


def _ab_row(g: pd.DataFrame) -> pd.Series:
    tr, ct = g[g["treated"] == 1], g[g["treated"] == 0]
    r_t, r_c = tr["rode"].mean(), ct["rode"].mean()
    uplift = r_t - r_c
    se = np.sqrt(r_t * (1 - r_t) / len(tr) + r_c * (1 - r_c) / len(ct))
    spend = tr["cost_tl"].sum()
    extra = uplift * len(tr)
    return pd.Series({
        "kullanici": len(g),
        "indirim_alan": len(tr),
        "harcama_tl": spend,
        "yolculuk_orani_indirimli": r_t,
        "yolculuk_orani_kontrol": r_c,
        "uplift": uplift,
        "uplift_alt": uplift - Z95 * se,
        "uplift_ust": uplift + Z95 * se,
        "ek_yolculuk": extra,
        "ek_yolculuk_basi_maliyet_tl": spend / extra if extra > 0 else np.inf,
        # İndirimi kullananların ne kadarı indirimsiz de yolculuk yapacaktı?
        "bosa_giden_pay": min(r_c / r_t, 1.0) if r_t > 0 else 1.0,
    })


def campaign_stats(city: str | None = None, engine=None) -> pd.DataFrame:
    df = pd.read_sql(AB_SQL, engine or get_engine())
    if city:
        df = df[df["sehir"].str.casefold() == city.casefold()]
    out = df.groupby("kampanya_adi").apply(_ab_row, include_groups=False).reset_index()
    out["anlamli_etki"] = out["uplift_alt"] > 0
    out["verimsiz"] = ~out["anlamli_etki"] | (out["ek_yolculuk_basi_maliyet_tl"] > ORTALAMA_YOLCULUK_GELIRI_TL)
    return out.sort_values("ek_yolculuk_basi_maliyet_tl").reset_index(drop=True)


def overall_waste_share(stats: pd.DataFrame) -> float:
    """Toplam harcamanın, zaten yolculuk yapacak kullanıcılara giden tahmini payı."""
    return float((stats["harcama_tl"] * stats["bosa_giden_pay"]).sum() / stats["harcama_tl"].sum())


def load_pool(city: str | None = None, engine=None) -> pd.DataFrame:
    """İndirim gönderilebilecek tüm (kullanıcı, kampanya) çiftleri."""
    pool = pd.read_sql(POOL_SQL, engine or get_engine())
    if city:
        pool = pool[pool["sehir"].str.casefold() == city.casefold()]
    pool["tahmini_maliyet"] = pool["birim_maliyet_tl"] * pool["p_treated"]
    pool["gercek_maliyet"] = pool["birim_maliyet_tl"] * pool["true_p1"]
    return pool.reset_index(drop=True)


def smart_order(pool: pd.DataFrame, max_cost_per_ride: float = ORTALAMA_YOLCULUK_GELIRI_TL) -> pd.DataFrame:
    """Her TL'nin en çok ek yolculuk getireceği sıra: tahmini uplift / tahmini maliyet.

    Tahmini ek yolculuk başına maliyeti yolculuk gelirini aşan çiftlere indirim gönderilmez;
    bu yüzden akıllı senaryo bütçenin tamamını harcamayabilir.
    """
    p = pool[pool["uplift"] > 0].copy()
    p["verim"] = p["uplift"] / p["tahmini_maliyet"].clip(lower=1.0)
    p = p[p["verim"] >= 1 / max_cost_per_ride]
    return p.sort_values("verim", ascending=False).reset_index(drop=True)


def random_scenario(pool: pd.DataFrame, budget: float) -> dict:
    """Rastgele dağıtımın beklenen sonucu: bütçe, havuzun ortalama verimiyle harcanır."""
    total_cost, total_gain = pool["gercek_maliyet"].sum(), pool["true_uplift"].sum()
    spend = min(budget, total_cost)
    extra = total_gain * spend / total_cost if total_cost else 0.0
    return _summary(spend, extra, n_users=round(len(pool) * spend / total_cost) if total_cost else 0)


def smart_scenario(ordered: pd.DataFrame, budget: float) -> tuple[dict, pd.DataFrame]:
    chosen = ordered[ordered["gercek_maliyet"].cumsum() <= budget]
    return _summary(chosen["gercek_maliyet"].sum(), chosen["true_uplift"].sum(), len(chosen)), chosen


def budget_to_match(ordered: pd.DataFrame, target_extra: float) -> float | None:
    """Akıllı sıralamayla `target_extra` ek yolculuğa ulaşmak için gereken bütçe."""
    cum_gain = np.maximum.accumulate(ordered["true_uplift"].cumsum().to_numpy())
    k = int(np.searchsorted(cum_gain, target_extra))
    if k >= len(ordered):
        return None
    return float(ordered["gercek_maliyet"].iloc[: k + 1].sum())


def _summary(spend: float, extra: float, n_users: int) -> dict:
    return {
        "harcanan_tl": round(float(spend)),
        "ek_yolculuk": round(float(extra)),
        "ek_yolculuk_basi_maliyet_tl": round(spend / extra, 1) if extra > 0 else None,
        "indirim_gonderilen": int(n_users),
    }


def compare(budget: float | None = None, city: str | None = None, engine=None) -> dict:
    """Aynı bütçeyle mevcut vs. akıllı senaryo + aynı sonuç için gereken bütçe tasarrufu."""
    pool = load_pool(city, engine)
    if pool.empty:
        raise ValueError(f"'{city}' için veri bulunamadı.")
    if budget is None:  # varsayılan: gerçekte harcanan toplam tutar
        budget = float(campaign_stats(city, engine)["harcama_tl"].sum())
    ordered = smart_order(pool)
    current = random_scenario(pool, budget)
    smart, chosen = smart_scenario(ordered, budget)
    needed = budget_to_match(ordered, current["ek_yolculuk"])
    smart["kullanilmayan_butce_tl"] = round(budget - smart["harcanan_tl"])
    return {
        "butce_tl": round(budget),
        "sehir": city or "Tümü",
        "mevcut_rastgele": current,
        "akilli_uplift": smart,
        "ayni_sonuc_icin_gereken_butce_tl": round(needed) if needed else None,
        "tasarruf_orani": round(1 - needed / current["harcanan_tl"], 3) if needed and current["harcanan_tl"] else None,
        "kampanya_dagilimi": _allocation(chosen),
        "dagitima_girmeyen_kampanyalar": sorted(set(pool["kampanya_adi"]) - set(chosen["kampanya_adi"])),
        "yontem": (
            "Kampanya ortalamaları değil, tek tek kullanıcı-kampanya çiftleri seçilir: tahmini ek yolculuk ÷ "
            "tahmini maliyet oranı en yüksek olandan başlanır. Aynı ikna edilebilir kullanıcıya daha ucuz bir "
            "indirim yetiyorsa pahalı kampanya seçilmez; bu yüzden ortalamada verimli bir kampanya dağıtımda "
            "hiç yer almayabilir. Ek yolculuk başına tahmini maliyeti "
            f"{ORTALAMA_YOLCULUK_GELIRI_TL} TL'yi aşan çiftlere indirim gönderilmez."
        ),
    }


def _allocation(chosen: pd.DataFrame) -> list[dict]:
    g = chosen.groupby("kampanya_adi").agg(
        indirim_gonderilen=("user_id", "size"),
        harcanan_tl=("gercek_maliyet", "sum"),
        ek_yolculuk=("true_uplift", "sum"),
    ).sort_values("harcanan_tl", ascending=False)
    g["harcanan_tl"] = g["harcanan_tl"].round()
    g["ek_yolculuk"] = g["ek_yolculuk"].round(1)
    return g.reset_index().to_dict("records")
