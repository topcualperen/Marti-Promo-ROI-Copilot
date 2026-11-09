"""Sentetik Martı promosyon verisi üretir ve veritabanına yazar.

Her kullanıcının gizli bir "tipi" vardır (zaten gelecek / ikna edilebilir / kayıp /
rahatsız olan). Bu tip ve gerçek uplift yalnızca `hidden_truth` tablosuna yazılır;
model eğitiminde kullanılmaz, sadece doğrulama içindir.

Çalıştırma: python data/generate.py   (veri + uplift skorları + doğrulama grafiği)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SEED, get_engine  # noqa: E402

N_USERS = 50_000
SEHIRLER = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"]
SEHIR_PAYLARI = [0.45, 0.18, 0.15, 0.12, 0.10]
YAS_GRUPLARI = ["18-24", "25-34", "35-44", "45+"]
HIZMETLER = ["scooter", "moped", "TAG"]

# Gizli kullanıcı tipleri: (pay, indirimsiz yolculuk olasılığı, indirimin etkisi)
SEGMENTLER = {
    "zaten_gelecek": (0.30, 0.85, 0.02),
    "ikna_edilebilir": (0.30, 0.22, 0.30),
    "kayip": (0.32, 0.04, 0.01),
    "rahatsiz": (0.08, 0.45, -0.06),
}

# Kampanyalar: indirim kullanıldığında TL maliyeti, etki gücü, kime gönderildiği
KAMPANYALAR = [
    # id, ad, birim_maliyet_tl, guc, hedef_kitle
    (1, "Hoş geldin %50", 45, 1.1, "Yeni kullanıcılar (uygulama yaşı ≤ 90 gün)"),
    (2, "Yağmurlu akşam %20", 12, 0.5, "Rastgele %40 kullanıcı"),
    (3, "Hafta sonu 30 TL", 30, 0.8, "Rastgele %50 kullanıcı"),
    (4, "Geri dön %40", 35, 1.0, "Son 30 günde yolculuk yapmayanlar"),
    (5, "Scooter→TAG çapraz", 25, 0.9, "Scooter kullanıcıları"),
    (6, "Sadakat bonusu", 50, 0.6, "Sık kullananlar (son 30 günde ≥ 10 yolculuk)"),
]
# Bursa'da TAG kapsaması zayıf varsayıyoruz: çapraz kampanya orada işe yaramıyor
SEHIR_ETKI_ISTISNALARI = {(5, "Bursa"): 0.0}


def _users(rng: np.random.Generator, n: int) -> pd.DataFrame:
    names = list(SEGMENTLER)
    segment = rng.choice(names, size=n, p=[SEGMENTLER[s][0] for s in names])
    rides = np.select(
        [segment == "zaten_gelecek", segment == "ikna_edilebilir", segment == "kayip"],
        [6 + rng.poisson(14, n), rng.poisson(2.5, n), rng.poisson(0.2, n)],
        default=2 + rng.poisson(6, n),
    )
    yeni = rng.random(n) < 0.6
    app_age = np.select(
        [segment == "zaten_gelecek", segment == "ikna_edilebilir", segment == "kayip"],
        [
            rng.integers(200, 1100, n),
            np.where(yeni, rng.integers(3, 90, n), rng.integers(90, 400, n)),
            rng.integers(300, 1400, n),
        ],
        default=rng.integers(150, 1000, n),
    )
    yas_genel, yas_rahatsiz = [0.32, 0.36, 0.20, 0.12], [0.10, 0.20, 0.25, 0.45]
    yas = np.where(
        segment == "rahatsiz",
        rng.choice(YAS_GRUPLARI, n, p=yas_rahatsiz),
        rng.choice(YAS_GRUPLARI, n, p=yas_genel),
    )
    return pd.DataFrame({
        "user_id": np.arange(1, n + 1),
        "sehir": rng.choice(SEHIRLER, n, p=SEHIR_PAYLARI),
        "yas_grubu": yas,
        "son_30_gun_yolculuk": rides,
        "uygulama_yasi_gun": app_age,
        "kullanilan_hizmet": rng.choice(HIZMETLER, n, p=[0.55, 0.20, 0.25]),
        "_segment": segment,
    })


def _eligible(users: pd.DataFrame, campaign_id: int, rng: np.random.Generator) -> pd.DataFrame:
    """Kampanyanın hedef kitlesini döndürür (gerçek hayattaki hedefleme kuralları)."""
    rule = {
        1: users["uygulama_yasi_gun"] <= 90,
        2: rng.random(len(users)) < 0.40,
        3: rng.random(len(users)) < 0.50,
        4: users["son_30_gun_yolculuk"] == 0,
        5: users["kullanilan_hizmet"] == "scooter",
        6: users["son_30_gun_yolculuk"] >= 10,
    }[campaign_id]
    return users[rule]


def _experiment(users, campaign, rng) -> pd.DataFrame:
    """Hedef kitlenin %50'sine indirim verir, %50'si kontrol (holdout) grubu kalır."""
    cid, _, unit_cost, strength, _ = campaign
    pool = _eligible(users, cid, rng).copy()
    n = len(pool)
    base_p0 = pool["_segment"].map(lambda s: SEGMENTLER[s][1]).to_numpy()
    base_up = pool["_segment"].map(lambda s: SEGMENTLER[s][2]).to_numpy()
    city_factor = pool["sehir"].map(lambda c: SEHIR_ETKI_ISTISNALARI.get((cid, c), 1.0)).to_numpy()

    p0 = np.clip(base_p0 + rng.normal(0, 0.04, n), 0.01, 0.97)
    uplift = base_up * strength * city_factor + rng.normal(0, 0.02, n) * (base_up > 0.1)
    p1 = np.clip(p0 + uplift, 0.005, 0.99)

    treated = rng.random(n) < 0.5
    rode = rng.random(n) < np.where(treated, p1, p0)
    extra = rng.poisson(pool["son_30_gun_yolculuk"].to_numpy() / 4 + 0.3)
    return pd.DataFrame({
        "user_id": pool["user_id"].to_numpy(),
        "campaign_id": cid,
        "treated": treated.astype(int),
        # İndirim sadece kullanılırsa (yolculuk yapılırsa) maliyet doğurur
        "cost_tl": np.where(treated & rode, unit_cost, 0).astype(float),
        "rode": rode.astype(int),
        "rides": np.where(rode, 1 + extra, 0),
        "segment": pool["_segment"].to_numpy(),
        "p0": p0,
        "p1": p1,
        "true_uplift": p1 - p0,
    })


def generate(n_users: int = N_USERS, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Tüm tabloları DataFrame olarak üretir (aynı seed → aynı veri)."""
    rng = np.random.default_rng(seed)
    users = _users(rng, n_users)
    exp = pd.concat([_experiment(users, c, rng) for c in KAMPANYALAR], ignore_index=True)
    campaigns = pd.DataFrame(
        KAMPANYALAR, columns=["campaign_id", "kampanya_adi", "birim_maliyet_tl", "_guc", "hedef_kitle"]
    ).drop(columns="_guc")
    return {
        "users": users.drop(columns="_segment"),
        "campaigns": campaigns,
        "assignments": exp[["user_id", "campaign_id", "treated", "cost_tl"]],
        "outcomes": exp[["user_id", "campaign_id", "rode", "rides"]],
        "hidden_truth": exp[["user_id", "campaign_id", "segment", "p0", "p1", "true_uplift"]],
    }


def write_db(tables: dict[str, pd.DataFrame], engine=None) -> None:
    engine = engine or get_engine()
    for name, df in tables.items():
        df.to_sql(name, engine, if_exists="replace", index=False, chunksize=20_000)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows konsolunda Türkçe karakterler için
    tables = generate()
    write_db(tables)
    a = tables["assignments"]
    print(f"Kullanıcı: {len(tables['users']):,} | atama: {len(a):,} | "
          f"indirim alan: {a['treated'].sum():,} | harcama: {a['cost_tl'].sum():,.0f} TL")

    from model.uplift import train_and_score
    train_and_score()


if __name__ == "__main__":
    main()
